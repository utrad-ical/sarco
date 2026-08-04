


import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import SimpleITK as sitk
import scipy.ndimage

import torch
import torch.nn
import torch.optim as optim
import unet

import torchvision.transforms

import readfiles
import volumeresampler
import extract_trunk




DEFAULT_ORIGINAL_SLICE_THICKNESS = 5.0            # mm
INPUT_VOXEL_SIZE=1      # mm, input for Unet
INPUT_SLICE_THICKNESS=1.666666666667    # mm
INPUT_X_SIZE = 512      # voxels
INPUT_Z_SIZE = 512      # voxels
NUM_FEATURE_IMAGE = 4
NUM_LABEL_IMAGE = 2



def dice_loss(outputs, labels, eps = 1.):
    n_batch = outputs.size()[0]

    pred = outputs.view([n_batch, -1])
    target = labels.view([n_batch, -1])

    intersection = torch.sum(pred*target, axis=1)
    loss = 1. - (2. * intersection) / (torch.sum(pred, axis=1) + torch.sum(target, axis=1) + eps)

    return torch.sum(loss)



def data_preparation(volume_list : str, labelused, data_augmentation_multiplier):

    i = 0
    training_gray = np.empty([0,NUM_FEATURE_IMAGE,INPUT_Z_SIZE,INPUT_X_SIZE], dtype='float32')
    training_label = np.empty([0,2,INPUT_Z_SIZE,INPUT_X_SIZE], dtype='float32')
    zs = []
    ztrues = []
    original_slice_thicknesses = []

    # malloc
    for N in readfiles.counter(volume_list):
        pass
    N=N+1
    training_gray = np.zeros([N*data_augmentation_multiplier,NUM_FEATURE_IMAGE,INPUT_Z_SIZE,INPUT_X_SIZE], dtype='float32')
    training_label = np.zeros([N*data_augmentation_multiplier,2,INPUT_Z_SIZE,INPUT_X_SIZE], dtype='float32')
    
    # main loop
    for n, (grayImage, labelImage) in enumerate(readfiles.reader(volume_list)):

        print(f"sample #{n}/{N}: ", end="", flush=True)

        # get original voxel size
        original_slice_thickness = labelImage.GetSpacing()[2]
        original_slice_thicknesses.append(original_slice_thickness)

        # true label
        lorig = sitk.GetArrayFromImage(labelImage)
        
        sumlabel = np.sum(np.sum(lorig == labelused, axis=2), axis=1)
        if np.sum(sumlabel) == 0:
            # no answer
            ztrue = -1
        else:
            ztrue = np.argmax(sumlabel)
            
        # first resize to (1mm,1mm,1.6666667mm)
        print("resampling...", end="", flush=True)
        new_voxel_size = [INPUT_VOXEL_SIZE, INPUT_VOXEL_SIZE, INPUT_SLICE_THICKNESS]
        new_matrix_size_elongated = [INPUT_X_SIZE, INPUT_X_SIZE, INPUT_Z_SIZE*2]

        # get an enlarged volume to allow for data augmentation
        grayImage = volumeresampler.resample(grayImage, -50*original_slice_thickness, new_voxel_size, new_matrix_size_elongated, is_label = False, apply_filter = True)
        labelImage = volumeresampler.resample(labelImage, -50*original_slice_thickness, new_voxel_size, new_matrix_size_elongated, is_label = True)

        print("arm killing...", end="", flush=True)
        grayImage = extract_trunk.kill_arms(grayImage)

        print("data augmenting...", end="", flush=True)

        for index in range(data_augmentation_multiplier):
        
            zbase = np.random.randint(-50, 50) * DEFAULT_ORIGINAL_SLICE_THICKNESS
            if index==0:
                zbase = 0 # use original
            if ztrue >= 0:
                # shift the answer along z-axis
                ztrue = ztrue - int(zbase / original_slice_thickness)
                ztrues.append(ztrue)
        
            # shift the image along z-axis
            zshift = int(np.floor((zbase + 50*DEFAULT_ORIGINAL_SLICE_THICKNESS) / INPUT_SLICE_THICKNESS))

            gRarr = sitk.GetArrayFromImage(grayImage)[zshift:zshift+INPUT_Z_SIZE,:,:]
            lRarr = sitk.GetArrayFromImage(labelImage)[zshift:zshift+INPUT_Z_SIZE,:,:]

            # modify grayscale and label
            gRarr[gRarr<100] = 100
            gRarr[1000<gRarr] = 1000
            lRarr = (lRarr == labelused).astype(float)

            # for violation...
            while len(gRarr.shape) >= 4:
                gRarr = np.mean(gRarr, axis=3)

            # projection along x- and y-axes
            g = np.zeros((1,NUM_FEATURE_IMAGE,INPUT_Z_SIZE,INPUT_X_SIZE), dtype='float32')
            g[0,0,:,:] = np.mean(gRarr, axis=2)
            g[0,1,:,:] = np.mean(gRarr, axis=1)
            g[0,2,:,:] = np.max(gRarr, axis=2)
            g[0,3,:,:] = np.max(gRarr, axis=1)

            l = np.zeros((1,2,INPUT_Z_SIZE,INPUT_X_SIZE))
            l[0,0,:,:] = np.max(lRarr, axis=2)
            l[0,1,:,:] = np.max(lRarr, axis=1)

#            for ii in range(NUM_FEATURE_IMAGE):
#                plt.imsave("feat_%d_%d.png"%(i,ii), g[0,ii,:,:])
#            for ii in range(NUM_LABEL_IMAGE):
#                plt.imsave("labl_%d_%d.png"%(i,ii), l[0,ii,:,:])

            #training_gray = np.concatenate((training_gray, g), axis=0)
            training_gray[i,:,:,:] = g
        
            # 2 label images

            #training_label = np.concatenate((training_label, l), axis=0)
            training_label[i,:,:,:] = l

            i+=1
    
        print("done.")

    return training_gray, training_label, zs, ztrues, original_slice_thicknesses




def main(mode : str, 
         volume_list : str, 
         batch_size, 
         device : str, 
         learning_rate : float,
         beta_1 : float, 
         model_file_name : str,
         output_file_name : str,
         n_epoch : int,
         labelused : int,
         data_augmentation_multiplier : int):
    
    # build a model
    model = unet.Unet(NUM_FEATURE_IMAGE,NUM_LABEL_IMAGE,16)
    
    
    
    if(mode == 'test'):
        model.load_state_dict(torch.load(model_file_name))
    
    
    
    # apply data parallel
    model = torch.nn.DataParallel(model)

    model = model.to(device)



    if(mode == 'train'):
        
        print("data preparation starting...")
        # data preparation
        (training_gray, training_label, zs, ztrues, original_slice_thicknesses) = \
                                                      data_preparation(volume_list, 
                                                                       labelused,
                                                                       data_augmentation_multiplier)
        
        
        
        # data loader
        
        tensor_x = torch.from_numpy(training_gray)
        tensor_y = torch.from_numpy(training_label)
        
        training_dataset = torch.utils.data.TensorDataset(tensor_x.type(torch.FloatTensor),
                                                          tensor_y.type(torch.FloatTensor))
        training_loader = torch.utils.data.DataLoader(training_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=True,
                                                  num_workers=0)



        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta_1, 0.999))



        # augmenter
        transform = torchvision.transforms.RandomErasing(p=0.5).to(device)



        for epoch in range(n_epoch):

            print("epoch "+str(epoch))

            model.train()
            
            sum0 = 0
            sum1 = 0
            for batch_idx, (data, labels) in enumerate(training_loader):
            
                optimizer.zero_grad()
            
                data, labels = data.to(device), labels.to(device)
                
                data = transform(data)
                
                outputs = model(data)
                
                outputs = torch.sigmoid(outputs)
                
                loss = dice_loss(outputs, labels)
                loss = torch.mean(loss)
                
                loss.backward()
                optimizer.step()
                
                sum0 += data.shape[0]
                sum1 += loss.detach().cpu().numpy()
            
            print("%f" % (sum1/sum0))

        if 0:
            res0 = np.squeeze(outputs.detach().cpu().numpy()[0,0,:,:])
            plt.imsave(r'res0.png', res0)
            res1 = np.squeeze(outputs.detach().cpu().numpy()[0,1,:,:])
            plt.imsave(r'res1.png', res1)
            lab0 = np.squeeze(labels.detach().cpu().numpy()[0,0,:,:])
            plt.imsave(r'lab0.png', lab0)
            lab1 = np.squeeze(labels.detach().cpu().numpy())[0,1,:,:]
            plt.imsave(r'lab1.png', lab1)

        torch.save(model.module.state_dict(), model_file_name)



    elif (mode == 'test'):
    
        test_gray, test_label, zs, ztrues, original_slice_thicknesses = \
                                            data_preparation(volume_list, 
                                                             labelused,
                                                             data_augmentation_multiplier=1)

        print("test_gray.shape=", test_gray.shape)
        
        tensor_x = torch.from_numpy(test_gray)
        tensor_y = torch.from_numpy(test_label)
        
        test_dataset = torch.utils.data.TensorDataset(tensor_x.type(torch.FloatTensor),
                                                      tensor_y.type(torch.FloatTensor))
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        zestimates = []
        likelihoods = []

        likelihood_matrix = np.zeros((8, test_label.shape[0]))

        base_index = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_loader):

                print("batch #%d" % batch_idx)

                data, labels = data.to(device), labels.to(device)
                
                res = model(data)
                
                res = torch.sigmoid(res)
                
                for i in range(res.size()[0]):
                    case_idx = base_index + i
                    
                    gyimage = (data[i,0:4,:,:]).detach().cpu().numpy()
                    lkimage = (res[i,0:2,:,:]).detach().cpu().numpy()
                    
                    if 0:
                        plt.imsave( r"true%d-0.png"%(base_index+i), labels[i,0,:,:].cpu().numpy())
                        plt.imsave( r"true%d-1.png"%(base_index+i), labels[i,1,:,:].cpu().numpy())
                        
                        plt.imsave( r"gray%d-0.png"%(base_index+i), gyimage[0,:,:])
                        plt.imsave( r"gray%d-1.png"%(base_index+i), gyimage[1,:,:])
                        plt.imsave( r"gray%d-2.png"%(base_index+i), gyimage[2,:,:])
                        plt.imsave( r"gray%d-3.png"%(base_index+i), gyimage[3,:,:])
                        
                        plt.imsave( r"like%d-0.png"%(base_index+i), lkimage[0,:,:])
                        plt.imsave( r"like%d-1.png"%(base_index+i), lkimage[1,:,:])
                    
                    likelihoods_raw = np.mean(np.mean(lkimage, axis=2), axis=0)
                    
                    # find one peak...
                    likelihoods_g = scipy.ndimage.gaussian_filter1d(likelihoods_raw, sigma=1)
                    argmaxz = np.argmax(likelihoods_g)
                    
                    likelihoods.append(np.max(likelihoods_g))
                    
                    if 1: 
                        # with real slice thickness
                        real_original_slice_thickness = original_slice_thicknesses[case_idx]
                        argmaxz = argmaxz / real_original_slice_thickness * INPUT_SLICE_THICKNESS
                        
                        argmaxz = int(argmaxz+.5)
                        
                    # output the slice
                    print("case #%d, real z=%d, detected z=%d" % (case_idx, ztrues[case_idx], argmaxz))
                    
                    zestimates.append(argmaxz)
                    
                    # append likelihooods of -4 ~ +3 z
                    for idx_diff in range(8):
                        posz = (argmaxz - 4 + idx_diff) * real_original_slice_thickness // INPUT_SLICE_THICKNESS
                        if posz<0 or INPUT_Z_SIZE<=posz:
                            continue
                        likelihood_matrix[idx_diff, case_idx] = likelihoods_raw[int(posz)]
                    

                base_index += res.size()[0]

        # file output
        if 0: # obsolete
            with open(output_file_name, "wt") as f:
                for count in range(len(zestimates)):
                    f.write("case, %d, z, %d, realz, %d, likelihood, %f, l-4tol+3, %f,%f,%f,%f,%f,%f,%f,%f \n" 
                            %  (count, zestimates[count], ztrues[count], likelihoods[count],
                                    likelihood_matrix[0,count],
                                    likelihood_matrix[1,count],
                                    likelihood_matrix[2,count],
                                    likelihood_matrix[3,count],
                                    likelihood_matrix[4,count],
                                    likelihood_matrix[5,count],
                                    likelihood_matrix[6,count],
                                    likelihood_matrix[7,count]
                                )
                            )

        else:
            df = pd.read_csv(volume_list, encoding='shift_jis')
            df["{z_estimated}"] = list(zestimates)
            df["{z_true}"] = list(ztrues)
            df["{likelihood}"] = list(likelihoods)
            print(df)
            df.iloc[:,0:].to_csv(output_file_name, encoding='shift_jis')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Lumbar detector',
        add_help=True)

    parser.add_argument('--mode', help="train or test", 
                        type=str, choices=['train', 'test'],  default='train')
    parser.add_argument('--volume_list',
                        help='File path list of training/test image data',
                        type=str, required=True)
    parser.add_argument('--batch_size',
                        help='batch_size',
                        type=int, default=50)
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        type=float, default=0.001)
    parser.add_argument('--beta_1',
                        help='beta_1',
                        type=float, default=0.99)
    parser.add_argument('--device',
                        help='device',
                        type=str, default="cuda")
    parser.add_argument('--model_file_name',
                        help='model_file_name',
                        type=str, default='model/lumbar_model.ckpt')
    parser.add_argument('--output_file_name',
                        help='output_file_name',
                        type=str, default='temp/l3.txt')
    parser.add_argument('--n_epoch',
                        help='total epoch num',
                        type=int, default=100)
    parser.add_argument('--labelused',
                        help='label index to be used (default=2)',
                        type=int, default=2)
    parser.add_argument('--data_augmentation_multiplier',
                        help='data_augmentation_multiplier (default=10)',
                        type=int, default=10)


    args = parser.parse_args()

    main(**args.__dict__)

