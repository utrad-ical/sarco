


import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import SimpleITK as sitk
import scipy.ndimage

import torch
import torch.nn
import torch.optim as optim

import torchvision.models.densenet
import torchvision.models.resnet

import torchvision.transforms

import readfiles





transform = torchvision.transforms.Compose(
    [torchvision.transforms.RandomErasing(p=0.5),
     torchvision.transforms.RandomAffine(degrees=5.0)]
)





ORIGINAL_SLICE_THICKNESS = 5            # mm
INPUT_VOXEL_SIZE=1      # mm
INPUT_X_SIZE = 512      # voxels
INPUT_Y_SIZE = 512      # voxels



def data_preparation(volume_list : str, labelused : int, istest : bool, inputzfile = None):

    i = 0
    training_gray = np.empty([0,3,INPUT_Y_SIZE,INPUT_X_SIZE], dtype='int64')
    training_label = np.empty([0], dtype='int32')
    zs = []
    ztrues = []
    basezs = []
    searchalls = []
    detectedzs = []
    case_indices = []
    
    if(inputzfile is not None and istest):
        for line in open(inputzfile, 'rt'):
            _, idxcase, _, detectedz, _, _realz, _, likelihood,_,d0,d1,d2,d3,d4,d5,d6,d7 = line.strip().split(',')
            idxcase = int(idxcase.strip())
            detectedz = int(detectedz.strip())
            likelihood = float(likelihood.strip())
            
            if likelihood < 0.00001:
                searchall = True
            else:
                searchall = False
            searchalls.append(searchall)
            detectedzs.append(detectedz)
    
    i = 0
    for (grayImage, labelImage) in readfiles.reader(volume_list):
        
        # true label
        lorig = sitk.GetArrayFromImage(labelImage)
        ztrue = np.argmax(np.sum(np.sum(lorig == labelused, axis=2), axis=1))

        ztrues.append(ztrue)
        
        # if test, only partial volume is used
        searchalls = False
        if(istest and searchalls):
            basez = 0
            depthz = numslice
        elif(istest and not searchalls):
            basez = detectedzs[i] * ORIGINAL_SLICE_THICKNESS - 20
            depthz = int(40 / ORIGINAL_SLICE_THICKNESS)
        else:
            basez = ztrue * ORIGINAL_SLICE_THICKNESS - 40
            depthz = int(80 / ORIGINAL_SLICE_THICKNESS)
        
        basezs.append(basez / ORIGINAL_SLICE_THICKNESS)
        
        # resize to (1mm,1mm,5mm)
        resample = sitk.ResampleImageFilter()
        resample.SetOutputDirection(grayImage.GetDirection())
        resample.SetOutputOrigin(grayImage.GetOrigin())
        new_spacing = [INPUT_VOXEL_SIZE, INPUT_VOXEL_SIZE, ORIGINAL_SLICE_THICKNESS]
        resample.SetOutputSpacing(new_spacing)

        gRspac = resample.GetOutputSpacing()

        gIorig = grayImage.GetOrigin()
        gIspac = grayImage.GetSpacing()
        gIsize = grayImage.GetSize()

        gRsize = [INPUT_X_SIZE, INPUT_X_SIZE, depthz]
        resample.SetSize(gRsize)
        gRsize = resample.GetSize()
        
        gRorigin_bkup = np.array(gIorig)
        gRorigin = np.array(gIorig) + (np.array(gIspac)*np.array(gIsize))/2 - (np.array(gRsize)*np.array(gRspac))/2
        gRorigin[2] = gIorig[2] + basez

        resample.SetOutputOrigin(gRorigin)
        
        resample.SetUseNearestNeighborExtrapolator(True)
        
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(-1000)
        gR = resample.Execute(grayImage)

        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetDefaultPixelValue(0)
        lR = resample.Execute(labelImage)

        # modify grayscale and label
        gRarr = sitk.GetArrayFromImage(gR)
        gRarr[gRarr<100] = 100
        lRarr = sitk.GetArrayFromImage(lR)
        lRarr = (lRarr == labelused).astype(float)

#        plt.imsave('__testu.png', gRarr[0,:,:])
#        plt.imsave('__testd.png', gRarr[-1,:,:])

        l = np.squeeze(np.max(np.max(lRarr,axis=2), axis=1)).astype('int64')

        print(l)

        g = np.repeat(gRarr[:, np.newaxis, :, :], 3, axis=1)
        
        training_gray = np.concatenate((training_gray, g), axis=0)
        training_label = np.concatenate((training_label, l), axis=0)

        case_indices += [i,] * l.shape[0] # for recover of indices

        i+=1
    
    return training_gray, training_label, zs, ztrues, basezs, case_indices




def main(mode : str, 
         volume_list : str, 
         batch_size, 
         device : str, 
         learning_rate : float,
         beta_1 : float, 
         model_file_name : str,
         input_file_name : str,
         output_file_name : str,
         n_epoch : int,
         labelused : int):
    
    # build a model
    model = torchvision.models.densenet121(num_classes = 2)
    #model = torchvision.models.resnet50(num_classes = 2)
    
    
    
    if(mode == 'test'):
        model.load_state_dict(torch.load(model_file_name))
    
    
    
    # apply data parallel
    model = torch.nn.DataParallel(model)

    model = model.to(device)



    if(mode == 'train'):
        
        # data preparation
        (training_gray, training_label, zs, ztrues, _, _) = data_preparation(volume_list, 
                                                                       labelused, False)
        
        
        
        # data loader
        
        tensor_x = torch.from_numpy(training_gray)
        tensor_y = torch.from_numpy(training_label)
        
        training_dataset = torch.utils.data.TensorDataset(tensor_x.type(torch.FloatTensor),
                                                          tensor_y.type(torch.LongTensor))
        training_loader = torch.utils.data.DataLoader(training_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True)



        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta_1, 0.999))



        weight = torch.tensor((0.05,0.95)).to(device)



        for epoch in range(n_epoch):
        
            model.train()
            
            sum0 = 0
            sum1 = 0
            for batch_idx, (data, labels) in enumerate(training_loader):
            
                optimizer.zero_grad()
            
                data, labels = data.to(device), labels.to(device)
                
                data = transform(data)
                
                outputs = model(data)
                
                #outputs = torch.nn.LogSoftmax()(outputs)
                
                loss = torch.nn.CrossEntropyLoss(weight)(outputs, labels)
                
                loss = torch.mean(loss)
                
                loss.backward()
                optimizer.step()
                
                sum0 += data.shape[0]
                sum1 += loss
            
            print("%f" % (sum1/sum0))
            torch.save(model.module.state_dict(), model_file_name)




    elif (mode == 'test'):
    
        test_gray, test_label, zs, ztrues, basezs, case_indices = data_preparation(volume_list, 
                                                                                   labelused, 
                                                                                   True, 
                                                                                   input_file_name)
        
        num_cases = len(ztrues)
        print("case num = %d" % num_cases)
        
        tensor_x = torch.from_numpy(test_gray)
        tensor_y = torch.from_numpy(test_label)
        
        test_dataset = torch.utils.data.TensorDataset(tensor_x.type(torch.FloatTensor),
                                                      tensor_y.type(torch.FloatTensor))
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        base_index = 0
        model.eval()
        likelihoods = np.zeros_like(test_label).astype('float32')
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_loader):

                batchsz = data.size()[0]

                data, labels = data.to(device), labels.to(device)
                
                res = model(data)
                
                sm = torch.nn.Softmax(dim=1)(res)
                
                l = sm.detach().cpu().numpy()[:,1]
                likelihoods[base_index:base_index+batchsz] = l

                base_index += batchsz
                
        # reconstruction
        zestimates = np.ndarray((num_cases))
        case_indices = np.array(case_indices)
        likelihoods = np.array(likelihoods)
        likelihoods_cases = np.ndarray((num_cases))
        
        for ind_case in range(num_cases):
            print(ind_case)
            print(test_label[case_indices == ind_case])
            print(likelihoods[case_indices == ind_case])
            zestimates[ind_case] = np.argmax(likelihoods[case_indices == ind_case]) + basezs[ind_case]
            likelihoods_cases[ind_case] = np.max(likelihoods[case_indices == ind_case])
        
        with open(output_file_name, "wt") as f:
            for ind_case in range(num_cases):
                f.write("case, %d, z, %d, realz, %d, likelihood, %f\n" % 
                    (ind_case, zestimates[ind_case], ztrues[ind_case], likelihoods_cases[ind_case]))



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
                        type=int, default=24)
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
                        type=str, default='model/modifier_model.ckpt')
    parser.add_argument('--input_file_name',
                        help='input_file_name',
                        default='temp/l3.txt')
    parser.add_argument('--output_file_name',
                        help='output_file_name',
                        type=str, default='temp/l3mod.txt')
    parser.add_argument('--n_epoch',
                        help='total epoch num',
                        type=int, default=100)
    parser.add_argument('--labelused',
                        help='label index to be used (default=2)',
                        type=int, default=2)


    args = parser.parse_args()

    main(**args.__dict__)

