


import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import shutil
from pathlib import Path
import imageio

import SimpleITK as sitk
import scipy.ndimage

import torch
import torch.nn
import torch.optim as optim
import unet

import readfiles
import volumeresampler
import sarcocalculate
import extract_trunk
import makeill



INPUT_SLICE_THICKNESS=5
INPUT_VOXEL_SIZE=1      # mm
INPUT_X_SIZE = 512      # voxels
INPUT_Z_SIZE = 512      # voxels
OUTPUT_SLICE_THICKNESS=1


def dice_loss(outputs, labels, eps = 1.):
    n_batch = outputs.size()[0]

    pred = outputs.view([n_batch, -1])
    target = labels.view([n_batch, -1])

    intersection = torch.sum(pred*target, axis=1)
    loss = 1. - (2. * intersection) / (torch.sum(pred, axis=1) + torch.sum(target, axis=1) + eps)

    return torch.sum(loss)



def slice_preparation(grayVolume, labelVolume_orig, labelused, usingz = None, tor_slice_diff = 0, trunkextract = False):

    lorig = sitk.GetArrayFromImage(labelVolume_orig)
    truez = np.argmax(np.sum(np.sum(lorig == labelused, axis=2), axis=1))

    if usingz is None:
        usingz = truez

    lorig = (lorig == labelused).astype('uint8')
    labelImage = sitk.GetImageFromArray(lorig)
    labelImage.SetDirection(labelVolume_orig.GetDirection())
    labelImage.SetOrigin(labelVolume_orig.GetOrigin())
    labelImage.SetSpacing(labelVolume_orig.GetSpacing())

    if 0<tor_slice_diff:
        bdif = sitk.BinaryDilateImageFilter()
        bdif.SetKernelRadius((0,0,tor_slice_diff))
        bdif.SetKernelType(sitk.sitkBox)
        labelImage = bdif.Execute(labelImage)
        cif = sitk.CastImageFilter()
        cif.SetOutputPixelType(sitk.sitkUInt8)
        labelImage = cif.Execute(labelImage)

    lorig = sitk.GetArrayFromImage(labelImage)


    new_voxel_size = [INPUT_VOXEL_SIZE, INPUT_VOXEL_SIZE, INPUT_SLICE_THICKNESS]
    new_matrix_size = [INPUT_X_SIZE, INPUT_X_SIZE, 1]
    
    basez = usingz*grayVolume.GetSpacing()[2]
    
    gR = volumeresampler.resample(grayVolume, basez, new_voxel_size, new_matrix_size)
    lR = volumeresampler.resample(labelImage, basez, new_voxel_size, new_matrix_size)



    gRarr = sitk.GetArrayFromImage(gR)
    gRarr[gRarr<-1000] = -1000

    lRarr = sitk.GetArrayFromImage(lR)


    if(trunkextract):
        image = gRarr[0,:,:]
        image = extract_trunk.extract(image)
        gRarr[0,:,:] = image



    return (np.squeeze(gRarr), np.squeeze(lRarr))



def data_preparation(volume_list : str, labelused, z_list = None, tor_slice_diff = 0, trunkextract = False):
    
    images_gray = np.empty([0,1,512,512], dtype='float32')
    images_label = np.empty([0,1,512,512], dtype='float32')
    
    zs = None
    if(z_list is not None):
        if 0: # obsolete
            zs = []
            for line in open(z_list, 'rt'):
                line = line.strip().split(',')
                zs.append(int(line[3].strip()))
        else:
            df = pd.read_csv(z_list, encoding='shift_jis')
            zs = df["{z_modified}"]
            
        print("%s read" % z_list)
    
    
    
    index = 0
    for grayVolume, labelVolume in readfiles.reader(volume_list):

        image_gray, image_label = slice_preparation(grayVolume, labelVolume, labelused, None if zs is None else zs[index], tor_slice_diff, trunkextract)
        images_gray = np.concatenate((images_gray, image_gray[np.newaxis, np.newaxis, : ,:]), axis=0)
        images_label = np.concatenate((images_label, image_label[np.newaxis, np.newaxis, : ,:]), axis=0)
    
        index += 1
    
    assert len(images_gray) == len(zs)
    
    return (images_gray, images_label, zs)



def visual_volume_creation(volume_list : str, estim_labels : np.array, zs : list, labelused : int, 
                             workdir : str) -> pd.DataFrame:

    allmetrics = []

    for i, (grayVolume, labelVolume) in enumerate(readfiles.reader(volume_list)):

        # extract true z
        lorig = sitk.GetArrayFromImage(labelVolume)
        truez = np.argmax(np.sum(np.sum(lorig == labelused, axis=2), axis=1))

        # load the answer

        answerlabelVolume = sitk.GetImageFromArray((lorig == labelused).astype('uint8'))
        answerlabelVolume.SetOrigin(grayVolume.GetOrigin())
        answerlabelVolume.SetSpacing(grayVolume.GetSpacing())
        answerlabelVolume.SetDirection(grayVolume.GetDirection())

        # make the result label

        detectedarray = np.zeros((grayVolume.GetSize()[2],
                                  INPUT_X_SIZE, #grayVolume.GetSize()[1], 
                                  INPUT_X_SIZE  #grayVolume.GetSize()[0]
                                  ))

        # extract detected z
        detectedz = zs[i]
        if(detectedz < 0 or grayVolume.GetSize()[2] <= detectedz):
             # out of view
             detectedz = -1
        else:
            print(f"{detectedz=}")
            # implant the detection result
            #if grayVolume.GetDirection()[8] < 0:
            #    # odd direction: reverse z
            #    detectedarray[grayVolume.GetSize()[2]-detectedz-1, :, :] = np.squeeze(estim_labels[i,:,:])
            #else:
            x_size_min = np.minimum(INPUT_X_SIZE, estim_labels.shape[2])
            y_size_min = np.minimum(INPUT_X_SIZE, estim_labels.shape[1])
            detectedarray[int(detectedz), 0:y_size_min, 0:x_size_min] = \
                estim_labels[i, 0:y_size_min, 0:x_size_min]

        # make a volume to embed
        detectedLabelVolume_1mm_template = volumeresampler.resample(
            answerlabelVolume, 
            0, 
            [1.0, 1.0, grayVolume.GetSpacing()[2]], 
            [INPUT_X_SIZE, INPUT_X_SIZE, grayVolume.GetSize()[2]],
            True
        )
        
        detectedLabelVolume_1mm = sitk.GetImageFromArray(detectedarray)
        detectedLabelVolume_1mm.SetOrigin(detectedLabelVolume_1mm_template.GetOrigin())
        detectedLabelVolume_1mm.SetSpacing(detectedLabelVolume_1mm_template.GetSpacing())
        detectedLabelVolume_1mm.SetDirection(detectedLabelVolume_1mm_template.GetDirection())
        
        detectedLabelVolume = volumeresampler.resample(detectedLabelVolume_1mm, 0, 
            grayVolume.GetSpacing(), grayVolume.GetSize(),
            True)

        # save final result label
        
        sitk.WriteImage(detectedLabelVolume, '%s/reslabel%d.nii.gz' % (workdir, i))



        # make illustrated results
        makeill.write_sag_fusion_res_ill(detectedLabelVolume, answerlabelVolume, grayVolume, 
                                         '%s/sag%d.png' % (workdir, i), 
                                         '%s/fusion%d.png' % (workdir, i), 
                                         '%s/res%d.png' % (workdir, i), 
                                         '%s/ill%d.png' % (workdir, i)
                                         )
        
        # ...and calculate all metrics and tricolor image
        (tri_volume, tri_png, metrics) = sarcocalculate.calculate(grayVolume, detectedLabelVolume)
        
        sitk.WriteImage(sitk.GetImageFromArray(tri_volume), '%s/tricolor%d.nii.gz' % (workdir, i))
        imageio.imwrite("%s/tricolor%d.png" % (workdir, i), tri_png)
        
        allmetrics.append(metrics)
        
    return pd.DataFrame.from_records(allmetrics)




def main(mode : str,
         volume_list_file : str, 
         z_list_file, 
         device : str, 
         model_file_name : str,
         logfilepath: str,
         labelused : int, 
         batch_size : int,
         n_epoch : int, 
         learning_rate : float,
         beta_1 : float,
         tor_slice_diff : int,
         sagittal : bool,
         workdir : str,
         resultdir : str,
         desc : str,
         trunkextract : bool
         ):



    if(mode == 'train'):
        # do not use detected z_list
        z_list_file = None
        tor_slice_diff = 0



    # data preparation
    (images_gray, images_label, zs) = data_preparation(volume_list_file, labelused, z_list_file, tor_slice_diff, trunkextract)
    
    
    
    # model preparation
    # build a model
    model = unet.Unet(1,1,16)
    if(mode == 'test'):
        model.load_state_dict(torch.load(model_file_name))
    
    # apply data parallel
    model = torch.nn.DataParallel(model)

    model = model.to(device)
    
    
    
    if(mode == 'train'):
    
        # train
        # data_loader
        tensor_x = torch.from_numpy(images_gray)
        tensor_y = torch.from_numpy(images_label)
        
        training_dataset = torch.utils.data.TensorDataset(tensor_x.type(torch.FloatTensor),
                                                          tensor_y.type(torch.FloatTensor))
        training_loader = torch.utils.data.DataLoader(training_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True)



        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta_1, 0.999))



        model.train()
        for epoch in range(n_epoch):
            
            sum0 = 0
            sum1 = 0
            for batch_idx, (data, labels) in enumerate(training_loader):
            
                optimizer.zero_grad()
            
                data, labels = data.to(device), labels.to(device)
                
                outputs = model(data)
                
                outputs = torch.sigmoid(outputs)
                
                loss = dice_loss(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                sum0 += data.shape[0]
                sum1 += loss
            
            print("%f" % (sum1/sum0))
#            for i in range(10):
#                plt.imsave(r"output%i.png"%i, outputs.detach().to("cpu").numpy()[i,0,:,:]*255)
#                plt.imsave(r"graylabel%i.png"%i, data.detach().to("cpu").numpy()[i,0,:,:]+labels.detach().to("cpu").numpy()[i,0,:,:]*1000)
#            print(np.max(outputs.detach().to("cpu").numpy()[0,0,:,:]))
#            print(np.mean(outputs.detach().to("cpu").numpy()[0,0,:,:]))

        torch.save(model.module.state_dict(), model_file_name)



    if(mode == 'test'):
    
        # test
        # data_loader
        tensor_x = torch.from_numpy(images_gray)
        tensor_y = torch.from_numpy(images_label)
        
        test_dataset = torch.utils.data.TensorDataset(tensor_x.type(torch.FloatTensor),
                                                          tensor_y.type(torch.FloatTensor))
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)


        res = []
        estim_labels = np.empty([0,512,512], dtype='float32')

        model.eval()
        with torch.no_grad():
            idx_base = 0
            for batch_idx, (data, labels) in enumerate(test_loader):
                
                data, labels = data.to(device), labels.to(device)
                
                outputs = model(data)
                
                outputs = torch.sigmoid(outputs)
                
                for i in range(data.shape[0]):
                    estim_label = (0.5<outputs[i,0,:,:]).detach().to("cpu").numpy()[:,:,np.newaxis].astype(float)
                    true_label = (0<labels[i,0,:,:]).detach().to("cpu").numpy()[:,:,np.newaxis].astype(float)
                    plt.imsave("%s/res%d.png"%(workdir, idx_base+i), np.concatenate((estim_label, true_label, estim_label), 2))
                    
                    gray = data[i,0,:,:].detach().to("cpu").numpy()[:,:,np.newaxis]
                    gray = ( np.clip(gray, -99, 99) +100 ) /200 
                    plt.imsave("%s/gray%d.png"%(workdir, idx_base+i), np.concatenate((gray, gray, gray), 2))

                    plt.imsave("%s/fusion%d.png"%(workdir, idx_base+i), np.concatenate((estim_label*.6+gray*.4, gray*.4, estim_label*.5+gray*.5), 2))
                    
                    plt.imsave("%s/muscle%d.png"%(workdir, idx_base+i), np.concatenate((estim_label, estim_label, estim_label), 2).astype('float32'))
                                        
                    dicecoeff = 2*np.sum(estim_label*true_label) / (1+np.sum(estim_label) + np.sum(true_label))
                    
                    res.append( dicecoeff )
                    
                    estim_labels = np.concatenate((estim_labels, estim_label[np.newaxis,:,:,0]), 0)
                    
                idx_base += data.shape[0]
        fp = open(logfilepath, 'at')
        for i in range(len(res)):
            fp.write("%d, %f\n" % (i, res[i]))
        print("avg dice=%f" % np.mean(np.array(res)))

        del model

        # visual image creation
        metrics = visual_volume_creation(volume_list_file, estim_labels, zs, labelused, workdir)
        print(metrics)
        

        # file copy to result dir
        df = pd.read_csv(volume_list_file, encoding='shift_jis')
        ill_pathes = []
        tricolor_pathes = []
        result_pathes = []
        namebodies = []
        for i, (_, line) in enumerate(df.iterrows()):
            namebody = str(Path(line["{volume_name}"]).name)
            if namebody[-7:] == '.nii.gz':
                namebody = namebody[:-7]
            else:
                namebody = str(Path(namebody).stem)
            
            namebodies.append(namebody)
            
            namebody = namebody + "_" + desc
            
            # copy ill.png
            pngfilename = Path(namebody+".ill.png")
            pngpath = (Path(resultdir) / pngfilename).resolve()
            shutil.copyfile("%s/ill%d.png"%(workdir, i), pngpath)
            ill_pathes.append(pngpath)
            
            # copy tricolor.png
            pngfilename = Path(namebody+".tricolor.png")
            pngpath = (Path(resultdir) / pngfilename).resolve()
            shutil.copyfile("%s/tricolor%d.png"%(workdir, i), pngpath)
            tricolor_pathes.append(pngpath)
            
            # copy result label
            resultfilename = Path(namebody+".result.nii.gz")
            resultpath = (Path(resultdir) / resultfilename).resolve()
            shutil.copyfile("%s/reslabel%d.nii.gz"%(workdir, i), resultpath)
            result_pathes.append(resultfilename)

        df["{" + desc + "_ill_path}"] = ill_pathes
        df["{" + desc + "_tricolor_path}"] = tricolor_pathes
        df["{" + desc + "_result_path}"] = result_pathes
        
        df["id"] = namebodies
        
        # and add metrics
        metrics = metrics.add_prefix(desc+'_')
        metrics["id"] = namebodies
        
        df = pd.merge(df, metrics)
        
        df.iloc[:,1:].to_csv(Path(resultdir) / ('__result_' + desc + '.csv'), encoding='shift_jis')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='L1/L3 level segmentationer',
        add_help=True)

    parser.add_argument('--mode', help="train or test", 
                        type=str, choices=['train', 'test'],  default='train')
    parser.add_argument('--volume_list_file',
                        help='File path list of training/test image data',
                        type=str, required=True)
    parser.add_argument('--z_list_file',
                        help='File path list of training/test image slice no',
                        default="temp/l3mod.txt")
    parser.add_argument('--batch_size',
                        help='batch_size',
                        type=int, default=32)
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
                        type=str, default='model/segmentation_model.ckpt')
    parser.add_argument('--logfilepath',
                        help='logfilepath',
                        type=str, default='temp/log.txt')
    parser.add_argument('--labelused',
                        help='labelused (1 or 2)',
                        type=int, default=2)
    parser.add_argument('--n_epoch',
                        help='total epoch num',
                        type=int, default=1000)
    parser.add_argument('--tor_slice_diff',
                        help='tolerance of slice z difference (in test only)',
                        type=int, default=100)
    parser.add_argument('--sagittal',
                        help='make sagittal images',
                        type=bool, default=True)
    parser.add_argument('--workdir', help = 'workdir path',
                        type=str, default='temp')
    parser.add_argument('--resultdir', help = 'workdir path',
                        type=str, default='result')
    parser.add_argument('--desc', help = 'description label',
                        type=str, default='result')
    parser.add_argument('--trunkextract', help = 'turn on a preprocessing (body trunk extract)',
                        action='store_true', default='result')
                        



    args = parser.parse_args()

    main(**args.__dict__)
