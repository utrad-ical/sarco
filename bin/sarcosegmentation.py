


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
import torch.nn.functional
import torch.optim as optim
import unet

import readfiles
import prepcache
import volumeresampler
import sarcocalculate
import extract_trunk
import makeill



INPUT_SLICE_THICKNESS=5
INPUT_VOXEL_SIZE=1      # mm
INPUT_X_SIZE = 512      # voxels
INPUT_Z_SIZE = 512      # voxels
OUTPUT_SLICE_THICKNESS=1

LABEL_NUM=3



def dice_loss(outputs, labels, eps = 1.e-6):

    n_batch = outputs.size()[0]

    losses = []
    
    for i_label in range(1,LABEL_NUM+1):

        pred = outputs[:, i_label, :].view([n_batch, -1])
        target = torch.where(labels == i_label, 1., 0.).view([n_batch, -1])

        intersection = torch.sum(pred*target, axis=1)
        loss = 1. - (2. * intersection + eps) / (torch.sum(pred, axis=1) + torch.sum(target, axis=1) + eps)

        losses.append(torch.sum(loss))

    return torch.stack(losses).sum() / LABEL_NUM



def mse_loss(outputs, labels):

    n_batch = outputs.size()[0]

    losses = []

    assert torch.sum(labels[:] - torch.floor(labels[:])) == 0
    assert torch.min(labels) == 0
    assert torch.max(labels) == LABEL_NUM

    onehot = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=LABEL_NUM+1).to(torch.float32).transpose(1, 4).squeeze()

    loss = torch.mean((outputs - onehot)**2, axis=(2,3))
    #loss = loss[:,0] * 0.000001 + loss[:,1] * 1.0 + loss[:,2] * 1.0 + loss[:,3] * 1.0 # weight for background vs foreground
    loss = torch.mean(loss, axis=1) # mean over labels
    return torch.sum(loss)



def cross_entropy_loss(outputs, labels):

    n_batch = outputs.size()[0]

    losses = []

    assert torch.sum(labels[:] - torch.floor(labels[:])) == 0
    assert torch.min(labels) == 0
    assert torch.max(labels) == LABEL_NUM

    onehot = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=LABEL_NUM+1).to(torch.float32).transpose(1, 4).squeeze()

    loss = torch.mean(torch.nn.BCEWithLogitsLoss(reduction='none')(outputs, onehot), axis=(2,3))
    #loss = loss[:,0] * 0.000001 + loss[:,1] * 1.0 + loss[:,2] * 1.0 + loss[:,3] * 1.0 # weight for background vs foreground
    loss = torch.mean(loss, axis=1) # mean over labels
    return torch.sum(loss)



def slice_preparation(grayVolume, labelVolume_orig, labelused, usingz = None, tor_slice_diff = 0, trunkextract = False):

    lorig = sitk.GetArrayFromImage(labelVolume_orig)
    truez = np.argmax(np.sum(np.sum(lorig == labelused, axis=2), axis=1))

    if usingz is None:
        usingz = truez

#    lorig = (lorig == labelused).astype('uint8')
    lorig = lorig.astype('uint8')
    labelImage = sitk.GetImageFromArray(lorig)
    labelImage.SetDirection(labelVolume_orig.GetDirection())
    labelImage.SetOrigin(labelVolume_orig.GetOrigin())
    labelImage.SetSpacing(labelVolume_orig.GetSpacing())

    if 0:
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
    
    gR = volumeresampler.resample(grayVolume, basez, new_voxel_size, new_matrix_size, is_label = False, apply_filter = True)
    lR = volumeresampler.resample(labelImage, basez, new_voxel_size, new_matrix_size, is_label = True, apply_filter = False)



    gRarr = sitk.GetArrayFromImage(gR)
    gRarr[gRarr<-1000] = -1000

    lRarr = sitk.GetArrayFromImage(lR)


    if(trunkextract):
        image = gRarr[0,:,:]
        image = extract_trunk.extract(image)
        gRarr[0,:,:] = image



    return (np.squeeze(gRarr), np.squeeze(lRarr))



def data_preparation(volume_list : str, labelused, z_list = None, tor_slice_diff = 0, trunkextract = False, cache_dir = None):
    
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
    for (_, line) in readfiles.read_list(volume_list).iterrows():

        gray_path, label_path = readfiles.paths(line)
        usingz = None if zs is None else zs[index]

        def compute():
            grayVolume, labelVolume = readfiles.readone(line)
            image_gray, image_label = slice_preparation(grayVolume, labelVolume, labelused, usingz, tor_slice_diff, trunkextract)
            return {"gray": image_gray, "label": image_label}

        # the volumes are read only on a cache miss
        prepared = prepcache.cached(cache_dir, "seg_slice",
                                    {"gray": prepcache.file_signature(gray_path),
                                     "label": None if label_path is None else prepcache.file_signature(label_path),
                                     "labelused": labelused,
                                     "usingz": None if usingz is None else float(usingz),
                                     "trunkextract": bool(trunkextract),
                                     "slice_thickness": INPUT_SLICE_THICKNESS,
                                     "voxel_size": INPUT_VOXEL_SIZE,
                                     "x_size": INPUT_X_SIZE},
                                    compute)
        image_gray, image_label = prepared["gray"], prepared["label"]
        images_gray = np.concatenate((images_gray, image_gray[np.newaxis, np.newaxis, : ,:]), axis=0)
        images_label = np.concatenate((images_label, image_label[np.newaxis, np.newaxis, : ,:]), axis=0)
    
        index += 1
    
    assert zs is None or len(images_gray) == len(zs)
    
    return (images_gray, images_label, zs)



def visual_volume_creation(volume_list : str, estim_labels : np.array, zs : list, labelused : int, 
                             workdir : str, cache_dir = None) -> pd.DataFrame:

    allmetrics = []
    volume_list_lines = [line for (_, line) in pd.read_csv(volume_list, encoding='shift_jis').iterrows()]

    for i, (grayVolume, labelVolume) in enumerate(readfiles.reader(volume_list)):

        # extract true z
        lorig = sitk.GetArrayFromImage(labelVolume)
        truez = np.argmax(np.sum(np.sum(lorig == labelused, axis=2), axis=1))

        # load the answer

#        answerlabelVolume = sitk.GetImageFromArray((lorig == labelused).astype('uint8'))
        answerlabelVolume = sitk.GetImageFromArray(lorig.astype('uint8'))
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
        sitk.WriteImage(answerlabelVolume, '%s/answerlabel%d.nii.gz' % (workdir, i))



        if 1:

            # make illustrated results
            makeill.write_sag_fusion_res_ill(detectedLabelVolume, answerlabelVolume, grayVolume, 
                                             '%s/sag%d.png' % (workdir, i), 
                                             '%s/fusion%d.png' % (workdir, i), 
                                             '%s/res%d.png' % (workdir, i), 
                                             '%s/ill%d.png' % (workdir, i)
                                             )
            
            # ...and calculate all metrics and tricolor image. the trunk mask
            # depends on the gray volume only, so L1 and L3 share one
            gray_path, _ = readfiles.paths(volume_list_lines[i])
            mask = prepcache.cached(cache_dir, "trunk_mask",
                                    {"gray": prepcache.file_signature(gray_path)},
                                    lambda: {"mask": extract_trunk.trunk_mask(grayVolume)})["mask"]
            (tri_volume, tri_png, metrics) = sarcocalculate.calculate(grayVolume, detectedLabelVolume, mask)
            
            sitk.WriteImage(sitk.GetImageFromArray(tri_volume), '%s/tricolor%d.nii.gz' % (workdir, i))
            imageio.imwrite("%s/tricolor%d.png" % (workdir, i), np.stack([tri_png]*3, axis=2).astype('uint8'))
            
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
         trunkextract : bool,
         cache_dir : str = None
         ):



    if(mode == 'train'):
        # do not use detected z_list
        z_list_file = None
        tor_slice_diff = 0
        volume_list_file = readfiles.drop_empty_labels(volume_list_file, labelused, cache_dir)



    # data preparation
    (images_gray, images_label, zs) = data_preparation(volume_list_file, labelused, z_list_file, tor_slice_diff, trunkextract, cache_dir)
    
    
    
    # model preparation
    # build a model
    model = unet.Unet(1,LABEL_NUM+1,16)
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
                
                outputs_softmax = torch.nn.functional.softmax(outputs, dim=1)
#                outputs = torch.nn.functional.sigmoid(outputs)
                
#                for i in range(data.shape[0]):
#                    png = outputs[i,1:4,:,:].detach().to("cpu").numpy().astype('float32').transpose(1,2,0)
#                    imageio.imwrite("g%d.png"%i, (np.clip(png, 0, 1)*255).astype('uint8'))
#
#                    png = labels[i,0,:,:].detach().to("cpu").numpy().astype('float32')
#                    png = np.stack([png]*3, axis=2)
#                    imageio.imwrite("l%d.png"%i, (np.clip(png, 0, 3)/3*255).astype('uint8'))

                loss = dice_loss(outputs_softmax, labels) + cross_entropy_loss(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                sum0 += data.shape[0]
                sum1 += loss
            
            print("%i:%f" % (epoch, sum1/sum0))

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
        estim_labels = np.empty([0,512,512], dtype='int')

        model.eval()
        with torch.no_grad():
            idx_base = 0
            for batch_idx, (data, labels) in enumerate(test_loader):
                
                data, labels = data.to(device), labels.to(device)
                
                outputs = model(data)
                
                outputs = torch.nn.functional.sigmoid(outputs)

                estim_labels_in_batch = torch.argmax(outputs, dim=1, keepdims=False).detach().to("cpu").numpy().astype(int)
                estim_labels = np.concatenate((estim_labels, estim_labels_in_batch), 0)
                    
                idx_base += data.shape[0]

        if 0:
            fp = open(logfilepath, 'at')
            for i in range(len(res)):
                fp.write("%d, %f\n" % (i, res[i]))
            print("avg dice=%f" % np.mean(np.array(res)))

        del model

        # visual image creation
        metrics = visual_volume_creation(volume_list_file, estim_labels, zs, labelused, workdir, cache_dir)
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
            
            while namebody in namebodies:
                namebody = namebody + '_'

            namebodies.append(namebody)
            
            namebody = namebody + "_" + desc
            
            if 1:
            
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

        if 1:
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
                        type=int, default=1)
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
                        action='store_true', default=True)
    parser.add_argument('--cache_dir', help = 'directory to cache per-case preprocessing in (default: no cache)',
                        type=str, default=None)



    args = parser.parse_args()

    main(**args.__dict__)
