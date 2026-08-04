
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import imageio

import SimpleITK as sitk

import volumeresampler

INPUT_X_SIZE = 512      # voxels
INPUT_Z_SIZE = 512      # voxels
OUTPUT_SLICE_THICKNESS=1



def write_sag_fusion_res_ill(detectedLabelVolume, 
                             answerLabelVolume, 
                             grayVolume, 
                             sag_filename, 
                             fus_filename, 
                             res_filename, 
                             ill_filename) -> None:

    if answerLabelVolume is None:
        answerLabelVolume = detectedLabelVolume * 0

    assert np.all(detectedLabelVolume.GetSpacing() == grayVolume.GetSpacing())
    assert np.all(detectedLabelVolume.GetSize() == grayVolume.GetSize())

    # resize to (1mm,1mm,1mm)
    new_voxel_size = [OUTPUT_SLICE_THICKNESS, OUTPUT_SLICE_THICKNESS, OUTPUT_SLICE_THICKNESS]
    new_matrix_size = [INPUT_X_SIZE, INPUT_X_SIZE, INPUT_Z_SIZE*2]
    
    gR = volumeresampler.resample(grayVolume, 0, new_voxel_size, new_matrix_size, False)
    dlR = volumeresampler.resample(detectedLabelVolume, 0, new_voxel_size, new_matrix_size, True)
    alR = volumeresampler.resample(answerLabelVolume, 0, new_voxel_size, new_matrix_size, True)

    # select yz slice
    arr = np.arange(dlR.GetSize()[0])
    yzsum = np.sum(np.sum(sitk.GetArrayFromImage(dlR), axis=1), axis=0)
    xsum0 = np.sum(yzsum, axis=0)
    xsum1 = np.sum(yzsum*arr, axis=0)
    if(xsum0 == 0):
        xgravity = INPUT_X_SIZE // 2
    else:
        xgravity = int(xsum1 / xsum0 +.5)
    
    # make a sagittal
    gRarr = sitk.GetArrayFromImage(gR)  [:, :, xgravity]
    dlRarr = np.max(sitk.GetArrayFromImage(dlR).astype('int16'), axis=2)
    alRarr = np.max(sitk.GetArrayFromImage(alR).astype('int16'), axis=2)

    sag = np.zeros((INPUT_Z_SIZE*2, INPUT_X_SIZE, 3), dtype='float32')
    sag[:,:,0:3] = gRarr[:,:,np.newaxis] / 300
    sag[:,:,0] += (dlRarr.astype('int16') > 0).astype(float) * 1
    sag[:,:,1] += (alRarr.astype('int16') > 0).astype(float) * 1
    sag[:,:,2] += (dlRarr.astype('int16') > 0).astype(float) * 1
    sag = np.clip(sag, 0., 1.)
    if sag_filename is not None:
        plt.imsave(sag_filename, sag)
    
    # (re)make the fusion and res
    dlarr3d = (sitk.GetArrayFromImage(dlR).astype('int16')).astype('float32')
    alarr3d = (sitk.GetArrayFromImage(alR).astype('int16')).astype('float32')
    garr3d = sitk.GetArrayFromImage(gR)

    xysum = np.sum(np.sum(dlarr3d, axis=2), axis=1)
    if(np.all(xysum == 0)):
        detectedslicenum = -1
        true_label = np.zeros_like(dlarr3d[0, :, :])[:,:,np.newaxis]
        estim_label = np.zeros_like(dlarr3d[0, :, :])[:,:,np.newaxis]
        gray = np.zeros_like(dlarr3d[0, :, :])[:,:,np.newaxis]
    else:
        detectedslicenum = np.argmax(xysum)
        true_label = alarr3d[detectedslicenum, :, :][:,:,np.newaxis]
        estim_label = dlarr3d[detectedslicenum, :, :][:,:,np.newaxis]
        gray = (np.clip(garr3d[detectedslicenum, :, :][:,:,np.newaxis], -100, 155)+100)/255
    fus = np.concatenate(((estim_label==1)*.6+gray*.4, (estim_label>1)*.6+gray*.4, estim_label*.1+gray*.5), 2)
    res = np.concatenate((estim_label, true_label, estim_label), 2)

    fus = np.clip(fus, 0., 1.)
    res = np.clip(res, 0., 1.)

    if fus_filename is not None:
        plt.imsave(fus_filename, fus)
    if res_filename is not None:
        plt.imsave(res_filename, res)

    # finally make the illust
    ill = np.concatenate((np.concatenate((fus,res), axis=0), sag), axis=1)
    
    if ill_filename is not None:
        plt.imsave(ill_filename, ill)
    



