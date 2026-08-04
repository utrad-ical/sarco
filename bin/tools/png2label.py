


import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

import SimpleITK as sitk



INPUT_SLICE_THICKNESS=5
INPUT_VOXEL_SIZE=1      # mm
INPUT_X_SIZE = 512      # voxels
INPUT_Z_SIZE = 512      # voxels





def png2array(mask_png : str, 
              z : int, 
              origVolume):
    
    # get original pixel size and dimension
#    origVolume = sitk.ReadImage(original_mhd)
    gOorig = origVolume.GetOrigin()
    gOspac = origVolume.GetSpacing()
    gOsize = origVolume.GetSize()
    
    # get mask png
    mask = (plt.imread(mask_png).astype('uint8')[:,:,0]>0).astype('uint8')
    
    # get z
    z_mm = z * INPUT_SLICE_THICKNESS
    
    # resample to volume
    gRspac = np.array([INPUT_VOXEL_SIZE, INPUT_VOXEL_SIZE, INPUT_SLICE_THICKNESS]).astype('float64')
    gRsize = np.array([INPUT_X_SIZE, INPUT_X_SIZE, 1])
    gRorig = np.array([
        - gRsize[0]*gRspac[0]/2 + gOsize[0]*gOspac[0]/2 + gOorig[0],
        - gRsize[1]*gRspac[1]/2 + gOsize[1]*gOspac[1]/2 + gOorig[1],
        z_mm
    ])
    
    # make mask volume
    #maskVolume = sitk.Image(gRsize[0], gRsize[1], gRsize[2], sitk.sitkUInt8)
    maskVolume = sitk.GetImageFromArray(mask[np.newaxis, :, :])
    maskVolume.SetDirection(origVolume.GetDirection())
    maskVolume.SetOrigin(gRorig)
    #maskVolume.SetSize(gRsize)
    maskVolume.SetSpacing(gRspac)
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(origVolume.GetDirection())
    resample.SetOutputOrigin(origVolume.GetOrigin())
    resample.SetOutputSpacing(origVolume.GetSpacing())
    resample.SetSize(origVolume.GetSize())
    
    resample.SetUseNearestNeighborExtrapolator(False)
    
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetDefaultPixelValue(0)

    newVolume = resample.Execute(maskVolume)



    return newVolume



if __name__ == '__main__':

    png2array(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])

