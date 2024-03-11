


import numpy as np

import SimpleITK as sitk



def resample(image, z_base_shift_mm : float, new_voxel_size, new_matrix_size = None, is_label = False):


    # resize given image

    new_voxel_size = np.array(new_voxel_size)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    new_spacing = list(np.array(new_voxel_size).astype("float64"))
    resample.SetOutputSpacing(new_spacing)

    gNewSpac = resample.GetOutputSpacing()

    gOldOrig = image.GetOrigin()
    gOldSpac = image.GetSpacing()
    gOldSize = image.GetSize()
    gOldDire = image.GetDirection()

    if new_matrix_size is not None:
        gNewSize = new_matrix_size
    else:
        gNewSize = np.floor( 0.5 + np.array(gOldSpac)*np.array(gOldSize)/np.array(new_voxel_size) ).astype("uint32")

    resample.SetSize(list(gNewSize))
    gNewsize = resample.GetSize()
    
    gNewOrig = np.array(gOldOrig) + (np.array(gOldSpac)*np.array(gOldSize))/2 - (np.array(gNewSize)*np.array(gNewSpac))/2

    gNewOrig[2] = z_base_shift_mm + gOldOrig[2]

    if gOldDire[8] < 0:
        # odd direction; reverse the base
        gNewOrig[2] = gOldOrig[2] - z_base_shift_mm # gOldSize[2] * gOldSpac[2] + z_base_shift_mm + gNewSize[2] * gNewSpac[2]

    # print(f"{gIorig=}, {gIspac=}, {gRsize=}, {gRspac=}, {gRorigin=}, {gIdire=}")
    
    resample.SetOutputOrigin(gNewOrig)
    
    resample.SetUseNearestNeighborExtrapolator(False)
    
    # if image.GetPixelID() in [sitk.sitkUInt8, sitk.sitkInt8]:
    #     is_label = True
    # else:
    #     is_label = False

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        resample.SetDefaultPixelValue(0)
        newImage = resample.Execute(image)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(-1000)
        newImage = resample.Execute(image)

    return newImage




