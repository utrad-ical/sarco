


import sys

import numpy as np

import SimpleITK as sitk
import scipy.ndimage

import volumeresampler



CALC_SLICE_THICKNESS = 5.0



def get_trunk_label(image: np.array) -> np.array:
    image = np.copy(image)
    
    # enhance gap
    gapenhancer = np.maximum(scipy.ndimage.gaussian_laplace(image, sigma=1.0), 0.0) * 10.0
    gapenhancer[0 < image] = 0
    image_l = image - gapenhancer
    
    # get foreground (HU>-500) mask
    mask = image_l > -500
    
    # erosion
    mask = scipy.ndimage.binary_erosion(mask, iterations=1)
    
    # get the largest
    (labeled, num_features) = scipy.ndimage.label(mask)
    areas = [np.sum(labeled == i+1) for i in np.arange(num_features)]
    if num_features==0:
        maxarea = np.zeros_like(mask)
    else:
        maxareaindex = np.argmax(areas)+1
        maxarea = labeled == maxareaindex
    
    # dilation, hole filling
    dilated = scipy.ndimage.binary_dilation(maxarea, iterations=1)
    holefilled = scipy.ndimage.binary_fill_holes(dilated)
    
    return holefilled



def extract(image: np.array) -> np.array:
    '''
    2-D body trunk extractor
    '''
    
    res = np.copy(image)
    
    # kill background
    holefilled = get_trunk_label(image)
    res[holefilled == 0] = -1000
    
    return res



def kill_arms(volume: sitk.Image) -> sitk.Image:
    '''
    3-D body trunk extractor for sitk volume
    '''

    original_voxel_size = volume.GetSpacing()

    resized = volumeresampler.resample(volume, 0, [original_voxel_size[0], original_voxel_size[1], CALC_SLICE_THICKNESS], new_matrix_size = None, is_label = False, apply_filter=True)
    
    arr = sitk.GetArrayFromImage(resized)
    for z in range(arr.shape[0]):
        arr[z,:,:] = get_trunk_label(arr[z,:,:])
    trunk_label_thick = sitk.GetImageFromArray(arr)

    trunk_label_thick.SetSpacing(resized.GetSpacing())
    trunk_label_thick.SetOrigin(resized.GetOrigin())
    trunk_label_thick.SetDirection(resized.GetDirection())

    trunk_label = volumeresampler.resample(trunk_label_thick, 0, volume.GetSpacing(), volume.GetSize(), is_label = True)


    arr1 = sitk.GetArrayFromImage(volume)
    arr2 = sitk.GetArrayFromImage(trunk_label)
    arr1[arr2 == 0] = -1000
    new_volume = sitk.GetImageFromArray(arr1)

    new_volume.SetSpacing(volume.GetSpacing())
    new_volume.SetOrigin(volume.GetOrigin())
    new_volume.SetDirection(volume.GetDirection())

    return new_volume




if __name__ == '__main__':

    sitk.WriteImage(kill_arms(sitk.ReadImage(sys.argv[1])), sys.argv[2])

