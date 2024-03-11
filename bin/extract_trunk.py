


import sys

import numpy as np

import SimpleITK as sitk
import scipy.ndimage



def extract(image: np.array) -> np.array:
    '''
    2-D body trunk extractor
    '''
    image = np.copy(image)
    
    # enhance gap
    gapenhancer = np.maximum(scipy.ndimage.gaussian_laplace(image, sigma=1), 0) * 10
    gapenhancer[0 < image] = 0
    image_l = image - gapenhancer
    
    # get foreground (HU>-300) mask
    mask = image_l > -300
    
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
    
    # kill background
    image[holefilled == 0] = -1000
    
    return image
    



