import numpy as np
import scipy.ndimage
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import SimpleITK as sitk
import json
import os
import sys
from typing import List, Tuple

import extract_trunk



BODY_TRUNK_THRESHOLD_HU = -300

OUTPUTGRAYSCALEIMAGE = True



def get_visceral_region(label_muscle : np.array, voxelsize : float, debug = False):

    label_muscle = np.copy(label_muscle)

    # get the median point
#    (g_y, g_x) = scipy.ndimage.center_of_mass(label_muscle)
    g_x_where = np.where(0<np.sum(label_muscle, axis=0))
    g_x = ( np.max(g_x_where) + np.min(g_x_where) ) / 2.0
    g_y_where = np.where(0<np.sum(label_muscle, axis=1))
    g_y = ( np.max(g_y_where) + np.min(g_y_where) ) / 2.0

    # get the polar coordinate expression
    num_points = 1000
    angles = np.linspace(-np.pi, +np.pi, num_points)
    radius = np.linspace(0, 250/voxelsize, num_points)
    angles = np.repeat(angles[:,np.newaxis], num_points, axis=1)
    radius = np.repeat(radius[np.newaxis,:], num_points, axis=0)
    polarcoords_x = g_x + radius * np.cos(angles)
    polarcoords_y = g_y + radius * np.sin(angles)

    polarcoords = np.array([polarcoords_y.flatten(), polarcoords_x.flatten()])

    polar = scipy.ndimage.map_coordinates(
        label_muscle.astype('uint8'), 
        polarcoords, 
        order=0, 
        mode='nearest').reshape((num_points, num_points))

    if(debug):
        plt.imsave("polar.png", polar)

    # get the nearest and farthest points for each angle
    nearestperangle = [np.amin(np.where(0<polar[angleindex, :]), initial=num_points) for angleindex in np.arange(num_points)]
    farthestperangle = [np.amax(np.where(0<polar[angleindex, :]), initial=0) for angleindex in np.arange(num_points)]

    # gaussian-smoothing 'farthestperangle' - first pass
    modifier = (0 < np.array(farthestperangle)).astype(float)
    sigma = 60
    farthestperangle_1stpass = scipy.ndimage.gaussian_filter1d(farthestperangle, sigma, mode='wrap', truncate=10.0)
    modifier = scipy.ndimage.gaussian_filter1d(modifier, sigma, mode='wrap', truncate=10.0)
    farthestperangle_1stpass = np.floor(farthestperangle_1stpass / modifier).astype(int)

    if(debug):
        polar2 = np.copy(polar)
        for (angle_index, angle) in enumerate(np.linspace(-np.pi, +np.pi, num_points)):
            polar2[angle_index, :farthestperangle_1stpass[angle_index]] = 1
            polar2[angle_index, farthestperangle_1stpass[angle_index]:] = 0
        plt.imsave("1stpass.png", polar2)

    # kill <5cm dip
    farthestperangle = np.array(farthestperangle)
    farthestperangle[farthestperangle < farthestperangle_1stpass - 50] = 0

    if(debug):
        polar2 = np.copy(polar)
        for (angle_index, angle) in enumerate(np.linspace(-np.pi, +np.pi, num_points)):
            polar2[angle_index, :farthestperangle[angle_index]] = 1
            polar2[angle_index, farthestperangle[angle_index]:] = 0

        plt.imsave("polar1.5.png", polar2)

    if(0):
        # curve fit (polynomial)
        xx = np.arange(num_points)
        p = np.polyfit(xx[0<farthestperangle].astype(float), farthestperangle[0<farthestperangle].astype(float), 4)
        x = xx.astype(float)
        farthestperangle[0==farthestperangle] = np.poly1d(p)(x[0==farthestperangle])
    elif(0):
        # gaussian-smoothing 'farthestperangle' - second pass
        farthestperangle = farthestperangle.astype('float64')
        modifier = (0 < farthestperangle).astype('float64')
        sigma = 10
        farthestperangle = scipy.ndimage.gaussian_filter1d(farthestperangle, sigma, mode='wrap', truncate=10.0) 
        modifier = scipy.ndimage.gaussian_filter1d(modifier, sigma, mode='wrap', truncate=10.0) 
        farthestperangle = np.floor(farthestperangle / modifier).astype(int)
    else:
        # Fourier curve fit
        def f(xo, s1, s2, s3, s4, c0, c1, c2, c3, c4):
            x = xo / 500 * np.pi - np.pi
            return s1*np.sin(x) + s2*np.sin(x*2) + s3*np.sin(x*3) + s4*np.sin(x*4) + c0*np.cos(x*0) + c1*np.cos(x*1) + c2*np.cos(x*2) + c3*np.cos(x*3) + c4*np.cos(x*4)
        xx = np.arange(num_points)
        popt, _ = scipy.optimize.curve_fit(f, xx[0<farthestperangle].astype(float), farthestperangle[0<farthestperangle].astype(float))
        x = xx.astype(float)
        farthestperangle[0==farthestperangle] = f(x[0==farthestperangle], *popt)


    # write back 'farthestperangle' to polar map
    for (angle_index, angle) in enumerate(np.linspace(-np.pi, +np.pi, num_points)):
        polar[angle_index, :farthestperangle[angle_index]] = 1
        polar[angle_index, farthestperangle[angle_index]:] = 0
    if(debug):
        plt.imsave("polar2.png", polar)

    # make a visceral mask
    visceral_label = np.zeros_like(label_muscle, dtype='uint8')
    cautesian_x, cautesian_y = np.meshgrid(np.arange(label_muscle.shape[0]), np.arange(label_muscle.shape[1]))
    polar_angle = np.arctan2((cautesian_y-g_y), (cautesian_x-g_x))
    polar_radius = np.sqrt((cautesian_y-g_y)**2+(cautesian_x-g_x)**2) * voxelsize

    cautesiancoords = np.array([(polar_angle.flatten()+np.pi)/(2*np.pi)*num_points, polar_radius.flatten()*num_points/250])

    visceral_label = scipy.ndimage.map_coordinates(
        polar.astype('uint8'), 
        cautesiancoords, 
        order=0, 
        mode='nearest').reshape((label_muscle.shape[0], label_muscle.shape[1]))

    visceral_label = scipy.ndimage.binary_erosion(visceral_label, np.ones((3,3)))

    # smoothing
    visceral_label = scipy.ndimage.binary_closing(0<visceral_label, np.ones((3,3)), 5)

    return visceral_label
    




def get_largest_connected_component(input : np.array):
    labeled, labelnum = scipy.ndimage.label(input)
    volumes = [np.sum(labeled==i+1) for i in np.arange(labelnum)]
    maxlabel = np.argmax(np.array(volumes))+1
    return labeled == maxlabel



def calculate_variables(label_visceral, label_superficial, label_muscle, gray, voxel_size, label_cortical_bone=None, label_medullary_bone=None):
#  ********
# 【測定項目】

# ①骨格筋面積（L3、 L1レベル）, Skeletal muscle area, SMA　しきい値：-29 to +150 HU　

# ②骨格筋濃度（L3、L1レベル）, 
# 　①のしきい値を適応後のROI内で、mean skeletal muscle density, mean SMD; 　mean muscle attenuation, mean MA

# ③内臓脂肪面積（L3、L1レベル）, vesceral adipose tissue index (VATI)　しきい値：-150 to -50 for visceral adipose tissue

# ④皮下脂肪面積（L3、L1レベル）, subcutaneous adipose tissue index (SATI),　しきい値：-190 to -30 for subcutaneous adipose tissue    　　https://journals.physiology.org/doi/full/10.1152/jappl.1998.85.1.115
#         →　内臓脂肪面積/皮下脂肪面積　比　 visceral to subcutaneous adipose tissue area ratios (VSR) 　は副次的に発生

# ⑤骨格筋内の明瞭な脂肪の割合（L3、L1レベル）　今回semiautoで作成したROI（①のしきい値適応前）内の -30 HU 以下の面積/ROI面積
#         →　骨格筋内の脂肪のサシの入り具合を見る

# しきい値まとめ
# -29 to +150 HU for skeletal muscle,            相当メジャーなしきい値でこれを用います。
# -190 to -30 for subcutaneous adipose tissue    https://journals.physiology.org/doi/full/10.1152/jappl.1998.85.1.115
# -150 to -50 for visceral adipose tissue　皮下と異なるようなのだが、根拠をきちんと調べていない。
# ********

    if np.sum(label_muscle) == 0:
        return {
            "SMA": 0, 
            "MSMD": 0, 
            "VAT": 0, 
            "SAT": 0,
            "SASHI": 0,
            "stdSMD": 0,
            "meanVAT": 0,
            "stdVAT": 0,
            "meanSAT": 0,
            "stdSAT": 0
        }

    SMA_ROI = np.logical_and(np.logical_and(-29<gray, gray<150), label_muscle)
    SMA = np.sum(SMA_ROI) * voxel_size**2
    MSMD = np.mean(gray[SMA_ROI])
    VAT_ROI = np.logical_and(np.logical_and(-150<gray, gray<-50), label_visceral)
    VAT = np.sum(VAT_ROI) * voxel_size**2
    SAT_ROI = np.logical_and(np.logical_and(-190<gray, gray<-30), label_superficial)
    SAT = np.sum(SAT_ROI) * voxel_size**2
    SASHI_ROI = np.logical_and(gray<-30, label_muscle)
    SASHI = np.sum(SASHI_ROI) / np.sum(label_muscle)

    stdSMD = np.std(gray[SMA_ROI])
    meanVAT = np.mean(gray[VAT_ROI])
    stdVAT = np.std(gray[VAT_ROI])
    meanSAT = np.mean(gray[SAT_ROI])
    stdSAT = np.std(gray[SAT_ROI])

    if(np.sum(label_cortical_bone) == 0):
        label_cortical_bone = None
    if(np.sum(label_medullary_bone) == 0):
        label_medullary_bone = None

    if(label_cortical_bone is not None and label_medullary_bone is not None):
        cortical_area = np.sum(label_cortical_bone) * voxel_size**2
        medullary_area = np.sum(label_medullary_bone) * voxel_size**2
        cortical_density = np.mean(gray[label_cortical_bone]) if label_cortical_bone is not None else 0
        medullary_density = np.mean(gray[label_medullary_bone]) if label_medullary_bone is not None else 0
        cortical_std = np.std(gray[label_cortical_bone]) if label_cortical_bone is not None else 0
        medullary_std = np.std(gray[label_medullary_bone]) if label_medullary_bone is not None else 0
        cortical_skewness = scipy.stats.skew(gray[label_cortical_bone].flatten()) if label_cortical_bone is not None else 0
        medullary_skewness = scipy.stats.skew(gray[label_medullary_bone].flatten()) if label_medullary_bone is not None else 0
        cortical_kurtosis = scipy.stats.kurtosis(gray[label_cortical_bone].flatten()) if label_cortical_bone is not None else 0
        medullary_kurtosis = scipy.stats.kurtosis(gray[label_medullary_bone].flatten()) if label_medullary_bone is not None else 0

        return {
            "SMA": SMA,
            "MSMD": MSMD,
            "VAT": VAT,
            "SAT": SAT,
            "SASHI": SASHI,
            "stdSMD": stdSMD,
            "meanVAT": meanVAT,
            "stdVAT": stdVAT,
            "meanSAT": meanSAT,
            "stdSAT": stdSAT,
            "cortical_area": cortical_area,
            "medullary_area": medullary_area,
            "cortical_density": cortical_density,
            "medullary_density": medullary_density,
            "cortical_std": cortical_std,
            "medullary_std": medullary_std,
            "cortical_skewness": cortical_skewness,
            "medullary_skewness": medullary_skewness,
            "cortical_kurtosis": cortical_kurtosis,
            "medullary_kurtosis": medullary_kurtosis
        }

    else: 
        return {
            "SMA": SMA, 
            "MSMD": MSMD, 
            "VAT": VAT, 
            "SAT": SAT,
            "SASHI": SASHI,
            "stdSMD": stdSMD,
            "meanVAT": meanVAT,
            "stdVAT": stdVAT,
            "meanSAT": meanSAT,
            "stdSAT": stdSAT
        }



def calculate(grayimage : sitk.Image,
              labelimage : sitk.Image,
              trunk_mask : np.array = None
              ) -> Tuple[np.array, np.array, dict]:
    '''
    
    return (volume[z,y,x], png[y,x, rgb], metrics).
    each voxel in the 3-d array consists 0 (background), 3 (muscle), 1 (visceral fat) or 2 (superficial fat). 
    
    '''
    
    grayarray = sitk.GetArrayFromImage(grayimage)
    labelarray = sitk.GetArrayFromImage(labelimage)

    voxel_size = grayimage.GetSpacing()[0]
    
    # get the slice
    zsum = np.sum(labelarray == 1, axis=(1,2))
    if(np.sum(zsum) == 0):
        # raise ValueError('Failed: null label (%s, %s)'% (gray_volume_path, result_volume_path))
        z = 0
        gray = np.zeros_like(grayarray[z,:,:])
        label_muscle = np.zeros_like(labelarray[z,:,:])
        label_visceral = np.zeros_like(labelarray[z,:,:])
        label_superficial = np.zeros_like(labelarray[z,:,:])
        label_cortical = np.zeros_like(labelarray[z,:,:])
        label_medullary = np.zeros_like(labelarray[z,:,:])
    else:
        z = np.argmax(zsum)
        print(f"{z=}")

        # get muscle image
        label_muscle = labelarray[z,:,:] == 1

        # get vertebral images
        label_cortical = labelarray[z,:,:] == 2
        label_medullary = labelarray[z,:,:] == 3

        # trunk_mask: extract_trunk.trunk_mask(grayimage) if the caller has it
        gray_arm_removed = sitk.GetArrayFromImage(extract_trunk.kill_arms(grayimage, trunk_mask))
        gray = gray_arm_removed[z,:,:]
        label_trunk = BODY_TRUNK_THRESHOLD_HU < gray
        
        # get visceral region label
        label_visceral = get_visceral_region(label_muscle, voxel_size)

        # exclusion
        label_visceral[0<label_muscle] = 0
        label_superficial = np.copy(label_trunk)
        label_superficial[0<label_visceral] = 0
        label_superficial[0<label_muscle] = 0

    # multilabel
    label_mix = label_visceral + label_superficial*2 + label_muscle*3 + label_cortical*4 + label_medullary*8
    
    # tricolor
    
    visceral_border = scipy.ndimage.binary_dilation(label_visceral).astype(float) - scipy.ndimage.binary_erosion(label_visceral).astype(float)
    muscle_border = scipy.ndimage.binary_dilation(label_muscle).astype(float) - scipy.ndimage.binary_erosion(label_muscle).astype(float)
    
    border = visceral_border + muscle_border

    border[::2,1::2] *= -1
    border[1::2,::2] *= -1

    png = np.clip((gray+200)//5 + border*200, 0, 255).astype('uint8')

    #ver 2
    
    SMA_ROI = np.logical_and(np.logical_and(-29<gray, gray<150), label_muscle)
    VAT_ROI = np.logical_and(np.logical_and(-150<gray, gray<-50), label_visceral)
    SAT_ROI = np.logical_and(np.logical_and(-190<gray, gray<-30), label_superficial)
    SASHI_ROI = np.logical_and(gray<-30, label_muscle)
    
    png1 = np.clip(255 * SMA_ROI, 0, 255).astype('uint8')
    png2 = np.clip(255 * VAT_ROI, 0, 255).astype('uint8')
    png3 = np.clip(255 * SAT_ROI, 0, 255).astype('uint8')
    png4 = np.clip((SMA_ROI*3+VAT_ROI*1+SAT_ROI*2)*255//3, 0, 255).astype('uint8')
    
    png5 = png
    png6 = np.clip((gray+200)//5, 0, 255).astype('uint8')
    
    png7 = np.clip(np.clip((gray[:,:]+200)//5, 0, 127) + label_mix[:,:]*24, 0, 255).astype('uint8')

    png8 = label_mix[:,:]*21 - (label_mix[:,:]==4)*36


    png = np.concatenate( (np.concatenate((png1,png2), axis=1), np.concatenate((png3,png4), axis=1), np.concatenate((png5,png6), axis=1), np.concatenate((png7,png8), axis=1)), axis=0)
    
    
    
    
    
    # volume
    volume = np.zeros_like(labelarray)
    volume[z,:,:] = label_mix
    
    return (volume, 
            png, 
            calculate_variables(
                label_visceral, 
                label_superficial, 
                label_muscle, 
                gray, 
                voxel_size,
                label_cortical_bone=label_cortical,
                label_medullary_bone=label_medullary)
            )







if __name__ == '__main__':

    if len(sys.argv)!=5:
        print("Usage: %s [InGrayFile] [InSarcoBinaryFile] [outTricolorPngFile] [outCsvFile]")

    inGrayFile = sitk.ReadImage(sys.argv[1])
    inBinaryFile = sitk.ReadImage(sys.argv[2])

    _, png, variables = calculate(inGrayFile, inBinaryFile)
    
    plt.imsave(sys.argv[3], png)
    
    pd.DataFrame.from_dict(variables).to_csv(sys.argv[4], encoding='shift-jis')



