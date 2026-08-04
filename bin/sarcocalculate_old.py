import numpy as np
import scipy.ndimage
import scipy.optimize
import matplotlib.pyplot as plt
import imageio
import SimpleITK as sitk
import json
import os
import sys



BODY_TRUNK_THRESHOLD_HU = -300



def get_visceral_region(label_muscle : np.array, voxelsize : float, debug = False):

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



def calculate_variables(label_visceral, label_superficial, label_muscle, gray, voxel_size):
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

    SMA_ROI = np.logical_and(np.logical_and(-29<gray, gray<150), label_muscle)
    SMA = np.sum(SMA_ROI) * voxel_size**2
    MSMD = np.mean(gray[SMA_ROI])
    VATI_ROI = np.logical_and(np.logical_and(-150<gray, gray<-50), label_visceral)
    VATI = np.sum(VATI_ROI) * voxel_size**2
    SATI_ROI = np.logical_and(np.logical_and(-190<gray, gray<-30), label_superficial)
    SATI = np.sum(SATI_ROI) * voxel_size**2
    SASHI_ROI = np.logical_and(gray<-30, label_muscle)
    SASHI = np.sum(SASHI_ROI) / np.sum(label_muscle)

    return SMA, MSMD, VATI, SATI, SASHI



def sarcocalculate(dirpath : str):

    grayimage = sitk.ReadImage(os.path.join(dirpath, 'vol000.mhd'))
    labelimage = sitk.ReadImage(os.path.join(dirpath, 'vol000-label.mhd'))

    grayarray = sitk.GetArrayFromImage(grayimage)
    labelarray = sitk.GetArrayFromImage(labelimage)

    voxel_size = grayimage.GetSpacing()[0]

    j = json.load(open(os.path.join(dirpath, 'data.json'), 'rt'))

    for labelzeroindex, labeldesc in enumerate(j["revision"]["series"][0]["labels"]):
        labelname = labeldesc["name"]
        labelindex = labelzeroindex + 1

        # get the slice
        zsum = np.sum(labelarray == labelindex, axis=(1,2))
        if(np.sum(zsum) == 0):
            raise ValueError('Failed: null label (%s @ %s)'% (labelname, dirpath))
        z = np.argmax(zsum)
        if(labelarray.shape[1]*labelarray.shape[2]*0.8 < np.sum(labelarray[z,:,:] == labelindex)):
            # errornous label; kill the slice
            labelarray[z,:,:] = 0
            zsum = np.sum(labelarray == labelindex, axis=(1,2))
            z = np.argmax(zsum)

        gray = grayarray[z,:,:]

        # plt.imsave("%s_gray.png" % labelname, gray)

        label_muscle = labelarray[z,:,:] == labelindex

        # plt.imsave("%s_label_muscle.png" % labelname, label_muscle)

        # get body trunk label
        label_trunk = get_largest_connected_component(BODY_TRUNK_THRESHOLD_HU < gray)
        label_trunk = scipy.ndimage.binary_fill_holes(label_trunk)
        label_trunk = scipy.ndimage.binary_opening(label_trunk, np.ones((3,3)))
        label_trunk = scipy.ndimage.binary_erosion(label_trunk, np.ones((3,3)))
        label_trunk = scipy.ndimage.binary_erosion(label_trunk, np.ones((3,3)), iterations=12)
        label_trunk = get_largest_connected_component(label_trunk)
        label_trunk = scipy.ndimage.binary_dilation(label_trunk, np.ones((3,3)), iterations=12)

        # plt.imsave("%s_trunk.png" % labelname, label_trunk)

        # get visceral region label
        label_visceral = get_visceral_region(label_muscle, voxel_size)

        # plt.imsave("%s_visceral.png" % labelname, label_visceral)

        # exclusion
        label_visceral[0<label_muscle] = 0
        label_superficial = label_trunk
        label_superficial[0<label_visceral] = 0
        label_superficial[0<label_muscle] = 0

        # three-color
        label_mix = label_visceral + label_superficial*2 + label_muscle*3
        # plt.imsave(os.path.join(dirpath, "%s_mix.png" % labelname), 
        #     np.concatenate((
        #         label_mix[:,:,np.newaxis]*42, 
        #         np.clip((gray[:,:,np.newaxis]+200)//5, 0, 255), 
        #         (label_mix[:,:,np.newaxis]==1)*128
        #     ), axis=2).astype('uint8')
        # )
        imageio.imwrite(os.path.join(dirpath, "%s_mix.png" % labelname), 
            np.concatenate((
                label_mix[:,:,np.newaxis]*42, 
                np.clip((gray[:,:,np.newaxis]+200)//5, 0, 255), 
                (label_mix[:,:,np.newaxis]==1)*128
            ), axis=2).astype('uint8')
        )

        (SMA, MSMD, VATI, SATI, SASHI) = calculate_variables(label_visceral, label_superficial, label_muscle, gray, voxel_size)

        print("%s, %s, %f, %f, %f, %f, %f" % (dirpath, labelname, SMA, MSMD, VATI, SATI, SASHI))



if __name__ == '__main__':

    sarcocalculate(sys.argv[1])
