
import numpy as np
import torch
import torch.nn
import scipy
import scipy.ndimage
import SimpleITK as sitk
from pathlib import Path
import sys
import matplotlib.pyplot as plt

def devide_vertebral_body_into_cortical_and_medullary_parts(
    input_grayscale: sitk.Image,
    lumbar_vertebra_mask: sitk.Image
) -> tuple[sitk.Image, sitk.Image]:
    """
    Divide the vertebral body into cortical and medullary parts.

    Args:
        lumbar_vertebra_mask (sitk.Image): The mask of the lumbar vertebrae.

    Returns:
        tuple[sitk.Image, sitk.Image]: The cortical and medullary masks.
    """
    # Implementation for dividing the vertebral body

    # get the bounding box of the lumbar vertebra mask
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(lumbar_vertebra_mask)
    try:
        bounding_box = stats.GetBoundingBox(1)
    except:
        # no vertebral label
        return lumbar_vertebra_mask, lumbar_vertebra_mask

    # resize to isotropic spacing
    rif = sitk.ResampleImageFilter()

    original_origin = lumbar_vertebra_mask.GetOrigin()
    original_spacing = lumbar_vertebra_mask.GetSpacing()
    original_direction = lumbar_vertebra_mask.GetDirection()

    temp_spacing = [0.5, 0.5, 0.5,]
    temp_origin = [
        bounding_box[0] * original_spacing[0] * original_direction[0] + original_origin[0],
        bounding_box[1] * original_spacing[1] * original_direction[4] + original_origin[1],
        bounding_box[2] * original_spacing[2] * original_direction[8] + original_origin[2]
    ]
    temp_size = [
        int(np.round(bounding_box[3] * (original_spacing[0] / temp_spacing[0]))),
        int(np.round(bounding_box[4] * (original_spacing[1] / temp_spacing[1]))),
        int(np.round(bounding_box[5] * (original_spacing[2] / temp_spacing[2])))
    ]

    rif.SetOutputSpacing(temp_spacing)
    rif.SetSize(temp_size)
    rif.SetOutputOrigin(temp_origin)
    rif.SetOutputDirection(lumbar_vertebra_mask.GetDirection())
    rif.SetInterpolator(sitk.sitkNearestNeighbor)
    rif.SetTransform(sitk.Transform())

    resized_vertebra_mask = rif.Execute(lumbar_vertebra_mask)

    rif.SetInterpolator(sitk.sitkBSpline)
    resized_input_grayscale = rif.Execute(input_grayscale)

    # apply binary erosion to the vertebrae mask to get the medullary part
    beif = sitk.BinaryErodeImageFilter()
    beif.SetKernelRadius(5)# voxels
    beif.SetForegroundValue(1)
    beif.SetBackgroundValue(0)

    resized_medullary_mask = beif.Execute(resized_vertebra_mask)
    resized_cortical_mask = sitk.Subtract(resized_vertebra_mask, resized_medullary_mask)



    # refine the masks by GAC
    gaclsf = sitk.GeodesicActiveContourLevelSetImageFilter()

    smoothed_gray = sitk.CurvatureAnisotropicDiffusion(sitk.Cast(resized_input_grayscale, sitk.sitkFloat32), timeStep=0.03125, conductanceParameter=3.0, numberOfIterations=5)
    gradient_magnitude = sitk.GradientMagnitude(smoothed_gray)
    edge_potential = sitk.Sigmoid(
        gradient_magnitude, 
        alpha=-50, 
        beta=50, # delta HU
        outputMaximum=1.0, 
        outputMinimum=0.0
    )

    gaclsf.SetPropagationScaling(0.05)
    gaclsf.SetCurvatureScaling(0.25)
    gaclsf.SetAdvectionScaling(1.0)
    gaclsf.SetMaximumRMSError(0.02)
    gaclsf.SetNumberOfIterations(100)

    resized_medullary_mask_levelset = sitk.SignedMaurerDistanceMap(resized_medullary_mask, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)

    resized_medullary_mask = gaclsf.Execute(resized_medullary_mask_levelset, edge_potential) < 0

    resized_cortical_mask = sitk.Subtract(resized_vertebra_mask, resized_medullary_mask)

    return sitk.Resample(resized_medullary_mask, lumbar_vertebra_mask), sitk.Resample(resized_cortical_mask, lumbar_vertebra_mask)



def compose_masks(
    grayscale_path : Path,
    segmentations_path : Path
) -> sitk.Image:
    """
    Compose the masks of the lumbar vertebrae and spinal cord into a single mask.

    Args:
        segmentations_path (Path): The path to the directory containing the segmentation masks. 

    Returns:
        sitk.Image: The composed mask image.
    """

    # load the input grayscale image
    input_grayscale = sitk.ReadImage(str(grayscale_path))

    # Load the lumbar vertebrae mask
    vertebra_mask_pathes = [
        segmentations_path / "vertebrae_T12.nii.gz",
        segmentations_path / "vertebrae_L1.nii.gz",
        segmentations_path / "vertebrae_L2.nii.gz",
        segmentations_path / "vertebrae_L3.nii.gz",
        segmentations_path / "vertebrae_L4.nii.gz"
    ]
    spinal_cord_mask_path = segmentations_path / "spinal_cord.nii.gz"

    # devide into cortical and medullary parts
    cortical_maskes = []
    medullary_maskes = []
    for p in vertebra_mask_pathes:
        mask = sitk.ReadImage(str(p))
        medullary, cortical = devide_vertebral_body_into_cortical_and_medullary_parts(input_grayscale, mask)
        cortical_maskes.append(cortical)
        medullary_maskes.append(medullary)
    
    # union of the vertebra masks
    cortical_mask = sitk.Image(cortical_maskes[0].GetSize(), cortical_maskes[0].GetPixelID())
    cortical_mask.CopyInformation(cortical_maskes[0])
    medullary_mask = sitk.Image(medullary_maskes[0].GetSize(), medullary_maskes[0].GetPixelID())
    medullary_mask.CopyInformation(medullary_maskes[0])

    for mask in cortical_maskes:
        cortical_mask = sitk.Or(cortical_mask, mask)

    for mask in medullary_maskes:
        medullary_mask = sitk.Or(medullary_mask, mask)

    # cut anterior part of the vertebrae
    spinal_cord_mask = sitk.ReadImage(str(spinal_cord_mask_path))
    arr_spinal_cord = sitk.GetArrayFromImage(spinal_cord_mask)

    # create a mask for the anterior part of the vertebrae
    arr_cortical = sitk.GetArrayFromImage(cortical_mask)
    arr_medullary = sitk.GetArrayFromImage(medullary_mask)

    for z in range(arr_spinal_cord.shape[0]):
        spinal_cord_slice = arr_spinal_cord[z, :, :]
        cortical_slice = arr_cortical[z, :, :]
        medullary_slice = arr_medullary[z, :, :]

        # find the anterior boundary of the spinal cord
        spinal_cord_boundary = np.where(spinal_cord_slice > 0)
        if len(spinal_cord_boundary[0]) > 0:
            anterior_boundary = np.min(spinal_cord_boundary[0])
            # cut the anterior part of the vertebrae
            cortical_slice[anterior_boundary:, :] = 0
            medullary_slice[anterior_boundary:, :] = 0

    # convert back to SimpleITK images
    cortical_mask = sitk.GetImageFromArray(arr_cortical)
    cortical_mask.CopyInformation(spinal_cord_mask)
    medullary_mask = sitk.GetImageFromArray(arr_medullary)
    medullary_mask.CopyInformation(spinal_cord_mask)

    return cortical_mask*2 + medullary_mask*3



def read_label_image_safe(filepath, another_label_image):
    try:
        img = sitk.ReadImage(filepath)
    except:
        another_label_image
        img = sitk.Image(another_label_image.GetSize(), another_label_image.GetPixelID())
        img.CopyInformation(another_label_image)
    
    return img


def main():

    # get vertebral mask
    vertebra_mask = compose_masks(
        grayscale_path=Path(sys.argv[1]),
        segmentations_path=Path(sys.argv[2])
    )
    arr_vertebra_mask = sitk.GetArrayFromImage(vertebra_mask)

    # get muscle masks
    L1_muscle_mask = read_label_image_safe(str(Path(sys.argv[3])), vertebra_mask)
    L3_muscle_mask = read_label_image_safe(str(Path(sys.argv[4])), vertebra_mask)

    # get z
    arr_L1_muscle_mask = sitk.GetArrayFromImage(L1_muscle_mask)
    arr_L3_muscle_mask = sitk.GetArrayFromImage(L3_muscle_mask)

    L1_muscle_boundary = np.where(arr_L1_muscle_mask > 0)
    L3_muscle_boundary = np.where(arr_L3_muscle_mask > 0)

    arr_gray = sitk.GetArrayFromImage(sitk.ReadImage(Path(sys.argv[1])))

    # label merge
    if len(L1_muscle_boundary[0]) > 0:
        L1_muscle_z = np.min(L1_muscle_boundary[0])
        arr_L1_muscle_mask[L1_muscle_z, :, :] = np.maximum(arr_L1_muscle_mask[L1_muscle_z, :, :], arr_vertebra_mask[L1_muscle_z, :, :])
        if(6<len(sys.argv)):
            gray_img = np.clip((arr_gray[L1_muscle_z, :, :] + 100) / 500 * 0.6, 0.0, 0.6)
            gray_img = np.stack([gray_img, gray_img, gray_img], axis=-1)
            color_img = np.stack((
                arr_L1_muscle_mask[L1_muscle_z, :, :] == 1, 
                arr_L1_muscle_mask[L1_muscle_z, :, :] == 2, 
                arr_L1_muscle_mask[L1_muscle_z, :, :] == 3
            ), axis=-1) * 0.4

            plt.imsave(str(Path(sys.argv[7])), np.clip(gray_img + color_img, 0., 1.))

    if len(L3_muscle_boundary[0]) > 0:
        L3_muscle_z = np.min(L3_muscle_boundary[0])
        arr_L3_muscle_mask[L3_muscle_z, :, :] = np.maximum(arr_L3_muscle_mask[L3_muscle_z, :, :], arr_vertebra_mask[L3_muscle_z, :, :])
        if(6<len(sys.argv)):
            gray_img = np.clip((arr_gray[L3_muscle_z, :, :] + 100) / 500 * 0.6, 0.0, 0.6)
            gray_img = np.stack([gray_img, gray_img, gray_img], axis=-1)
            color_img = np.stack((
                arr_L3_muscle_mask[L3_muscle_z, :, :] == 1, 
                arr_L3_muscle_mask[L3_muscle_z, :, :] == 2, 
                arr_L3_muscle_mask[L3_muscle_z, :, :] == 3
            ), axis=-1) * 0.4

            plt.imsave(str(Path(sys.argv[8])), np.clip(gray_img + color_img, 0., 1.))

    # convert back to SimpleITK images
    L1_muscle_mask = sitk.GetImageFromArray(arr_L1_muscle_mask)
    L1_muscle_mask.CopyInformation(vertebra_mask)
    L3_muscle_mask = sitk.GetImageFromArray(arr_L3_muscle_mask)
    L3_muscle_mask.CopyInformation(vertebra_mask)

    # write
    sitk.WriteImage(L1_muscle_mask, str(Path(sys.argv[5])))
    sitk.WriteImage(L3_muscle_mask, str(Path(sys.argv[6])))



if __name__ == "__main__":
    main()

