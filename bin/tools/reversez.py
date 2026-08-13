import SimpleITK as sitk

import sys

name = sys.argv[1]

i = sitk.ReadImage(name + ".nii.gz")

d = sitk.GetArrayFromImage(i)

d = d[::-1]

ii = sitk.GetImageFromArray(d)

ii.SetOrigin(i.GetOrigin())
ii.SetSpacing(i.GetSpacing())

ii = sitk.DICOMOrient(ii, "LPI")

sitk.WriteImage(ii, name + ".nii.gz")

