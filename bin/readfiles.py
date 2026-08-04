

import numpy as np
import pandas as pd
import SimpleITK as sitk



def readone(line):
    filepath_gray = line["{volume_name}"]
    
    if "{label_volume_name}" in line:
        filepath_label = line["{label_volume_name}"]

        print("now reading" + filepath_gray + " & " + filepath_label)
        
        filepath_gray = filepath_gray.strip()
        filepath_label = filepath_label.strip()
        labelImage = sitk.ReadImage(filepath_label)
        grayImage = sitk.ReadImage(filepath_gray)

    else:
        print("now reading " + filepath_gray)
        
        # create void label volume
        filepath_gray = filepath_gray.strip()
        grayImage = sitk.ReadImage(filepath_gray)
        filepath_label = None
        labelImage = sitk.GetImageFromArray(np.zeros(
                                                     (grayImage.GetSize()[2],
                                                      grayImage.GetSize()[1], 
                                                      grayImage.GetSize()[0]
                                                     )
                                                    )
                                           )
        labelImage.SetDirection(grayImage.GetDirection())
        labelImage.SetOrigin(grayImage.GetOrigin())
        labelImage.SetSpacing(grayImage.GetSpacing())

    if 1<grayImage.GetNumberOfComponentsPerPixel():
        # multichannel; use the 1st channel
        grayArray = sitk.GetArrayFromImage(grayImage)[:,:,:,0]
        grayImage2 = sitk.GetImageFromArray(grayArray)
        grayImage2.SetDirection(grayImage.GetDirection())
        grayImage2.SetOrigin(grayImage.GetOrigin())
        grayImage2.SetSpacing(grayImage.GetSpacing())
        grayImage = grayImage2

#    grayImage = sitk.DICOMOrient(grayImage, 'LPI')
#    labelImage = sitk.DICOMOrient(labelImage, 'LPI')

    return (grayImage, labelImage)



def reader(volume_list_file : str):
    
    df = pd.read_csv(volume_list_file, encoding='shift_jis')
    
    # df = df[["volume_name", "label_volume_name"]]
    
    for (_, line) in df.iterrows():
        yield readone(line)



def counter(volume_list_file : str):
    
    df = pd.read_csv(volume_list_file, encoding='shift_jis')
    
    # df = df[["volume_name", "label_volume_name"]]
    
    for i, (_, line) in enumerate(df.iterrows()):
        yield i



