

import numpy as np
import pandas as pd
import SimpleITK as sitk

import prepcache



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



def paths(line):
    '''(gray path, label path or None) of one row of a volume list'''
    gray_path = line["{volume_name}"].strip()
    if "{label_volume_name}" in line:
        return gray_path, line["{label_volume_name}"].strip()
    return gray_path, None



def read_list(volume_list):
    '''a volume list as a DataFrame, from its file path or as it is'''
    if isinstance(volume_list, pd.DataFrame):
        return volume_list
    return pd.read_csv(volume_list, encoding='shift_jis')



def drop_empty_labels(volume_list, labelused : int, cache_dir = None) -> pd.DataFrame:
    '''
    the rows of a training list whose label file has at least one voxel of
    labelused. a case without the label teaches every step something wrong:
    lumbar trains on "nothing here", and lumbarmodifier and sarcosegmentation
    take slice 0 as the true slice and train on an empty label. excluded
    cases are listed so that the cohort does not shrink silently.
    '''
    df = read_list(volume_list)
    if "{label_volume_name}" not in df.columns:
        print("[warn] the list has no {label_volume_name} column; nothing excluded")
        return df

    keep = []
    for (_, line) in df.iterrows():
        _, label_path = paths(line)
        result = prepcache.cached(cache_dir, "label_nonempty",
                                  {"label": prepcache.file_signature(label_path),
                                   "labelused": labelused},
                                  lambda: {"nonempty": np.any(sitk.GetArrayFromImage(sitk.ReadImage(label_path)) == labelused)})
        keep.append(bool(result["nonempty"]))

    keep = np.array(keep)
    excluded = df[~keep]
    if len(excluded):
        print(f"[info] excluded {len(excluded)}/{len(df)} cases without label {labelused}:")
        for (_, line) in excluded.iterrows():
            print("    " + str(paths(line)[1]))
    if not keep.any():
        raise ValueError(f"no case in the list has label {labelused}")

    return df[keep].reset_index(drop=True)



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



