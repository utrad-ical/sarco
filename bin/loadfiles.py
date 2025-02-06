


import sys
from pathlib import Path
import shutil
import pandas as pd

import SimpleITK as sitk



# copy .mhd files to a given work directory and build a new file list



if __name__ == '__main__':

    in_list_file_path = Path(sys.argv[1])
    out_list_file_path = Path(sys.argv[2])
    src_dir_path = Path(sys.argv[3])
    dest_dir_path = Path(sys.argv[4])

    dest_dir_path.mkdir(exist_ok=True)

    inlist = pd.read_csv(in_list_file_path, encoding='shift_jis')

    if not "{volume_name}" in inlist.columns:
        # not pandas list; read as simple txt
        inlist = pd.read_csv(in_list_file_path, encoding='shift_jis', header=None)
        inlist = inlist.set_axis(["{volume_name}"], axis="columns")

    # rename the column
    outlist = pd.DataFrame(inlist)
    outlist = outlist.rename(columns = {"{volume_name}": "{original_volume}"})

    case_name_dict = {}

    for index, line in outlist.iterrows():

        src_name = line.loc["{original_volume}"]

        # source
        src = Path(src_name)
        
        if(not src.is_absolute()):
            # resolve relative path from src_dir_path
            src = src_dir_path / src

        # resolve with current path
        src = src.resolve()

        # case name
        case_name = src.stem
        ext = src.suffix
        if(str(ext) == '.gz'):
            print(case_name[-4:])
            assert(case_name[-4:] =='.nii')
            case_name = case_name[:-4]
            ext = '.nii.gz'
        if case_name in list(case_name_dict.keys()):
            # duplicated name
            index = len(case_name_dict[case_name])
            new_case_name = case_name + "_" + str(index)
            case_name_dict[case_name].append(new_case_name)
            case_name = new_case_name
        else:
            case_name_dict[case_name] = [case_name, ]

        # dest
        dest = (dest_dir_path / case_name).with_suffix(ext)

        
        # copy
        print("%s -> %s" % (src, dest))
        # shutil.copy(src, dest)
        sitk.WriteImage(sitk.ReadImage(src), dest)

        outlist.loc[index, "{volume_name}"] = str(dest.resolve())

outlist.to_csv(out_list_file_path, encoding='shift_jis', index=False)

