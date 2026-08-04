
import sys

import numpy as np

import SimpleITK as sitk

import pydicom

from pathlib import Path
import shutil



# input / output
input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
if 3<len(sys.argv):
    work_path = Path(sys.argv[3])
else:
    work_path = Path('./temp5y92384')



# NG words (for exclusion of unusable series)
NG_words = { "cor", "sag", "pet", "mr", "mip" }



# minimal slice number
min_file_num = 20



# thickness min / max (mm)
thickness_min = 0.9
thickness_max = 5.1



# file name renamer
trans = str.maketrans({':': '_colon_', '/': '_slash_', '\\': '_backslash_', 
                       ' ': '_space_', '<': '_lesser_', '>': '_greater_',
                       '*': '_asterisk_', '?': '_question_', '|': '_verticalbar_'})





###################################################################





reader = sitk.ImageSeriesReader()



output_path.mkdir(exist_ok=True)
work_path.mkdir(exist_ok=True)







for object in input_path.glob('**'):

    if(Path.is_dir(object)):

        file_list = [file for file in object.iterdir() if file.is_file()]
        file_num =  len(file_list)
        
        # NG word killer
        for NG_word in NG_words:
            if NG_word in str(object).lower():
                print("including NG word %s" % NG_word)
                continue
        if file_num < min_file_num:
            print("file num lesser than %d" % min_file_num)
            continue
        
        
        
        # (maybe) 3-d CT volume
        
        volume_name = str(object.relative_to(input_path)).translate(trans) + '.nii.gz'
        dest = str((output_path / volume_name).resolve())
        #print(volume_name)
        
#        if Path(dest).exists():
#            print("pass")
#            continue
        
        
        
        # make a work dir and copy
        temp_files = []
        for file in file_list:
            if file.suffix.casefold() == '.txt'.casefold():
                # kill text files
                continue
            temp_file = work_path / Path('temp_dcm%08.8d.dcm' % len(temp_files))
            shutil.copy(file, temp_file)
            temp_files.append(str(temp_file.resolve()))
        
        
        
        # get (sub)series identifier
        series_dict = {}
        for temp_file in temp_files:
            with pydicom.dcmread(temp_file, force=True) as df:
                series_identifier = ""
                try:
                    series_identifier += "_ser%d" % df.SeriesNumber
                except:
                    pass # some image has no tag...
                try:
                    series_identifier += "_acq%d" % df.AcquisitionNumber
                except:
                    pass # some image has no tag...
                try:
                    series_identifier += "_spc%fx%f" % (df.PixelSpacing[0], df.PixelSpacing[1])
                except:
                    pass # some image has no tag...
                try:
                    series_identifier += "_mat%dx%d" % (df.Rows, df.Columns)
                except:
                    pass # some image has no tag...
                try:
                    series_identifier += "_ori%d%d%d%d%d%d" % tuple(map(int,df.ImageOrientationPatient))
                except:
                    pass # some image has no tag...
            if series_identifier not in series_dict.keys():
                series_dict[series_identifier] = [temp_file,]
            else:
                series_dict[series_identifier].append(temp_file)
            
            
            
        # convert each series
        for i, (series_identifier, series_list) in enumerate(series_dict.items()):
            if len(series_list)<20:
                continue
            
            # get seriesdescription
            with pydicom.dcmread(series_list[0], force=True) as df:
                try:
                    series_description = str(df.SeriesDescription).translate(trans)
                except:
                    series_description = "(none)"
            
            for NG_word in NG_words:
                if NG_word in str(series_description).lower():
                    print("including NG word %s" % NG_word)
                    continue
            
            if 1:
                # spatial sort
                coords = np.ndarray((3, len(series_list)))
                for i, file in enumerate(series_list):
                    with pydicom.dcmread(file, force=True) as df:
                        try:
                            coords[:,i] = np.array(df.ImagePositionPatient)
                        except:
                            pass
                if(coords.flatten().sum() == 0):
                    print("missing coordinates")
                    continue
                
                coordcov = np.cov(coords)
                assert np.all( np.array(coordcov.shape) == 3 )
                try:
                    e, v = np.linalg.eigh(coordcov)
                except:
                    print('eigh fail (non-crosssectional?)')
                    continue
                vmax = v[np.argmax(e)]]
                
                if vmax[2]<0:
                    vmax *= -1 # invert so that z increases as instance number increases
                
                dislocs = np.sum(coords * vmax[:,np.newaxis], axis=0) # dislocations along the 1st eigenvector
                
                series_list = [series_list[k] for k in list(np.argsort(dislocs))]
                
            if 1:
                # estimte slice thickness
                with pydicom.dcmread(series_list[0], force=True) as df:
                    try:
                        pos_1 = np.array(df.ImagePositionPatient)
                    except:
                        print("Fail")
                        continue
                
                with pydicom.dcmread(series_list[1], force=True) as df:
                    try:
                        pos_2 = np.array(df.ImagePositionPatient)
                    except:
                        print("Fail")
                        continue
                
                delta = np.abs(pos_1 - pos_2)
                if 0.5<delta[0] or 0.5<delta[1]:
                    print("not axial")
                    continue
                thickness = delta[2]
                #print("thickness = %f" % thickness)
                if thickness<thickness_min or thickness_max<thickness:
                    print("thickness is not %f - %fmm" % (thickness_min, thickness_max))
                    continue
                

            if len(series_dict) == 1:
                series_suffix = "_seriesdesc_%s_slicethickness_%f" %(series_description, thickness)
            else:
                series_suffix = "_seriesdesc_%s_subseries_%d_slicethickness_%f" % (series_description, i, thickness)
            
            volume_name = str(object.relative_to(input_path)).translate(trans) + series_suffix + '.nii.gz'
            dest = str((output_path / volume_name).resolve())

            print(dest)
            
            # convert it
            image = None
            reader.SetImageIO("GDCMImageIO")
            reader.SetFileNames(series_list)
            try:
                image = reader.Execute()
            except BaseException as e:
                print(e)
            if image is not None:
                temp_niigz = work_path / Path('temp.nii.gz')
                sitk.WriteImage(image, str(temp_niigz.resolve()))
                shutil.copy(temp_niigz, dest)




