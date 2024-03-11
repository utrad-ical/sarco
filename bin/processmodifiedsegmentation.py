
import sys

import numpy as np
import matplotlib.pyplot as plt
import imageio
import pandas as pd

import SimpleITK as sitk



import readfiles
import sarcocalculate
import makeill

from pathlib import Path 


if __name__ == '__main__':

    if len(sys.argv)==1:
        print("Usage: %s [volume_list_file] [overwrited_label_file_path] [output_path]", sys.argv[0])
    
    df = pd.read_csv(sys.argv[1], encoding='shift_jis')
    
    label_file_overwriter_path = Path(sys.argv[2])
    
    output_path = Path(sys.argv[3])
    output_path.mkdir(exist_ok=True)

    metriclist = {"L1": [], "L3": []}

    for desc in ["L1", "L3"]:
        for(_, line) in df.iterrows():
            
            graypath = None
            try:
                graypath = line["{volume_name}"]
            except:
                pass
            try:
                graypath = line["{volume_name}_x"]
            except:
                pass
            try:
                graypath = line["{volume_name}_y"]
            except:
                pass
                
            if graypath is None:
                assert 0
            
            if graypath[-7:] == '.nii.gz':
                nameroot = Path(Path(graypath).stem).stem
            else:
                nameroot = Path(graypath).stem
            
            overwrited_result_path = label_file_overwriter_path / Path(nameroot + "_"+ desc + ".result.nii.gz")
            
            print("testing " + str(overwrited_result_path))
            
            try:
                resultImage = sitk.ReadImage(str(overwrited_result_path))
            except:
                continue
            
            print("processing " + str(overwrited_result_path))
            
            grayImage = sitk.ReadImage(graypath)
            
            # write ill
            illfilepath = str((Path(output_path) / Path(nameroot + "_" + desc + ".ill.png")).resolve())
            makeill.write_sag_fusion_res_ill(resultImage, None, grayImage, None, None, None, illfilepath)
            
            # calc
            (tri_volume, tri_png, metrics) = sarcocalculate.calculate(grayImage, resultImage)
            
            # write tricolor
            trifilepath = str((Path(output_path) / Path(nameroot + "_" + desc + ".tricolor.png")).resolve())
            imageio.imwrite(trifilepath, tri_png)
            
            # add to dict
            print(metrics)
            metrics = {(desc+"_"+k):v for (k,v) in metrics.items()}
            metrics['id'] = nameroot
            metriclist[desc].append(metrics)
            
    csvfilepath = str(Path(output_path) / Path("modified.csv"))
    
    dfL3 = pd.DataFrame.from_records(metriclist["L3"]).set_index('id')    
    dfL3.to_csv(csvfilepath+"_L3.csv")
    
    dfL1 = pd.DataFrame.from_records(metriclist["L1"]).set_index('id')
    dfL1.to_csv(csvfilepath+"_L1.csv")

#    pd.DataFrame.merge(dfL1, dfL3, on='id').to_csv(csvfilepath)
    dfL1.join(dfL3, on='id', how='outer').to_csv(csvfilepath)

