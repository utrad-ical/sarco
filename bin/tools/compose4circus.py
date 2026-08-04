


import sys
import argparse

import os

from typing import Any, Optional, Sequence, Union

import numpy as np
import scipy.ndimage

import SimpleITK as sitk

import json

import readfiles

import png2label

import nakaocircusutil.voxel_label
import nakaocircusutil.circus_api





def circus_replace_labelset(caseid : str,
                            labelset : Sequence, 
                            nameset : Sequence[str]
                            ):
    voxel_label_set = []
    labels_json = []
    for i, (labelimage, labelname) in enumerate(zip(labelset, nameset)):
    
        # kill null volume
        labelarray = sitk.GetArrayFromImage(labelimage)
        if np.sum(labelarray) == 0:
            continue
        
        # build CircusVoxelLabel
        cvl = nakaocircusutil.voxel_label.CircusVoxelLabel.from_array(labelarray)
        voxel_label_set.append(cvl)
        # upload blob
        nakaocircusutil.circus_api.blob_post(cvl.packed)
        
        # build "labels" json
        label_json = {}
        label_json["name"] = labelname
        label_json["type"] = "voxel"
        label_json["data"] = cvl.get_meta_dict()
        label_json["data"]["alpha"] = .5
        label_json["data"]["color"] = \
            [
            "#ff0000", "#00ff00", "#7f00ff", "#007fff", "#0000ff", "#7f007f",
            "#3f7f7f", "#7f3f7f", "#7f7f3f", "#1f3f7f", "#7f3f1f", "#3f7f1f"
            ][i]
        label_json["attributes"] = {}
        labels_json.append(label_json)
    
    # get and addrev
    case = nakaocircusutil.circus_api.get_case(caseid)
    revision = case["revisions"][-1]
    revision["series"][0]["labels"] = labels_json
    revision["description"] = "automatic update"
    print(json.dumps(revision))
    nakaocircusutil.circus_api.case_addrev_dict(caseid, revision, ["--force", "-D"])



def main(volume_list : str,
         workdir_l1 : str,
         workdir_l3 : str,
         outputdir : str):


    try:
        os.mkdir(outputdir)
    except FileExistsError as e:
        print(e)
        pass
    
    
    
    # case id list
    caseidlist = []
    for line in open(volume_list, 'rt'):
        caseidlist.append( (line.split(','))[2].strip() )
    
    
    
    # z list
    zlist_l1 = []
    for line in open(os.path.join(workdir_l1, "l1mod.txt"), 'rt'):
        zlist_l1.append( int( line.split(',')[3].strip() ) )
    
    zlist_l3 = []
    for line in open(os.path.join(workdir_l3, "l3mod.txt"), 'rt'):
        zlist_l3.append( int( line.split(',')[3].strip() ) )
    
    
    
    # main
    
    for i, (grayVolume, manualLabelVolume) in enumerate(readfiles.reader(volume_list)):
        
        caseid = caseidlist[i]
        
        # compile all
        manualLabelArray = sitk.GetArrayFromImage(manualLabelVolume)

        l1_manualArray = (manualLabelArray == 1).astype('uint8')
        l1_manualImage = sitk.GetImageFromArray(l1_manualArray)
        sitk.WriteImage(l1_manualImage, os.path.join(outputdir, "l1_manual%d.nii.gz"%i))

        l3_manualArray = (manualLabelArray == 2).astype('uint8')
        l3_manualImage = sitk.GetImageFromArray(l3_manualArray)
        sitk.WriteImage(l1_manualImage, os.path.join(outputdir, "l3_manual%d.nii.gz"%i))

        l1_autoImage = png2label.png2array(os.path.join(workdir_l1, "muscle%d.png"%i),
                  zlist_l1[i],
                  grayVolume
                 )
        sitk.WriteImage(l1_autoImage, os.path.join(outputdir, "l1_auto%d.nii.gz"%i))
        l3_autoImage = png2label.png2array(os.path.join(workdir_l3, "muscle%d.png"%i),
                  zlist_l3[i],
                  grayVolume
                 )
        sitk.WriteImage(l3_autoImage, os.path.join(outputdir, "l3_auto%d.nii.gz"%i))
        
        # write to CIRCUS
        circus_replace_labelset(caseid, 
            [ l1_manualImage ,  l3_manualImage ,  l1_autoImage ,  l3_autoImage ],
            ["l1_manualImage", "l3_manualImage", "l1_autoImage", "l3_autoImage"],
            )
        
        print("case %d (%s) done"%(i,caseid))










if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='composer for upload of results to CIRCUS',
        add_help=True)
        
    parser.add_argument('--volume_list', type=str, required=True)
    
    parser.add_argument('--workdir_l1', type=str, default='temp_l1')
    
    parser.add_argument('--workdir_l3', type=str, default='temp_l3')
    
    parser.add_argument('--outputdir', type=str, default='4circus')
    
    args = parser.parse_args()
    
    main(**args.__dict__)

