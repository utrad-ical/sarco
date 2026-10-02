


import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import SimpleITK as sitk
import scipy.ndimage

import torch
import torch.nn
import torch.optim as optim

import torchvision.models.densenet
import torchvision.models.resnet

import torchvision.transforms

import readfiles
import prepcache
import volumeresampler




transform = torchvision.transforms.Compose(
    [torchvision.transforms.RandomErasing(p=0.5),
     torchvision.transforms.RandomAffine(degrees=5.0)]
)





DEFAULT_ORIGINAL_SLICE_THICKNESS = 5            # mm
INPUT_VOXEL_SIZE=1      # mm
INPUT_X_SIZE = 512      # voxels
INPUT_Y_SIZE = 512      # voxels



GRAY_CLIP = (100, 1000)



def resampled_geometry(grayImage):
    '''1 mm in plane, the original slices along z'''
    new_voxel_size = [INPUT_VOXEL_SIZE, INPUT_VOXEL_SIZE, grayImage.GetSpacing()[2]]
    new_matrix_size = [INPUT_X_SIZE, INPUT_X_SIZE, grayImage.GetSize()[2]]
    return new_voxel_size, new_matrix_size



def label_presence(grayImage, labelImage, labelused) -> dict:
    '''the true slice, and whether each resampled slice carries the label'''

    lorig = sitk.GetArrayFromImage(labelImage)
    ztrue = np.argmax(np.sum(np.sum(lorig == labelused, axis=2), axis=1))

    new_voxel_size, new_matrix_size = resampled_geometry(grayImage)
    # nearest neighbour: linear interpolation of a multi-label image creates
    # in-between values at label borders (between 0 and 2 it passes through
    # 1), so another structure was flagged as the target label
    lR = volumeresampler.resample(labelImage, 0, new_voxel_size, new_matrix_size, is_label = True)
    lRarr = (sitk.GetArrayFromImage(lR) == labelused).astype(float)
    presence = np.max(np.max(lRarr, axis=2), axis=1).astype('uint8')

    return {"ztrue": ztrue, "presence": presence}



def gray_window(grayImage, startindexz, endindexz):
    '''slices [startindexz, endindexz) resampled to 1 mm in plane and clipped'''

    new_voxel_size, new_matrix_size = resampled_geometry(grayImage)
    gR = volumeresampler.resample(grayImage, 0, new_voxel_size, new_matrix_size)

    gRarr = sitk.GetArrayFromImage(gR)[startindexz : endindexz, :, :]

    # modify grayscale
    gRarr[gRarr<GRAY_CLIP[0]] = GRAY_CLIP[0]
    gRarr[GRAY_CLIP[1]<gRarr] = GRAY_CLIP[1]

    return gRarr



def data_preparation(volume_list : str, labelused : int, istest : bool, inputzfile = None, cache_dir = None):

    i = 0
    
    training_gray = np.empty([0,3,INPUT_Y_SIZE,INPUT_X_SIZE], dtype='int64')
    training_label = np.empty([0], dtype='int32')
    
    training_gray_list = []
    training_label_list = []
    
    zs = []
    ztrues = []
    basezs = []
    searchalls = []
    detectedzs = []
    case_indices = []
    
    if(inputzfile is not None and istest):
        if 0: # obsolete
            for line in open(inputzfile, 'rt'):
                _, idxcase, _, detectedz, _, _realz, _, likelihood,_,d0,d1,d2,d3,d4,d5,d6,d7 = line.strip().split(',')
                idxcase = int(idxcase.strip())
                detectedz = int(detectedz.strip())
                likelihood = float(likelihood.strip())
                if likelihood < 0.00001:
                    searchall = True
                else:
                    searchall = False
                searchalls.append(searchall)
                detectedzs.append(detectedz)
        else:
            df = pd.read_csv(inputzfile, encoding='shift_jis')
            detectedzs = np.array(df['{z_estimated}'])
            likelihoods = np.array(df['{likelihood}'])
            detectedzs[likelihoods < 0.00001] = -1 # not detected
            searchalls = [False, ] * len(detectedzs)
    
    i = 0
    for (_, line) in readfiles.read_list(volume_list).iterrows():

        gray_path, label_path = readfiles.paths(line)

        # the volumes are read only when something has to be computed
        images = []
        def read():
            if not images:
                images.extend(readfiles.readone(line))
            return images

        # geometry from the header alone
        info = sitk.ImageFileReader()
        info.SetFileName(gray_path)
        info.ReadImageInformation()
        zsize = info.GetSize()[2]
        zspacing = info.GetSpacing()[2]

        # true label, and which resampled slices carry it
        if label_path is None:
            # no label file: an empty label
            label = {"ztrue": 0, "presence": np.zeros(zsize, dtype='uint8')}
        else:
            label = prepcache.cached(cache_dir, "modifier_label",
                                     {"label": prepcache.file_signature(label_path),
                                      "gray_geometry": [list(info.GetSize()), list(info.GetSpacing())],
                                      "labelused": labelused},
                                     lambda: label_presence(read()[0], read()[1], labelused))
        ztrue = int(label["ztrue"])

        ztrues.append(ztrue)
        
        
        
        # if test, only partial volume is used
        if istest and (detectedzs[i] < 0 or zsize<=detectedzs[i]):
            # out of view
            basez = -100000
            depthz= 0
            pass
        else:
            if istest:
                basez = detectedzs[i] * zspacing - 20
                depthz = int(40 / zspacing)
            else:
                basez = ztrue * zspacing - 40
                depthz = int(80 / zspacing)
        
            basez = int(np.floor(basez/zspacing)) * zspacing
        
        depthz = int(depthz+0.0001)

        # slice range of the search window, clamped inside the volume. near the
        # top basez is negative, and a negative start index counts from the
        # END of the array in python, so the window came out empty: the case
        # was silently dropped in training and reported as not detected in test.
        startindexz = int(np.round(basez/zspacing))
        endindexz = startindexz + depthz
        startindexz = int(np.clip(startindexz, 0, zsize))
        endindexz = int(np.clip(endindexz, startindexz, zsize))

        # the slice index the window starts at, to map the argmax back
        basezs.append(startindexz)

        l = label["presence"][startindexz:endindexz].astype('int64')

        print(l)

        if l.shape[0] == 0:
            # l is empty
            pass
        else:
            window = prepcache.cached(cache_dir, "modifier_gray",
                                      {"gray": prepcache.file_signature(gray_path),
                                       "start": startindexz, "end": endindexz,
                                       "clip": GRAY_CLIP},
                                      lambda: {"window": gray_window(read()[0], startindexz, endindexz)})
            g = np.repeat(window["window"][:, np.newaxis, :, :], 3, axis=1)

            #training_gray = np.concatenate((training_gray, g), axis=0)
            #training_label = np.concatenate((training_label, l), axis=0)
            training_gray_list.append(g)
            training_label_list.append(l)
            
            case_indices += [i,] * l.shape[0] # for recover of indices

        i+=1

    if not training_gray_list:
        # every case out of view (test), e.g. L3 below an upper-abdominal scan
        return None, None, zs, ztrues, basezs, case_indices

    training_gray = np.concatenate(training_gray_list, axis=0)
    training_label = np.concatenate(training_label_list, axis=0)
    
    print(f"{training_gray.shape[0]=}")
    print(f"{training_label.shape[0]=}")
    print(f"{len(case_indices)=}")

    return training_gray, training_label, zs, ztrues, basezs, case_indices




def main(mode : str, 
         volume_list : str, 
         batch_size, 
         device : str, 
         learning_rate : float,
         beta_1 : float, 
         model_file_name : str,
         input_file_name : str,
         output_file_name : str,
         n_epoch : int,
         labelused : int,
         cache_dir : str = None):
    
    # build a model
    model = torchvision.models.densenet121(num_classes = 2)
    #model = torchvision.models.resnet50(num_classes = 2)
    
    
    
    if(mode == 'test'):
        model.load_state_dict(torch.load(model_file_name))
    
    
    
    # apply data parallel
    model = torch.nn.DataParallel(model)

    model = model.to(device)



    if(mode == 'train'):
        
        volume_list = readfiles.drop_empty_labels(volume_list, labelused, cache_dir)

        # data preparation
        (training_gray, training_label, zs, ztrues, _, _) = data_preparation(volume_list, 
                                                                       labelused, False,
                                                                       cache_dir=cache_dir)
        
        
        
        # data loader
        
        tensor_x = torch.from_numpy(training_gray)
        tensor_y = torch.from_numpy(training_label)
        
        training_dataset = torch.utils.data.TensorDataset(tensor_x.type(torch.FloatTensor),
                                                          tensor_y.type(torch.LongTensor))
        training_loader = torch.utils.data.DataLoader(training_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True)



        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta_1, 0.999))



        weight = torch.tensor((0.05,0.95)).to(device)



        for epoch in range(n_epoch):
        
            model.train()
            
            sum0 = 0
            sum1 = 0
            for batch_idx, (data, labels) in enumerate(training_loader):
            
                optimizer.zero_grad()
            
                data, labels = data.to(device), labels.to(device)
                
                data = transform(data)
                
                outputs = model(data)
                
                #outputs = torch.nn.LogSoftmax()(outputs)
                
                loss = torch.nn.CrossEntropyLoss(weight)(outputs, labels)
                
                loss = torch.mean(loss)
                
                loss.backward()
                optimizer.step()
                
                sum0 += data.shape[0]
                sum1 += loss.detach().cpu().numpy()
            
            print("%f" % (sum1/sum0))
            torch.save(model.module.state_dict(), model_file_name)




    elif (mode == 'test'):
    
        test_gray, test_label, zs, ztrues, basezs, case_indices = data_preparation(volume_list, 
                                                                                   labelused, 
                                                                                   True, 
                                                                                   input_file_name,
                                                                                   cache_dir=cache_dir)

        if test_gray is None:
            # nothing to refine: report every case as out of view (z = -1)
            df = pd.read_csv(input_file_name, encoding='shift_jis')
            df["{z_modified}"] = -1
            df["{likelihoods_modified}"] = -1000.0
            print(df)
            df.iloc[:,0:].to_csv(output_file_name, encoding='shift-jis')
            return

        num_cases = len(ztrues)
        print("case num = %d" % num_cases)
        
        tensor_x = torch.from_numpy(test_gray)
        tensor_y = torch.from_numpy(test_label)
        
        test_dataset = torch.utils.data.TensorDataset(tensor_x.type(torch.FloatTensor),
                                                      tensor_y.type(torch.FloatTensor))
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        base_index = 0
        model.eval()
        likelihoods = np.zeros_like(test_label).astype('float32')
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_loader):

                batchsz = data.size()[0]

                data, labels = data.to(device), labels.to(device)
                
                res = model(data)
                
                sm = torch.nn.Softmax(dim=1)(res)
                
                l = sm.detach().cpu().numpy()[:,1]
                likelihoods[base_index:base_index+batchsz] = l

                base_index += batchsz
                
        # number of slices of each volume, for the edge rule below
        zsizes = []
        for (_, line) in readfiles.read_list(volume_list).iterrows():
            info = sitk.ImageFileReader()
            info.SetFileName(readfiles.paths(line)[0])
            info.ReadImageInformation()
            zsizes.append(info.GetSize()[2])

        # reconstruction
        zestimates = np.ndarray((num_cases))
        case_indices = np.array(case_indices)
        likelihoods = np.array(likelihoods)
        likelihoods_cases = np.ndarray((num_cases))
        
        for ind_case in range(num_cases):
            # print(ind_case)
            # print(test_label[case_indices == ind_case])
            # print(likelihoods[case_indices == ind_case])
            try:
                zestimates[ind_case] = np.argmax(likelihoods[case_indices == ind_case]) + basezs[ind_case]
                likelihoods_cases[ind_case] = np.max(likelihoods[case_indices == ind_case])
            except Exception as e:
                zestimates[ind_case] = -1
                likelihoods_cases[ind_case] = -1000.0
                print(e)
            # a level found on the first or last slice is taken as out of view:
            # when the vertebra lies beyond the scan, the search window is
            # clamped to the edge and the argmax lands there
            if zestimates[ind_case] >= 0 and zsizes[ind_case] > 2 and \
                    zestimates[ind_case] in (0, zsizes[ind_case] - 1):
                print(f"case #{ind_case}: z={int(zestimates[ind_case])} is on the edge of the "
                      f"volume; treated as out of view")
                zestimates[ind_case] = -1
                likelihoods_cases[ind_case] = -1000.0
        
        if 0: # obsolete
            with open(output_file_name, "wt") as f:
                for ind_case in range(num_cases):
                    f.write("case, %d, z, %d, realz, %d, likelihood, %f\n" % 
                        (ind_case, zestimates[ind_case], ztrues[ind_case], likelihoods_cases[ind_case]))
        else:
            df = pd.read_csv(input_file_name, encoding='shift_jis')
            df["{z_modified}"] = zestimates
            df["{likelihoods_modified}"] = likelihoods_cases
            print(df)
            df.iloc[:,0:].to_csv(output_file_name, encoding='shift-jis')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Lumbar detection modifier',
        add_help=True)

    parser.add_argument('--mode', help="train or test", 
                        type=str, choices=['train', 'test'],  default='train')
    parser.add_argument('--volume_list',
                        help='File path list of training/test image data',
                        type=str, required=True)
    parser.add_argument('--batch_size',
                        help='batch_size',
                        type=int, default=24)
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        type=float, default=0.001)
    parser.add_argument('--beta_1',
                        help='beta_1',
                        type=float, default=0.99)
    parser.add_argument('--device',
                        help='device',
                        type=str, default="cuda")
    parser.add_argument('--model_file_name',
                        help='model_file_name',
                        type=str, default='model/modifier_model.ckpt')
    parser.add_argument('--input_file_name',
                        help='input_file_name',
                        default='temp/l3.txt')
    parser.add_argument('--output_file_name',
                        help='output_file_name',
                        type=str, default='temp/l3mod.txt')
    parser.add_argument('--n_epoch',
                        help='total epoch num',
                        type=int, default=100)
    parser.add_argument('--labelused',
                        help='label index to be used (default=2)',
                        type=int, default=2)
    parser.add_argument('--cache_dir',
                        help='directory to cache per-case preprocessing in (default: no cache)',
                        type=str, default=None)


    args = parser.parse_args()

    main(**args.__dict__)

