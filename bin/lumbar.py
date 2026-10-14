


import sys
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import SimpleITK as sitk
import scipy.ndimage

import torch
import torch.nn
import torch.optim as optim
import unet


import readfiles
import prepcache
import volumeresampler
import extract_trunk




DEFAULT_ORIGINAL_SLICE_THICKNESS = 5.0            # mm
INPUT_VOXEL_SIZE=1      # mm, input for Unet
INPUT_SLICE_THICKNESS=1.666666666667    # mm
INPUT_X_SIZE = 512      # voxels
INPUT_Z_SIZE = 512      # voxels
NUM_FEATURE_IMAGE = 4
NUM_LABEL_IMAGE = 2



def dice_loss(outputs, labels, eps = 1.):
    n_batch = outputs.size()[0]

    pred = outputs.view([n_batch, -1])
    target = labels.view([n_batch, -1])

    intersection = torch.sum(pred*target, axis=1)
    loss = 1. - (2. * intersection) / (torch.sum(pred, axis=1) + torch.sum(target, axis=1) + eps)

    return torch.sum(loss)



# the elongated volume: 250 mm of margin above the first original slice, then
# 2 x INPUT_Z_SIZE rows. the margin is a fixed physical length, the same one
# zshift removes below, so that crop row 0 is always the first original slice
# whatever the slice thickness. (it used to be 50 original slices, which equals
# 250 mm only for 5 mm slices and shifted thinner volumes.)
Z_MARGIN_MM = 50*DEFAULT_ORIGINAL_SLICE_THICKNESS
ELONGATED_VOXEL_SIZE = [INPUT_VOXEL_SIZE, INPUT_VOXEL_SIZE, INPUT_SLICE_THICKNESS]
ELONGATED_MATRIX_SIZE = [INPUT_X_SIZE, INPUT_X_SIZE, INPUT_Z_SIZE*2]
GRAY_CLIP = (100, 1000)



def gray_projections(grayImage) -> dict:
    '''
    the 4 feature images of the WHOLE elongated volume, (4, 2*INPUT_Z_SIZE, X).

    every training/test sample is a window of INPUT_Z_SIZE rows of these:
    clipping is per voxel and the projections run along x or y, so projecting
    the whole volume and then taking rows gives exactly what cropping first and
    projecting the crop gave. one resampling and one arm removal per case thus
    serve every augmented copy, and the result can be cached.
    '''
    print("resampling...", end="", flush=True)
    resampled = volumeresampler.resample(grayImage, -Z_MARGIN_MM, ELONGATED_VOXEL_SIZE, ELONGATED_MATRIX_SIZE, is_label = False, apply_filter = True)

    print("arm killing...", end="", flush=True)
    resampled = extract_trunk.kill_arms(resampled)

    gRarr = sitk.GetArrayFromImage(resampled)

    # modify grayscale
    gRarr[gRarr<GRAY_CLIP[0]] = GRAY_CLIP[0]
    gRarr[GRAY_CLIP[1]<gRarr] = GRAY_CLIP[1]

    # for violation...
    while len(gRarr.shape) >= 4:
        gRarr = np.mean(gRarr, axis=3)

    # projection along x- and y-axes
    g = np.zeros((NUM_FEATURE_IMAGE, gRarr.shape[0], INPUT_X_SIZE), dtype='float32')
    g[0,:,:] = np.mean(gRarr, axis=2)
    g[1,:,:] = np.mean(gRarr, axis=1)
    g[2,:,:] = np.max(gRarr, axis=2)
    g[3,:,:] = np.max(gRarr, axis=1)

    return {"projections": g, "slice_thickness": grayImage.GetSpacing()[2]}



def label_projections(labelImage, labelused) -> dict:
    '''the 2 label images of the whole elongated volume, and the true slice'''

    # true label
    lorig = sitk.GetArrayFromImage(labelImage)

    sumlabel = np.sum(np.sum(lorig == labelused, axis=2), axis=1)
    if np.sum(sumlabel) == 0:
        # no answer
        ztrue = -1
    else:
        ztrue = np.argmax(sumlabel)

    resampled = volumeresampler.resample(labelImage, -Z_MARGIN_MM, ELONGATED_VOXEL_SIZE, ELONGATED_MATRIX_SIZE, is_label = True)
    lRarr = (sitk.GetArrayFromImage(resampled) == labelused)

    l = np.zeros((NUM_LABEL_IMAGE, lRarr.shape[0], INPUT_X_SIZE), dtype='uint8')
    l[0,:,:] = np.max(lRarr, axis=2)
    l[1,:,:] = np.max(lRarr, axis=1)

    return {"projections": l, "ztrue": ztrue, "slice_thickness": labelImage.GetSpacing()[2]}



def case_projections(line, labelused, cache_dir = None):
    '''
    (gray projections, label projections, ztrue, original slice thickness) of
    one row of the volume list, from the cache when possible. the volumes are
    read only when something has to be computed.
    '''
    gray_path, label_path = readfiles.paths(line)

    images = []
    def read():
        if not images:
            images.extend(readfiles.readone(line))
        return images

    gray = prepcache.cached(cache_dir, "lumbar_gray",
                            {"gray": prepcache.file_signature(gray_path),
                             **readfiles.flip_key(line),
                             "z_margin_mm": Z_MARGIN_MM,
                             "voxel_size": ELONGATED_VOXEL_SIZE,
                             "matrix_size": ELONGATED_MATRIX_SIZE,
                             "clip": GRAY_CLIP},
                            lambda: gray_projections(read()[0]))

    if label_path is None:
        # no label file: an empty label on the gray volume's grid
        label = {"projections": np.zeros((NUM_LABEL_IMAGE,) + gray["projections"].shape[1:], dtype='uint8'),
                 "ztrue": -1,
                 "slice_thickness": gray["slice_thickness"]}
    else:
        label = prepcache.cached(cache_dir, "lumbar_label",
                                 {"label": prepcache.file_signature(label_path),
                                  **readfiles.flip_key(line),
                                  "labelused": labelused,
                                  "z_margin_mm": Z_MARGIN_MM,
                                  "voxel_size": ELONGATED_VOXEL_SIZE,
                                  "matrix_size": ELONGATED_MATRIX_SIZE},
                                 lambda: label_projections(read()[1], labelused))

    return (gray["projections"], label["projections"],
            int(label["ztrue"]), float(label["slice_thickness"]))



def data_preparation(volume_list : str, labelused, data_augmentation_multiplier, cache_dir = None):

    i = 0
    zs = []
    ztrues = []
    original_slice_thicknesses = []

    # malloc
    df = readfiles.read_list(volume_list)
    N = len(df)
    training_gray = np.zeros([N*data_augmentation_multiplier,NUM_FEATURE_IMAGE,INPUT_Z_SIZE,INPUT_X_SIZE], dtype='float32')
    training_label = np.zeros([N*data_augmentation_multiplier,2,INPUT_Z_SIZE,INPUT_X_SIZE], dtype='float32')
    
    # main loop
    for n, (_, line) in enumerate(df.iterrows()):

        print(f"sample #{n}/{N}: ", end="", flush=True)

        gproj, lproj, ztrue, original_slice_thickness = case_projections(line, labelused, cache_dir)
        original_slice_thicknesses.append(original_slice_thickness)

        print("data augmenting...", end="", flush=True)

        for index in range(data_augmentation_multiplier):
        
            zbase = np.random.randint(-50, 50) * DEFAULT_ORIGINAL_SLICE_THICKNESS
            if index==0:
                zbase = 0 # use original
            # shift the answer along z-axis. computed from the original ztrue
            # each time: overwriting ztrue made every augmented copy carry the
            # sum of all earlier shifts.
            if ztrue >= 0:
                ztrue_shifted = ztrue - int(np.round(zbase / original_slice_thickness))
            else:
                ztrue_shifted = -1
            ztrues.append(ztrue_shifted)
        
            # shift the image along z-axis
            # round, not floor: 250 / 1.666666666667 is 149.99999999994 in
            # floating point, and floor dropped one row (1.67 mm) every time
            zshift = int(np.round((zbase + Z_MARGIN_MM) / INPUT_SLICE_THICKNESS))

            # a window of the whole-volume projections (see gray_projections)
            training_gray[i,:,:,:] = gproj[:, zshift:zshift+INPUT_Z_SIZE, :]
            training_label[i,:,:,:] = lproj[:, zshift:zshift+INPUT_Z_SIZE, :]

            i+=1
    
        print("done.")

    return training_gray, training_label, zs, ztrues, original_slice_thicknesses




def load_presence_threshold(model_file_name):
    """threshold on the peak likelihood written by --online training, or None"""
    path = Path(str(model_file_name) + ".json")
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["presence_threshold"])



SCAN_ROW0 = int(round(Z_MARGIN_MM / INPUT_SLICE_THICKNESS))   # elongated row of slice 0
BLANK = float(GRAY_CLIP[0])          # what rows outside the scan read as after clipping



class LumbarWindows(torch.utils.data.Dataset):
    """
    INPUT_Z_SIZE-row windows of the cached whole-volume projections, drawn
    afresh every time, with the scan coverage varied as it is in practice:

    * with probability truncate_prob, the scan is cut to a random craniocaudal
      range (min_scan_mm up to the whole scan) -- an upper-abdominal dynamic
      study, say. The level may then lie outside it; its label is then empty
      and the network has to answer "nowhere" (hence the BCE term).
    * the window starts at the (cut) first slice, as at test time, or up to
      max_lead_mm above it (rows of nothing).
    * a random rectangle is erased (as torchvision's RandomErasing, p = 0.5).

    Cutting rows of the projections is exact: they run along x and y, and
    rows outside a scan read as the clip floor anyway.
    """

    def __init__(self, gprojs, lprojs, scan_rows, truncate_prob=0.6,
                 min_scan_mm=120.0, max_lead_mm=150.0, erase_prob=0.5, augment=True):
        self.g, self.l, self.scan = gprojs, lprojs, scan_rows
        self.truncate_prob, self.erase_prob, self.augment = truncate_prob, erase_prob, augment
        self.min_rows = int(round(min_scan_mm / INPUT_SLICE_THICKNESS))
        self.max_lead = int(round(max_lead_mm / INPUT_SLICE_THICKNESS))

    def __len__(self):
        return len(self.g)

    def window(self, i, a, b, s):
        """rows [s, s + INPUT_Z_SIZE) with only the scan rows [a, b) kept"""
        g, l = self.g[i], self.l[i]
        x = np.full((g.shape[0], INPUT_Z_SIZE, g.shape[2]), BLANK, np.float32)
        y = np.zeros((l.shape[0], INPUT_Z_SIZE, l.shape[2]), np.float32)
        lo, hi = max(a, s), min(b, s + INPUT_Z_SIZE, g.shape[1])
        if hi > lo:
            x[:, lo - s:hi - s] = g[:, lo:hi]
            y[:, lo - s:hi - s] = l[:, lo:hi]
        return x, y

    def __getitem__(self, i):
        rng = np.random.default_rng()
        a, b = self.scan[i]
        if self.augment and rng.random() < self.truncate_prob and b - a > self.min_rows:
            length = int(rng.integers(self.min_rows, b - a + 1))
            a = int(rng.integers(a, b - length + 1))
            b = a + length
        s = a
        if self.augment and rng.random() < 0.5:
            s = a - int(rng.integers(0, self.max_lead + 1))
        x, y = self.window(i, a, b, s)
        if self.augment and rng.random() < self.erase_prob:
            area = x.shape[1] * x.shape[2] * rng.uniform(0.02, 0.33)
            ratio = np.exp(rng.uniform(np.log(0.3), np.log(3.3)))
            h = int(min(x.shape[1], round(np.sqrt(area * ratio))))
            w = int(min(x.shape[2], round(np.sqrt(area / ratio))))
            r0 = int(rng.integers(0, x.shape[1] - h + 1))
            c0 = int(rng.integers(0, x.shape[2] - w + 1))
            x[:, r0:r0 + h, c0:c0 + w] = 0.0
        return torch.from_numpy(x), torch.from_numpy(y)



def load_online(volume_list, labelused, cache_dir):
    """cached projections and the scan's row range in the elongated grid, per case"""
    df = readfiles.read_list(volume_list)
    gprojs, lprojs, scans = [], [], []
    for n, (_, line) in enumerate(df.iterrows()):
        g, l, ztrue, thick = case_projections(line, labelused, cache_dir)
        info = sitk.ImageFileReader()
        info.SetFileName(readfiles.paths(line)[0])
        info.ReadImageInformation()
        nz = info.GetSize()[2]
        gprojs.append(g.astype(np.float32))
        lprojs.append(l.astype(np.uint8))
        scans.append((SCAN_ROW0, min(g.shape[1], SCAN_ROW0 + int(round(nz * thick / INPUT_SLICE_THICKNESS)))))
        if (n + 1) % 100 == 0:
            print(f"loaded {n + 1}/{len(df)}", flush=True)
    return gprojs, lprojs, scans



def peak_rows(model, x, device):
    """(likelihood peak, its row) of each window, as the test path computes them"""
    with torch.no_grad():
        res = torch.sigmoid(model(x.to(device))).cpu().numpy()
    raw = res[:, 0:2].mean(axis=3).mean(axis=1)               # (batch, rows)
    smooth = scipy.ndimage.gaussian_filter1d(raw, sigma=1, axis=1)
    return smooth.max(axis=1), smooth.argmax(axis=1)



def evaluation_windows(scans, lprojs, seed=1):
    """for each case: the whole scan, a cut keeping the level, a cut without it"""
    rng = np.random.default_rng(seed)
    out = []
    for i, ((a, b), l) in enumerate(zip(scans, lprojs)):
        rows = np.flatnonzero(l.max(axis=(0, 2)) > 0)
        if len(rows) == 0:
            continue
        r = int(rows.mean())
        out.append((i, a, b, a, r, True, "whole"))
        L = int(round(200 / INPUT_SLICE_THICKNESS))                 # a 200 mm scan
        # the level at least 30 rows (50 mm) inside the cut, the cut inside the scan
        first, last = max(a, r - L + 30), min(r - 30, b - L)
        if first <= last:
            lo = int(rng.integers(first, last + 1))
            out.append((i, lo, lo + L, lo, r, True, "cut, level in"))
        if r - 30 - L >= a:                                          # ends above the level
            out.append((i, r - 30 - L, r - 30, r - 30 - L, r, False, "cut above"))
        if r + 30 + L <= b:                                          # starts below it
            out.append((i, r + 30, r + 30 + L, r + 30, r, False, "cut below"))
    return out



def evaluate_online(model, ds, device, threshold=None):
    """peak likelihood and z error (mm) on the evaluation windows of ds"""
    rows = []
    for (i, a, b, s, r, present, kind) in evaluation_windows(ds.scan, ds.l):
        x, _ = ds.window(i, a, b, s)
        peak, row = peak_rows(model, torch.from_numpy(x)[None], device)
        rows.append(dict(case=i, kind=kind, present=present, peak=float(peak[0]),
                         err_mm=float(abs(row[0] + s - r) * INPUT_SLICE_THICKNESS)))
    df = pd.DataFrame(rows)
    if threshold is None:
        # the threshold that best separates present from absent levels
        cand = np.unique(df.peak)
        acc = [((df.peak >= t) == df.present).mean() for t in cand]
        threshold = float(cand[int(np.argmax(acc))])
    df["called_present"] = df.peak >= threshold
    return df, threshold



def train_online(volume_list, val_list, labelused, model_file_name, batch_size, device,
                 learning_rate, n_epoch, bce_weight, truncate_prob, init_model, cache_dir,
                 num_workers=8):
    model = unet.Unet(NUM_FEATURE_IMAGE, NUM_LABEL_IMAGE, 16)
    if init_model:
        model.load_state_dict(torch.load(init_model, map_location="cpu"))
    model = torch.nn.DataParallel(model).to(device)

    volume_list = readfiles.drop_empty_labels(volume_list, labelused, cache_dir)
    train_ds = LumbarWindows(*load_online(volume_list, labelused, cache_dir), truncate_prob=truncate_prob)
    val_ds = LumbarWindows(*load_online(val_list, labelused, cache_dir), augment=False)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                         drop_last=True, num_workers=num_workers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch)
    bce = torch.nn.BCEWithLogitsLoss()
    best, best_state = None, None
    for epoch in range(n_epoch):
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.mean(dice_loss(torch.sigmoid(logits), y)) / x.shape[0] + bce_weight * bce(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * x.shape[0]
            n += x.shape[0]
        sched.step()
        if epoch % 10 == 9 or epoch == n_epoch - 1:
            model.eval()
            df, thr = evaluate_online(model, val_ds, device)
            p = df[df.present]
            score = ((df.called_present == df.present).mean()
                     + (p.called_present & (p.err_mm <= 10)).mean())
            print(f"epoch {epoch:4d}  loss {total / n:.4f}  val: presence acc "
                  f"{(df.called_present == df.present).mean():.4f} (threshold {thr:.3f}), "
                  f"level within 10 mm {(p.err_mm <= 10).mean():.4f}, "
                  f"median error {p.err_mm.median():.1f} mm", flush=True)
            if best is None or score >= best[0]:
                best = (score, thr, epoch)
                best_state = {k: v.detach().cpu().clone() for k, v in model.module.state_dict().items()}
        else:
            print(f"epoch {epoch:4d}  loss {total / n:.4f}", flush=True)
    torch.save(best_state, model_file_name)
    Path(str(model_file_name) + ".json").write_text(json.dumps(
        {"presence_threshold": best[1], "epoch": best[2], "truncate_prob": truncate_prob,
         "bce_weight": bce_weight}, indent=1))
    print(f"wrote {model_file_name} (epoch {best[2]}, presence threshold {best[1]:.3f})")



def main(mode : str, 
         volume_list : str, 
         batch_size, 
         device : str, 
         learning_rate : float,
         beta_1 : float, 
         model_file_name : str,
         output_file_name : str,
         n_epoch : int,
         labelused : int,
         data_augmentation_multiplier : int,
         cache_dir : str = None):
    
    # build a model
    model = unet.Unet(NUM_FEATURE_IMAGE,NUM_LABEL_IMAGE,16)
    
    
    
    if(mode == 'test'):
        model.load_state_dict(torch.load(model_file_name))
        presence_threshold = load_presence_threshold(model_file_name)
    
    
    
    # apply data parallel
    model = torch.nn.DataParallel(model)

    model = model.to(device)



    if(mode == 'train'):
        
        volume_list = readfiles.drop_empty_labels(volume_list, labelused, cache_dir)

        print("data preparation starting...")
        # data preparation
        (training_gray, training_label, zs, ztrues, original_slice_thicknesses) = \
                                                      data_preparation(volume_list, 
                                                                       labelused,
                                                                       data_augmentation_multiplier,
                                                                       cache_dir)
        
        
        
        # data loader
        
        tensor_x = torch.from_numpy(training_gray)
        tensor_y = torch.from_numpy(training_label)
        
        training_dataset = torch.utils.data.TensorDataset(tensor_x.type(torch.FloatTensor),
                                                          tensor_y.type(torch.FloatTensor))
        training_loader = torch.utils.data.DataLoader(training_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=True,
                                                  num_workers=0)



        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta_1, 0.999))



        # augmenter
        import torchvision.transforms     # only this (original) training path needs it
        transform = torchvision.transforms.RandomErasing(p=0.5).to(device)



        for epoch in range(n_epoch):

            print("epoch "+str(epoch))

            model.train()
            
            sum0 = 0
            sum1 = 0
            for batch_idx, (data, labels) in enumerate(training_loader):
            
                optimizer.zero_grad()
            
                data, labels = data.to(device), labels.to(device)
                
                data = transform(data)
                
                outputs = model(data)
                
                outputs = torch.sigmoid(outputs)
                
                loss = dice_loss(outputs, labels)
                loss = torch.mean(loss)
                
                loss.backward()
                optimizer.step()
                
                sum0 += data.shape[0]
                sum1 += loss.detach().cpu().numpy()
            
            print("%f" % (sum1/sum0))

        if 0:
            res0 = np.squeeze(outputs.detach().cpu().numpy()[0,0,:,:])
            plt.imsave(r'res0.png', res0)
            res1 = np.squeeze(outputs.detach().cpu().numpy()[0,1,:,:])
            plt.imsave(r'res1.png', res1)
            lab0 = np.squeeze(labels.detach().cpu().numpy()[0,0,:,:])
            plt.imsave(r'lab0.png', lab0)
            lab1 = np.squeeze(labels.detach().cpu().numpy())[0,1,:,:]
            plt.imsave(r'lab1.png', lab1)

        torch.save(model.module.state_dict(), model_file_name)



    elif (mode == 'test'):
    
        test_gray, test_label, zs, ztrues, original_slice_thicknesses = \
                                            data_preparation(volume_list, 
                                                             labelused,
                                                             data_augmentation_multiplier=1,
                                                             cache_dir=cache_dir)

        print("test_gray.shape=", test_gray.shape)
        
        tensor_x = torch.from_numpy(test_gray)
        tensor_y = torch.from_numpy(test_label)
        
        test_dataset = torch.utils.data.TensorDataset(tensor_x.type(torch.FloatTensor),
                                                      tensor_y.type(torch.FloatTensor))
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        zestimates = []
        likelihoods = []

        likelihood_matrix = np.zeros((8, test_label.shape[0]))

        base_index = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_loader):

                print("batch #%d" % batch_idx)

                data, labels = data.to(device), labels.to(device)
                
                res = model(data)
                
                res = torch.sigmoid(res)
                
                for i in range(res.size()[0]):
                    case_idx = base_index + i
                    
                    gyimage = (data[i,0:4,:,:]).detach().cpu().numpy()
                    lkimage = (res[i,0:2,:,:]).detach().cpu().numpy()
                    
                    if 0:
                        plt.imsave( r"true%d-0.png"%(base_index+i), labels[i,0,:,:].cpu().numpy())
                        plt.imsave( r"true%d-1.png"%(base_index+i), labels[i,1,:,:].cpu().numpy())
                        
                        plt.imsave( r"gray%d-0.png"%(base_index+i), gyimage[0,:,:])
                        plt.imsave( r"gray%d-1.png"%(base_index+i), gyimage[1,:,:])
                        plt.imsave( r"gray%d-2.png"%(base_index+i), gyimage[2,:,:])
                        plt.imsave( r"gray%d-3.png"%(base_index+i), gyimage[3,:,:])
                        
                        plt.imsave( r"like%d-0.png"%(base_index+i), lkimage[0,:,:])
                        plt.imsave( r"like%d-1.png"%(base_index+i), lkimage[1,:,:])
                    
                    likelihoods_raw = np.mean(np.mean(lkimage, axis=2), axis=0)
                    
                    # find one peak...
                    likelihoods_g = scipy.ndimage.gaussian_filter1d(likelihoods_raw, sigma=1)
                    argmaxz = np.argmax(likelihoods_g)
                    
                    likelihoods.append(np.max(likelihoods_g))
                    
                    if 1: 
                        # with real slice thickness
                        real_original_slice_thickness = original_slice_thicknesses[case_idx]
                        argmaxz = argmaxz / real_original_slice_thickness * INPUT_SLICE_THICKNESS
                        
                        argmaxz = int(argmaxz+.5)
                        
                    # output the slice
                    print("case #%d, real z=%d, detected z=%d" % (case_idx, ztrues[case_idx], argmaxz))
                    
                    # a level outside the scan: the likelihood peak stays low
                    # (models trained with --online and a threshold file only)
                    if presence_threshold is not None and likelihoods[-1] < presence_threshold:
                        print("case #%d: peak likelihood %.3f < %.3f, out of view"
                              % (case_idx, likelihoods[-1], presence_threshold))
                        argmaxz = -1

                    zestimates.append(argmaxz)
                    
                    # append likelihooods of -4 ~ +3 z
                    for idx_diff in range(8):
                        posz = (argmaxz - 4 + idx_diff) * real_original_slice_thickness // INPUT_SLICE_THICKNESS
                        if posz<0 or INPUT_Z_SIZE<=posz:
                            continue
                        likelihood_matrix[idx_diff, case_idx] = likelihoods_raw[int(posz)]
                    

                base_index += res.size()[0]

        # file output
        if 0: # obsolete
            with open(output_file_name, "wt") as f:
                for count in range(len(zestimates)):
                    f.write("case, %d, z, %d, realz, %d, likelihood, %f, l-4tol+3, %f,%f,%f,%f,%f,%f,%f,%f \n" 
                            %  (count, zestimates[count], ztrues[count], likelihoods[count],
                                    likelihood_matrix[0,count],
                                    likelihood_matrix[1,count],
                                    likelihood_matrix[2,count],
                                    likelihood_matrix[3,count],
                                    likelihood_matrix[4,count],
                                    likelihood_matrix[5,count],
                                    likelihood_matrix[6,count],
                                    likelihood_matrix[7,count]
                                )
                            )

        else:
            df = pd.read_csv(volume_list, encoding='shift_jis')
            df["{z_estimated}"] = list(zestimates)
            df["{z_true}"] = list(ztrues)
            df["{likelihood}"] = list(likelihoods)
            print(df)
            df.iloc[:,0:].to_csv(output_file_name, encoding='shift_jis')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Lumbar detector',
        add_help=True)

    parser.add_argument('--mode', help="train or test", 
                        type=str, choices=['train', 'test'],  default='train')
    parser.add_argument('--volume_list',
                        help='File path list of training/test image data',
                        type=str, required=True)
    parser.add_argument('--batch_size',
                        help='batch_size',
                        type=int, default=50)
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
                        type=str, default='model/lumbar_model.ckpt')
    parser.add_argument('--output_file_name',
                        help='output_file_name',
                        type=str, default='temp/l3.txt')
    parser.add_argument('--n_epoch',
                        help='total epoch num',
                        type=int, default=100)
    parser.add_argument('--labelused',
                        help='label index to be used (default=2)',
                        type=int, default=2)
    parser.add_argument('--data_augmentation_multiplier',
                        help='data_augmentation_multiplier (default=10)',
                        type=int, default=10)
    parser.add_argument('--cache_dir',
                        help='directory to cache per-case preprocessing in (default: no cache)',
                        type=str, default=None)


    parser.add_argument('--online', action='store_true',
                        help='train on windows drawn afresh each epoch, with scan truncation '
                             '(see LumbarWindows); writes <model>.json with the presence threshold')
    parser.add_argument('--val_list', default=None, help='--online: validation volume list')
    parser.add_argument('--truncate_prob', type=float, default=0.6)
    parser.add_argument('--bce_weight', type=float, default=1.0)
    parser.add_argument('--init_model', default=None, help='--online: start from these weights')
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    if args.online:
        if args.mode != 'train' or args.val_list is None:
            sys.exit('--online is for --mode train and needs --val_list')
        train_online(args.volume_list, args.val_list, args.labelused, args.model_file_name,
                     args.batch_size, args.device, args.learning_rate, args.n_epoch,
                     args.bce_weight, args.truncate_prob, args.init_model, args.cache_dir,
                     args.num_workers)
    else:
        a = {k: v for k, v in args.__dict__.items()
             if k not in ('online', 'val_list', 'truncate_prob', 'bce_weight', 'init_model', 'num_workers')}
        main(**a)

