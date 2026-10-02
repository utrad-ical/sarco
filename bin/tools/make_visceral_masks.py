"""
Make visceral masks (input of the contrast-phase model) from existing results,
without rerunning the segmentation and without a GPU.

The mask is sarcocalculate.visceral_mask: the region enclosed by muscle
(label 1) minus muscle and the vertebral body (labels 2, 3), on the slice with
the most muscle, written on the label's own grid.

Input: a CSV with columns gray, label, name. For each row this writes
  <out_dir>/<name>.visceral.nii.gz
  <out_dir>/png/<name>.png        (QC: CT, muscle outline red, vertebra blue,
                                   visceral green)
and a manifest <out_dir>/manifest.csv with the QC numbers per row:
  z, muscle_area_cm2, visceral_area_cm2 and muscle_coverage, the fraction of
  the 360 one-degree directions around the muscle's centre that hit muscle
  (a broken ring, frequent at L1, shows as a low value).

  python bin/tools/make_visceral_masks.py jobs.csv out_dir --workers 32
"""

import argparse
import multiprocessing
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import scipy.ndimage
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sarcocalculate  # noqa: E402


PNG_WINDOW = (-200.0, 300.0)


def muscle_coverage(muscle : np.array) -> float:
    ys, xs = np.nonzero(muscle)
    cy = (ys.max() + ys.min()) / 2.0
    cx = (xs.max() + xs.min()) / 2.0
    degrees = np.floor(np.degrees(np.arctan2(ys - cy, xs - cx))).astype(int) % 360
    return len(np.unique(degrees)) / 360.0


def outline(mask : np.array) -> np.array:
    return mask & ~scipy.ndimage.binary_erosion(mask)


def qc_png(gray : np.array, label : np.array, visceral : np.array) -> np.array:
    lo, hi = PNG_WINDOW
    g = np.clip((gray.astype(float) - lo) / (hi - lo), 0, 1)
    rgb = np.stack([g, g, g], axis=2)
    rgb[visceral] = 0.6 * rgb[visceral] + 0.4 * np.array([0.0, 1.0, 0.0])
    rgb[outline(np.isin(label, (2, 3)))] = (0.2, 0.4, 1.0)
    rgb[outline(label == 1)] = (1.0, 0.0, 0.0)
    return (rgb * 255).astype('uint8')


def process(job : dict, out_dir : Path) -> dict:
    row = {"name": job["name"], "gray": job["gray"], "label": job["label"]}
    try:
        grayimage = sitk.ReadImage(job["gray"])
        labelimage = sitk.ReadImage(job["label"])
        if labelimage.GetSize() != grayimage.GetSize():
            labelimage = sitk.Resample(labelimage, grayimage,
                                       sitk.Transform(), sitk.sitkNearestNeighbor)
            row["resampled"] = 1
        labelimage = sitk.Cast(labelimage, sitk.sitkUInt8)
        mask = sarcocalculate.visceral_mask(labelimage)
        for d in (out_dir, out_dir / "png"):
            (d / job["name"]).parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(mask, str(out_dir / (job["name"] + ".visceral.nii.gz")), True)

        labelarray = sitk.GetArrayViewFromImage(labelimage)
        zsum = np.sum(labelarray == 1, axis=(1, 2))
        if zsum.sum() == 0:
            row["status"] = "no_muscle"
            return row
        z = int(np.argmax(zsum))
        sp = grayimage.GetSpacing()
        cm2 = sp[0] * sp[1] / 100.0
        label = labelarray[z]
        visceral = sitk.GetArrayViewFromImage(mask)[z] > 0
        row.update(z=z, muscle_slices=int(np.count_nonzero(zsum)),
                   muscle_area_cm2=float(np.sum(label == 1) * cm2),
                   visceral_area_cm2=float(visceral.sum() * cm2),
                   muscle_coverage=muscle_coverage(label == 1),
                   status="ok")
        gray = sitk.GetArrayViewFromImage(grayimage)[z]
        imageio.imwrite(out_dir / "png" / (job["name"] + ".png"),
                        qc_png(gray, label, visceral))
    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jobs", help="CSV with columns gray, label, name")
    parser.add_argument("out_dir")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip rows whose mask already exists")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    (out_dir / "png").mkdir(parents=True, exist_ok=True)
    jobs = pd.read_csv(args.jobs).to_dict(orient="records")
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names):
        raise SystemExit("names in the job list are not unique")
    if args.skip_existing:
        jobs = [j for j in jobs
                if not (out_dir / (j["name"] + ".visceral.nii.gz")).exists()]

    rows = []
    # spawn, not fork: a caller with threads running could deadlock a forked child
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for i, row in enumerate(pool.map(process, jobs, [out_dir] * len(jobs),
                                         chunksize=4)):
            rows.append(row)
            if (i + 1) % 200 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)}", flush=True)

    manifest = pd.DataFrame(rows)
    path = out_dir / "manifest.csv"
    if path.exists():   # keep the rows of names not processed this time
        old = pd.read_csv(path)
        manifest = pd.concat([old[~old["name"].isin(manifest["name"])], manifest])
    manifest.to_csv(path, index=False)
    print(manifest["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
