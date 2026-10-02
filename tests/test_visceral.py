"""sarcocalculate.visceral_mask and bin/tools/make_visceral_masks.py."""

import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from conftest import BIN, L1_Z, label_slice, make_gray, make_label

import sarcocalculate


def test_mask_is_inside_the_ring_without_muscle_and_vertebra():
    gray = make_gray()
    lab = make_label(gray, L1_Z)
    m = sarcocalculate.visceral_mask(lab)
    v = sitk.GetArrayFromImage(m)
    assert m.GetOrigin() == lab.GetOrigin() and m.GetSpacing() == lab.GetSpacing()
    assert m.GetDirection() == lab.GetDirection()
    assert np.flatnonzero(v.reshape(len(v), -1).any(1)).tolist() == [L1_Z]
    s, labels = v[L1_Z] > 0, label_slice()
    assert not (s & np.isin(labels, (1, 2, 3))).any()
    inner = np.zeros_like(s)
    inner[50:78, 50:78] = True                 # well inside the ring, away from bone
    inner &= labels == 0
    assert s[inner].all()
    corner = np.zeros_like(s)
    corner[:10, :10] = True
    assert not s[corner].any()


def test_mask_takes_the_slice_with_the_most_muscle():
    gray = make_gray()
    arr = np.zeros(sitk.GetArrayFromImage(gray).shape, np.uint8)
    arr[5] = label_slice()
    arr[6][label_slice() == 1] = 1
    arr[6][:, :64] = 0                         # half the ring on slice 6
    lab = sitk.GetImageFromArray(arr)
    lab.CopyInformation(gray)
    v = sitk.GetArrayFromImage(sarcocalculate.visceral_mask(lab))
    assert v[5].any() and not v[6].any()


def test_empty_label_gives_an_empty_mask():
    gray = make_gray()
    v = sitk.GetArrayFromImage(sarcocalculate.visceral_mask(make_label(gray, None)))
    assert v.sum() == 0


def test_make_visceral_masks_tool(tmp_path):
    sys.path.insert(0, str(BIN / "tools"))
    import make_visceral_masks

    gray = make_gray()
    sitk.WriteImage(gray, str(tmp_path / "g.nii.gz"))
    sitk.WriteImage(make_label(gray, L1_Z), str(tmp_path / "l.nii.gz"))
    sitk.WriteImage(make_label(gray, None), str(tmp_path / "e.nii.gz"))
    jobs = pd.DataFrame([dict(gray=tmp_path / "g.nii.gz", label=tmp_path / "l.nii.gz", name="a/one"),
                         dict(gray=tmp_path / "g.nii.gz", label=tmp_path / "e.nii.gz", name="a/empty")])
    jobs.to_csv(tmp_path / "jobs.csv", index=False)
    out = tmp_path / "out"
    make_visceral_masks.main([str(tmp_path / "jobs.csv"), str(out), "--workers", "1"])
    m = pd.read_csv(out / "manifest.csv").set_index("name")
    assert m.loc["a/one", "status"] == "ok" and m.loc["a/one", "z"] == L1_Z
    assert m.loc["a/one", "muscle_coverage"] > 0.95
    assert m.loc["a/empty", "status"] == "no_muscle"
    assert (out / "a/one.visceral.nii.gz").exists() and (out / "png/a/one.png").exists()
    # a second run on a subset keeps the other rows of the manifest
    jobs.iloc[[1]].to_csv(tmp_path / "jobs1.csv", index=False)
    make_visceral_masks.main([str(tmp_path / "jobs1.csv"), str(out), "--workers", "1"])
    assert set(pd.read_csv(out / "manifest.csv").name) == {"a/one", "a/empty"}
