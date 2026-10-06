"""Synthetic volumes and tiny checkpoints for the sarco tests.

Nothing here needs the real data, the segmentation checkpoints or a GPU: the
phantom is an elliptic trunk with a muscle ring and a vertebral body on two
slices (L1 and L3), and the ct25d models are untrained, so the tests check the
plumbing and the arithmetic, not the accuracy.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sitk = pytest.importorskip("SimpleITK")

BIN = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN))

N_XY, PX, N_Z, DZ = 128, 2.0, 30, 5.0
L1_Z, L3_Z = 10, 20                       # slice 0 is cranial


def disc(r_mm, cy=0.0, cx=0.0, sy=1.0):
    yy, xx = (np.mgrid[0:N_XY, 0:N_XY] - (N_XY - 1) / 2.0) * PX
    return ((xx - cx) / r_mm) ** 2 + ((yy - cy) / (sy * r_mm)) ** 2 <= 1.0


def label_slice():
    """muscle ring (1), cortical (2) and medullary (3) bone of one slice."""
    lab = np.zeros((N_XY, N_XY), np.uint8)
    lab[disc(95, sy=0.75) & ~disc(85, sy=0.72)] = 1
    lab[disc(16, cy=40) & ~disc(13, cy=40)] = 2
    lab[disc(13, cy=40)] = 3
    return lab


def make_gray(direction_z=1.0):
    vol = np.full((N_Z, N_XY, N_XY), -1000.0, np.float32)
    body = disc(110, sy=0.75)
    vol[:, body] = 10.0                              # fat-ish trunk
    vol[:, disc(80, sy=0.7)] = 40.0                  # viscera
    lab = label_slice()
    for z in range(N_Z):
        vol[z][lab == 1] = 50.0 + z                  # muscle, varies along z
        vol[z][lab >= 2] = 300.0
    # lungs in the cranial third, so the volume has an up and a down
    vol[:8][:, disc(30, cx=-45) | disc(30, cx=45)] = -850.0
    img = sitk.GetImageFromArray(vol.astype(np.int16))
    img.SetSpacing((PX, PX, DZ))
    img.SetOrigin((-(N_XY - 1) / 2 * PX, -(N_XY - 1) / 2 * PX, 0.0))
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, direction_z))
    return img


def make_label(gray, z):
    arr = np.zeros((N_Z, N_XY, N_XY), np.uint8)
    if z is not None:
        arr[z] = label_slice()
    lab = sitk.GetImageFromArray(arr)
    lab.CopyInformation(gray)
    return lab


@pytest.fixture
def gray():
    return make_gray()


def write_resultdir(d: Path, gray, levels=("L1", "L3"), metrics=None):
    """A sarco result directory as sarcosegmentation.py leaves it."""
    import sarcocalculate

    d.mkdir(parents=True, exist_ok=True)
    gray_path = d / "gray.nii.gz"
    sitk.WriteImage(gray, str(gray_path))
    for lv, z in (("L1", L1_Z), ("L3", L3_Z)):
        present = lv in levels
        lab = make_label(gray, z if present else None)
        sitk.WriteImage(lab, str(d / f"0_{lv}.result.nii.gz"))
        sitk.WriteImage(sarcocalculate.visceral_mask(lab), str(d / f"0_{lv}.visceral.nii.gz"))
        row = {"{volume_name}": str(gray_path), "{z_estimated}": z if present else -1,
               "{z_modified}": z if present else -1,
               "{%s_result_path}" % lv: f"0_{lv}.result.nii.gz",
               "{%s_visceral_path}" % lv: f"0_{lv}.visceral.nii.gz", "id": 0}
        m = (metrics or {}).get(lv, {})
        row.update({f"{lv}_{k}": (v if present else 0) for k, v in m.items()})
        pd.DataFrame([row]).to_csv(d / f"__result_{lv}.csv", encoding="shift_jis")
    return gray_path


SMALL_CT25D = dict(crop_size=32, n_slices=3, gap_mm=5.0, slab_mm=0.0, in_plane_mm=4.0,
                   window=[-1000.0, 1000.0], mask_channel="sdf", keep_context=False,
                   gate=None, log_target=False, target_power=0, hu_shift=0.0)


def write_ct25d(path, task, label_value, seed):
    import torch
    from ct25d.checkpoint import build_model, save_checkpoint
    from ct25d.transforms import TargetStandardizer

    torch.manual_seed(seed)
    arch = dict(name="resnet18", n_slices=3, n_mask_channels=1, norm="group",
                dropout=0.0, use_cbam=True)
    if task == "classification":
        arch["n_classes"] = 5
    model = build_model(arch)
    cfg = dict(SMALL_CT25D, arch=arch, task=task, label_value=label_value,
               target="phase" if task == "classification" else "age",
               image_col="gray", mask_col="mask", n_classes=arch.get("n_classes", 0))
    std = None if task == "classification" else TargetStandardizer().fit([30.0, 80.0])
    save_checkpoint(path, model, std, cfg, sigma_scale=1.0, temperature=1.0)


def write_bodysize(path, p_drop=0.0):
    import torch
    from ct25d.bodysize import BodySizeGeometry, BodySizeNet, save_bodysize
    from ct25d.transforms import TargetStandardizer

    torch.manual_seed(0)
    g = BodySizeGeometry(proj_mm=8.0, rows=64, cols=32, l3_row=24, axial_mm=4.0,
                         axial_size=64)
    stds = {"height": TargetStandardizer().fit([150.0, 175.0]),
            "weight": TargetStandardizer().fit([45.0, 80.0])}
    cfg = dict(model=dict(norm="group", dropout=0.0, use_cbam=True, n_targets=2),
               targets=["height", "weight"], geometry=g.as_dict(), p_drop=p_drop)
    save_bodysize(path, BodySizeNet(**cfg["model"]), stds, cfg,
                  {"height": 1.0, "weight": 1.0})


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    """model/ with every estimate checkpoint (untrained) and the two JSON files."""
    pytest.importorskip("ct25d")
    d = tmp_path_factory.mktemp("model")
    for i, (kind, lv) in enumerate([("marrow", "L1"), ("marrow", "L3"),
                                    ("muscle", "L1"), ("muscle", "L3")]):
        write_ct25d(d / f"age_{kind}_{lv}.pt", "regression",
                    3 if kind == "marrow" else 1, seed=i)
    for i, lv in enumerate(("L1", "L3")):
        write_ct25d(d / f"phase_{lv}.pt", "classification", 1, seed=10 + i)
    write_bodysize(d / "bodysize.pt")
    # phase k: value0 = 10 k + (1 + k / 10) value_k, easy to recognise
    coef = {lv: {v: {str(k): {"a": 10.0 * k, "b": 1.0 + k / 10, "n": 100}
                     for k in range(1, 5)}
                 for v in ("MSMD", "meanVAT", "meanSAT", "cortical_density",
                           "medullary_density")}
            for lv in ("L1", "L3")}
    (d / "phase_correction.json").write_text(json.dumps({"coefficients": coef}))
    (d / "age_combination.json").write_text(json.dumps({"combined_sigma_scale": 2.0}))
    return d
