"""sarcocalculate.calculate on a phantom whose areas and densities are known."""

import numpy as np
import pytest
import SimpleITK as sitk
from conftest import L1_Z, PX, disc, label_slice, make_gray, make_label

import sarcocalculate


def test_metrics_of_the_phantom():
    gray = make_gray()
    arr = sitk.GetArrayFromImage(gray)
    fat = disc(20, cx=-30)                     # visceral fat inside the ring
    arr[L1_Z][fat] = -100
    g = sitk.GetImageFromArray(arr)
    g.CopyInformation(gray)
    volume, png, m = sarcocalculate.calculate(g, make_label(g, L1_Z))
    lab = label_slice()
    px2 = PX * PX
    assert m["SMA"] == pytest.approx((lab == 1).sum() * px2)        # 50+z HU, all in range
    assert m["MSMD"] == pytest.approx(50.0 + L1_Z)
    assert m["VAT"] == pytest.approx(fat.sum() * px2)
    assert m["meanVAT"] == pytest.approx(-100.0)
    assert m["cortical_density"] == pytest.approx(300.0)
    assert m["medullary_density"] == pytest.approx(300.0)
    assert m["medullary_area"] == pytest.approx((lab == 3).sum() * px2)
    assert np.flatnonzero(volume.reshape(len(volume), -1).any(1)).tolist() == [L1_Z]
    assert png.shape == (4 * lab.shape[0], 2 * lab.shape[1])      # 4 x 2 panels


def test_an_empty_result_gives_zeros():
    gray = make_gray()
    _, _, m = sarcocalculate.calculate(gray, make_label(gray, None))
    assert m["SMA"] == 0 and m["MSMD"] == 0
