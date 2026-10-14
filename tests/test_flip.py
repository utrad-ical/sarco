"""A "{flip}" column in a volume list reverses the slices of the gray and label
volumes on reading (training on feet-first volumes), and keys the caches apart."""

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk

import readfiles
import prepcache


def volume(tmp_path, name):
    arr = np.arange(6, dtype=np.int16)[:, None, None] * np.ones((1, 4, 4), np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 5.0))
    img.SetOrigin((0.0, 0.0, 10.0))
    p = tmp_path / name
    sitk.WriteImage(img, str(p))
    return str(p)


@pytest.mark.parametrize("value,expect", [(1, True), ("1", True), ("true", True), (0, False),
                                          ("0", False), ("", False), (np.nan, False)])
def test_flipped_values(value, expect):
    assert readfiles.flipped({"{volume_name}": "x", "{flip}": value}) is expect
    assert readfiles.flipped({"{volume_name}": "x"}) is False


def test_readone_reverses_gray_and_label_and_keeps_the_header(tmp_path):
    g, l = volume(tmp_path, "g.nii.gz"), volume(tmp_path, "l.nii.gz")
    row = {"{volume_name}": g, "{label_volume_name}": l}
    g0, l0 = readfiles.readone(row)
    g1, l1 = readfiles.readone(dict(row, **{"{flip}": 1}))
    a0, a1 = sitk.GetArrayFromImage(g0), sitk.GetArrayFromImage(g1)
    assert np.array_equal(a1, a0[::-1])
    assert np.array_equal(sitk.GetArrayFromImage(l1), sitk.GetArrayFromImage(l0)[::-1])
    assert g1.GetOrigin() == g0.GetOrigin() and g1.GetSpacing() == g0.GetSpacing()
    assert g1.GetDirection() == g0.GetDirection()


def test_flip_from_a_list_csv_and_the_cache_key(tmp_path):
    g, l = volume(tmp_path, "g.nii.gz"), volume(tmp_path, "l.nii.gz")
    lst = tmp_path / "list.csv"
    pd.DataFrame([{"{volume_name}": g, "{label_volume_name}": l, "{flip}": 1},
                  {"{volume_name}": g, "{label_volume_name}": l, "{flip}": 0}]
                 ).to_csv(lst, index=False, encoding="shift_jis")
    rows = [r for _, r in readfiles.read_list(str(lst)).iterrows()]
    assert readfiles.flipped(rows[0]) and not readfiles.flipped(rows[1])
    assert readfiles.flip_key(rows[0]) == {"flip": True} and readfiles.flip_key(rows[1]) == {}
    # flipped and plain reads of the same file are cached apart
    cache = tmp_path / "cache"
    first = lambda r: lambda: {"z0": int(sitk.GetArrayFromImage(readfiles.readone(r)[0])[0, 0, 0])}
    keys = [{"gray": prepcache.file_signature(g), **readfiles.flip_key(r)} for r in rows]
    a = prepcache.cached(str(cache), "lumbar_gray", keys[0], first(rows[0]))
    b = prepcache.cached(str(cache), "lumbar_gray", keys[1], first(rows[1]))
    assert (a["z0"], b["z0"]) == (5, 0)
