"""bin/orientation.py: features in index order, check / restore round trip, training."""

import json

import numpy as np
import pandas as pd
import pytest
import SimpleITK as sitk
from conftest import L1_Z, N_Z, make_gray, write_resultdir

import orientation

torch = pytest.importorskip("torch")


def reversed_copy(img):
    out = sitk.GetImageFromArray(sitk.GetArrayFromImage(img)[::-1].copy())
    out.CopyInformation(img)
    return out


def constant_model(path, logit):
    """A classifier that always answers `logit` (weights zero, bias set)."""
    model = orientation.build_model()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model[-1].bias.fill_(logit)
    torch.save({"state_dict": model.state_dict()}, path)


def test_features_follow_the_index_order_not_the_header():
    a = orientation.features(make_gray(direction_z=1.0))
    b = orientation.features(make_gray(direction_z=-1.0))
    assert a.shape == (4, N_Z, orientation.COLS)       # 5 mm slices -> one row each
    assert np.allclose(a, b)
    c = orientation.features(reversed_copy(make_gray()))
    assert np.allclose(c, a[:, ::-1], atol=1e-5)
    assert 0.0 <= a.min() and a.max() <= 1.0


def test_long_volumes_are_scored_in_windows(tmp_path):
    constant_model(tmp_path / "m.pt", 3.0)
    model = orientation.load_model(tmp_path / "m.pt")
    p = orientation.prob_feet_first(model, np.zeros((4, 300, orientation.COLS), np.float32))
    assert p == pytest.approx(1 / (1 + np.exp(-3.0)), rel=1e-5)


def run_check(tmp_path, gray, model):
    inp = tmp_path / "in.mhd"
    sitk.WriteImage(gray, str(inp))
    out, rep = tmp_path / "work/0.mhd", tmp_path / "res/__orientation.json"
    orientation.main(["check", "--input", str(inp), "--output", str(out), "--report", str(rep),
                      "--model", str(model), "--device", "cpu"])
    return sitk.ReadImage(str(out)), json.loads(rep.read_text()), rep


def test_check_without_a_model_passes_the_input_through(tmp_path):
    gray = make_gray()
    out, rep, _ = run_check(tmp_path, gray, tmp_path / "missing.pt")
    assert rep["flipped"] is False and "prob_feet_first" not in rep
    assert np.array_equal(sitk.GetArrayFromImage(out), sitk.GetArrayFromImage(gray))


def test_head_first_input_is_left_alone(tmp_path):
    constant_model(tmp_path / "m.pt", -5.0)
    gray = make_gray()
    out, rep, _ = run_check(tmp_path, gray, tmp_path / "m.pt")
    assert rep["flipped"] is False and rep["prob_feet_first"] < 0.5
    assert np.array_equal(sitk.GetArrayFromImage(out), sitk.GetArrayFromImage(gray))


def test_feet_first_input_is_reversed_and_the_results_restored(tmp_path):
    constant_model(tmp_path / "m.pt", 5.0)
    feet_first = reversed_copy(make_gray())
    out, rep, rep_path = run_check(tmp_path, feet_first, tmp_path / "m.pt")
    assert rep["flipped"] is True
    head_first = sitk.GetArrayFromImage(out)
    assert np.array_equal(head_first, sitk.GetArrayFromImage(make_gray()))
    assert out.GetOrigin() == feet_first.GetOrigin()          # header kept
    assert out.GetDirection() == feet_first.GetDirection()

    # the pipeline writes its results on the head-first copy ...
    res = rep_path.parent
    write_resultdir(res, out)
    orientation.main(["restore", "--report", str(rep_path), "--resultdir", str(res)])
    # ... and they come back in the input's slice order
    lab = sitk.GetArrayFromImage(sitk.ReadImage(str(res / "0_L1.result.nii.gz")))
    assert np.flatnonzero(lab.reshape(N_Z, -1).any(1)).tolist() == [N_Z - 1 - L1_Z]
    vis = sitk.GetArrayFromImage(sitk.ReadImage(str(res / "0_L1.visceral.nii.gz")))
    assert np.flatnonzero(vis.reshape(N_Z, -1).any(1)).tolist() == [N_Z - 1 - L1_Z]
    row = pd.read_csv(res / "__result_L1.csv", encoding="shift_jis").iloc[0]
    assert row["{z_estimated}"] == N_Z - 1 - L1_Z and row["{z_modified}"] == N_Z - 1 - L1_Z
    assert row["{volume_name}"] == rep["input"]


def test_restore_keeps_out_of_view_levels(tmp_path):
    constant_model(tmp_path / "m.pt", 5.0)
    out, rep, rep_path = run_check(tmp_path, reversed_copy(make_gray()), tmp_path / "m.pt")
    write_resultdir(rep_path.parent, out, levels=("L1",))
    orientation.main(["restore", "--report", str(rep_path), "--resultdir", str(rep_path.parent)])
    row = pd.read_csv(rep_path.parent / "__result_L3.csv", encoding="shift_jis").iloc[0]
    assert row["{z_estimated}"] == -1


def test_training_writes_a_loadable_checkpoint(tmp_path):
    """A smoke test: whether it learns is judged on the real cohorts."""
    rows = []
    for i in range(6):
        np.save(tmp_path / f"{i}.npy", orientation.features(make_gray()))
        rows.append(dict(feature=str(tmp_path / f"{i}.npy"),
                         split=(["train"] * 4 + ["val", "test"])[i]))
    pd.DataFrame(rows).to_csv(tmp_path / "list.csv", index=False)
    orientation.main(["train", "--list", str(tmp_path / "list.csv"), "--out",
                      str(tmp_path / "o.pt"), "--epochs", "1", "--batch-size", "4",
                      "--device", "cpu"])
    ck = torch.load(tmp_path / "o.pt", weights_only=True)
    assert set(ck["metrics"]) == {f"test_acc_{mm}mm" for mm in (100, 150, 200, 300, 2000)}
    p = orientation.prob_feet_first(orientation.load_model(tmp_path / "o.pt"),
                                    orientation.features(make_gray()))
    assert 0.0 <= p <= 1.0


def test_an_uncertain_answer_is_not_acted_on(tmp_path):
    constant_model(tmp_path / "m.pt", 0.5)                 # p = 0.62 < threshold 0.7
    gray = make_gray()
    out, rep, _ = run_check(tmp_path, gray, tmp_path / "m.pt")
    assert 0.5 < rep["prob_feet_first"] < rep["threshold"] and rep["flipped"] is False
    assert np.array_equal(sitk.GetArrayFromImage(out), sitk.GetArrayFromImage(gray))
