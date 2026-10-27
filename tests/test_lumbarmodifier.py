"""bin/lumbarmodifier.py --mode passthrough: lumbar.py's z with the edge rule, no model."""

import pandas as pd
import pytest
import SimpleITK as sitk
from conftest import DZ, N_Z, make_gray

import lumbarmodifier


@pytest.fixture
def run(tmp_path):
    gray_path = tmp_path / "gray.nii.gz"
    sitk.WriteImage(make_gray(), str(gray_path))

    def run(z, likelihood=0.4, edge_mm=10.0):
        lst, lumbar_out, out = tmp_path / "list.txt", tmp_path / "l1.txt", tmp_path / "l1mod.txt"
        pd.DataFrame({"{volume_name}": [str(gray_path)]}).to_csv(lst, index=False)
        pd.DataFrame({"{volume_name}": [str(gray_path)], "{z_estimated}": [z],
                      "{z_true}": [-1], "{likelihood}": [likelihood]}).to_csv(lumbar_out, encoding="shift_jis")
        lumbarmodifier.main("passthrough", str(lst), 1, "cpu", 0.0, 0.0, "", str(lumbar_out), str(out),
                            0, 1, edge_mm=edge_mm)
        row = pd.read_csv(out, encoding="shift_jis").iloc[0]
        return int(row["{z_modified}"]), float(row["{likelihoods_modified}"])

    return run


def test_lumbar_z_is_kept(run):
    assert run(12, 0.4) == (12, pytest.approx(0.4))


def test_level_within_the_edge_margin_is_out_of_view(run):
    edge = int(round(10.0 / DZ))
    assert run(edge - 1)[0] == -1
    assert run(edge)[0] == edge
    assert run(N_Z - 1 - edge)[0] == N_Z - 1 - edge
    assert run(N_Z - edge)[0] == -1


def test_out_of_view_from_lumbar_or_beyond_the_volume_stays_out(run):
    assert run(-1) == (-1, -1000.0)
    assert run(N_Z + 3)[0] == -1
    assert run(12, likelihood=0.0)[0] == -1


def test_edge_margin_is_in_mm(run):
    assert run(3, edge_mm=20.0)[0] == -1
    assert run(4, edge_mm=20.0)[0] == 4


def test_edge_rule_off(run):
    assert run(0, edge_mm=0.0)[0] == 0
    assert run(N_Z - 1, edge_mm=0.0)[0] == N_Z - 1
    assert run(N_Z, edge_mm=0.0)[0] == -1
    assert run(-1, edge_mm=0.0) == (-1, -1000.0)
