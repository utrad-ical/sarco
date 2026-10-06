"""bin/estimate.py with untrained checkpoints, and bin/resultuniter_circus.py.

The numbers themselves are meaningless (the models are random); what is
checked is that every estimate is made from the right inputs, combined and
converted by the documented formulas, and skipped cleanly when an input or a
checkpoint is missing.
"""

import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from conftest import BIN, make_gray, write_resultdir

import estimate

pytest.importorskip("ct25d")

METRICS = {lv: {"SMA": 100.0, "MSMD": 40.0, "VAT": 50.0, "SAT": 60.0,
                "meanVAT": -70.0, "meanSAT": -80.0,
                "cortical_density": 200.0, "medullary_density": 150.0}
           for lv in ("L1", "L3")}


def run(tmp_path, model_dir, levels=("L1", "L3"), orientation=None):
    res = tmp_path / "res"
    gray = write_resultdir(res, make_gray(), levels=levels, metrics=METRICS)
    if orientation is not None:
        (res / "__orientation.json").write_text(json.dumps(orientation))
    estimate.main(["--gray", str(gray), "--resultdir", str(res), "--model_dir",
                   str(model_dir), "--device", "cpu"])
    path = res / "__estimates.csv"
    if path.read_text().strip() == "":
        return pd.Series(dtype=object), res          # nothing could be estimated
    return pd.read_csv(path, encoding="shift_jis").iloc[0], res


def test_every_estimate_with_both_levels(tmp_path, model_dir):
    e, _ = run(tmp_path, model_dir, orientation={"flipped": True, "prob_feet_first": 0.9})
    for kind in ("marrow", "muscle"):
        for lv in ("L1", "L3"):
            k = f"age_{kind}_{lv}"
            assert np.isfinite(e[k]) and e[k + "_sigma"] > 0
            assert e[k + "_lo95"] < e[k] < e[k + "_hi95"]
    # inverse-variance combination, sigma widened by age_combination.json (2.0)
    names = [f"age_{k}_{lv}" for k in ("marrow", "muscle") for lv in ("L1", "L3")]
    w = np.array([1 / e[n + "_sigma"] ** 2 for n in names])
    mu = np.array([e[n] for n in names])
    assert e.age == pytest.approx((w * mu).sum() / w.sum())
    assert e.age_sigma == pytest.approx(2.0 * np.sqrt(1 / w.sum()))
    assert e.age_lo95 == pytest.approx(e.age - 1.96 * e.age_sigma)
    assert e.age_n_models == 4
    # phase: the mean of the L1 and L3 probabilities, its argmax
    probs = np.array([e[f"phase_prob_{k}"] for k in range(5)])
    assert probs.sum() == pytest.approx(1.0) and e.phase == probs.argmax()
    assert e.phase_n_levels == 2 and e.phase_name == estimate.PHASE_NAMES[e.phase]
    assert e.height_lo95 < e.height < e.height_hi95 and e.weight_sigma > 0
    assert e.input_feet_first == 1 and e.input_feet_first_prob == pytest.approx(0.9)
    # phase-0 values by the formula of the estimated phase
    k = int(e.phase)
    for lv in ("L1", "L3"):
        for v, x in METRICS[lv].items():
            if v not in estimate.CORRECTED:
                assert f"{lv}_{v}_phase0" not in e
                continue
            want = x if k == 0 else 10.0 * k + (1 + k / 10) * x
            assert e[f"{lv}_{v}_phase0"] == pytest.approx(want)


def test_phase_and_age_from_l1_alone(tmp_path, model_dir):
    """An upper-abdominal scan: L3 is out of view, so its result is empty."""
    e, _ = run(tmp_path, model_dir, levels=("L1",))
    assert e.phase_n_levels == 1 and e.phase == e.phase_L1
    assert "phase_L3" not in e and np.isfinite(e.phase)
    assert {"age_marrow_L1", "age_muscle_L1"} <= set(e.index)
    assert "age_marrow_L3" not in e and e.age_n_models == 2
    assert "height" not in e                     # body size needs both levels
    assert np.isfinite(e.L1_MSMD_phase0) and "L3_MSMD_phase0" not in e


def test_phase_from_l3_alone(tmp_path, model_dir):
    e, _ = run(tmp_path, model_dir, levels=("L3",))
    assert e.phase_n_levels == 1 and e.phase == e.phase_L3


def test_missing_checkpoints_skip_only_their_estimate(tmp_path, model_dir):
    partial = tmp_path / "model"
    shutil.copytree(model_dir, partial)
    for f in ("age_muscle_L1.pt", "phase_L3.pt", "bodysize.pt", "age_combination.json"):
        (partial / f).unlink()
    e, _ = run(tmp_path, partial)
    assert "age_muscle_L1" not in e and e.age_n_models == 3
    names = ["age_marrow_L1", "age_marrow_L3", "age_muscle_L3"]
    w = np.array([1 / e[n + "_sigma"] ** 2 for n in names])
    assert e.age_sigma == pytest.approx(np.sqrt(1 / w.sum()))       # no widening file
    assert e.phase_n_levels == 1 and "height" not in e


def test_no_level_at_all(tmp_path, model_dir):
    e, res = run(tmp_path, model_dir, levels=())
    assert len(e) == 0                              # an empty table, not an error
    uniter = [sys.executable, str(BIN / "resultuniter_circus.py"), str(res / "__result_L1.csv"),
              str(res / "__result_L3.csv"), str(res / "results.json"), str(res / "__estimates.csv")]
    subprocess.run(uniter, check=True, capture_output=True)
    assert "estimates" not in json.loads((res / "results.json").read_text())["results"]


def test_results_json_carries_the_estimates(tmp_path, model_dir):
    _, res = run(tmp_path, model_dir)
    uniter = [sys.executable, str(BIN / "resultuniter_circus.py"), str(res / "__result_L1.csv"),
              str(res / "__result_L3.csv"), str(res / "results.json")]
    subprocess.run(uniter + [str(res / "__estimates.csv")], check=True, capture_output=True)
    d = json.loads((res / "results.json").read_text())["results"]
    assert d["L1"][0]["L1_MSMD"] == 40.0
    est = d["estimates"][0]
    assert {"age", "phase_name", "height", "L3_MSMD_phase0", "age_muscle_L3"} <= set(est)
    # every value the plugin displays resolves
    plugin = json.loads((BIN.parent / "plugin.json").read_text())
    for im in plugin["displayStrategy"][0]["options"]["imageInfo"]:
        for lab in im.get("labelData", []):
            if "dataPath" in lab:
                v = {"results": d}
                for key in lab["dataPath"].split("."):
                    v = v[int(key)] if isinstance(v, list) else v[key]
    # without (or with an empty) estimates file the tables still merge
    (res / "empty.csv").write_text("")
    for extra in ([], [str(res / "empty.csv")], [str(res / "nonexistent.csv")]):
        subprocess.run(uniter + extra, check=True, capture_output=True)
        d = json.loads((res / "results.json").read_text())["results"]
        assert "estimates" not in d and d["L3"][0]["L3_SMA"] == 100.0


def test_a_level_without_any_table_still_gives_results_json(tmp_path, model_dir):
    """lumbar/modifier could not place L3 at all: no __result_L3.csv is written."""
    _, res = run(tmp_path, model_dir, levels=("L1",))
    (res / "__result_L3.csv").unlink()
    subprocess.run([sys.executable, str(BIN / "resultuniter_circus.py"),
                    str(res / "__result_L1.csv"), str(res / "__result_L3.csv"),
                    str(res / "results.json"), str(res / "__estimates.csv")],
                   check=True, capture_output=True)
    d = json.loads((res / "results.json").read_text())["results"]
    assert d["L3"] == [] and d["L1"][0]["L1_MSMD"] == 40.0
    assert d["estimates"][0]["phase_n_levels"] == 1


def test_body_size_from_one_level_with_a_level_dropping_model(tmp_path, model_dir):
    from conftest import write_bodysize

    m = tmp_path / "model"
    shutil.copytree(model_dir, m)
    write_bodysize(m / "bodysize.pt", p_drop=0.15)
    for levels in (("L1",), ("L3",)):
        e, _ = run(tmp_path / levels[0], m, levels=levels)
        assert e.body_n_levels == 1 and e.height_lo95 < e.height < e.height_hi95
    e, _ = run(tmp_path / "both", m)
    assert e.body_n_levels == 2
