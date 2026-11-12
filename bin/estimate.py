"""
Estimates from the L1/L3 results with ct25d-regressor checkpoints in model/:

  age      age_marrow_L1/L3.pt (medullary bone, label 3), age_muscle_L1/L3.pt
           (muscle, label 1); combined by inverse-variance weighting, the
           combined sigma widened by age_combination.json
  phase    phase_L1/L3.pt (visceral mask); the L1 and L3 probabilities are
           averaged and the most probable phase is taken, one per examination;
           the phase names follow the number of classes (PHASE_SCHEMES:
           5 = v8, 0 plain .. 4 equilibrium; 6 = v9, with portal and venous
           split)
  body     bodysize.pt: height and weight from the trunk projections and the
           L1/L3 planes
  phase-0  phase_correction.json: MSMD, meanVAT, meanSAT, cortical_density,
           medullary_density and SASHI converted to the plain-CT equivalent
           with the linear formula of the estimated phase (unchanged for phase
           0); only when the file's phase list matches the phase models'
           classes. SASHI is a fraction: 0 is a value, the result is kept in
           [0, 1]

A missing checkpoint, or a level without a result, skips that estimate and the
rest still runs. Writes <resultdir>/__estimates.csv (one row; shift_jis like
the other result tables), which resultuniter_circus.py adds to results.json.

  python bin/estimate.py --gray in/0.mhd --resultdir out --model_dir model
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

import extract_trunk
import prepcache
import readfiles


LEVELS = ("L1", "L3")
# contrast phases by the number of classes of the phase models
PHASE_SCHEMES = {
    5: {0: "plain", 1: "early arterial", 2: "late arterial", 3: "portal venous",
        4: "equilibrium"},
    6: {0: "plain", 1: "early arterial", 2: "late arterial", 3: "portal",
        4: "venous", 5: "equilibrium"},
}
PHASE_NAMES = PHASE_SCHEMES[5]
CORRECTED = ("MSMD", "meanVAT", "meanSAT", "cortical_density", "medullary_density", "SASHI")
# fractions (SASHI: muscle below -30 HU): 0 is a real value when the level has a
# result (SMA > 0), and the corrected value stays in [0, 1]
FRACTIONS = ("SASHI",)
# areas (mm2): converted when phase_correction.json has their coefficients, by the
# formula named in its "forms" ({"VAT": "power", ...}): power: value0 = exp(a) * value_k ** b
# (positive, through the origin); a variable not listed there is linear: a + b * value_k
AREAS = ("SMA", "VAT", "SAT")
AGE_MODELS = [("marrow", "L1"), ("marrow", "L3"), ("muscle", "L1"), ("muscle", "L3")]
# direction cosines that make physical z follow the (head-first) index order
INDEX_ORDER = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)


def level_inputs(resultdir : Path, level : str):
    """(result label, visceral mask, metrics row) of one level, or Nones."""
    csv = resultdir / f"__result_{level}.csv"
    if not csv.exists():
        return None, None, None
    row = pd.read_csv(csv, encoding="shift_jis").iloc[0]
    def image(col):
        name = row.get("{%s_%s_path}" % (level, col))
        if not isinstance(name, str):
            return None
        path = resultdir / name
        return sitk.ReadImage(str(path)) if path.exists() else None
    return image("result"), image("visceral"), row


def has_label(image, value):
    return image is not None and bool(np.any(sitk.GetArrayViewFromImage(image) == value))


def run(name, fn, out):
    try:
        fn(out)
    except Exception:
        print(f"[estimate] {name} failed; skipped")
        traceback.print_exc()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gray", required=True)
    p.add_argument("--resultdir", required=True)
    p.add_argument("--model_dir", default="model")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache_dir", default=None)
    args = p.parse_args(argv)

    import torch
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    try:
        from ct25d.inference import Predictor
    except ImportError:
        print("[estimate] ct25d is not installed; no estimates")
        return

    model_dir, resultdir = Path(args.model_dir), Path(args.resultdir)
    gray, _ = readfiles.readone({"{volume_name}": args.gray})
    inputs = {lv: level_inputs(resultdir, lv) for lv in LEVELS}
    out = {}

    def predictor(name):
        path = model_dir / name
        if not path.exists():
            print(f"[estimate] {path} not found; skipped")
            return None
        return Predictor(path, device=device)

    def age(out):
        mus, ws = [], []
        for kind, lv in AGE_MODELS:
            label = inputs[lv][0]
            value = 3 if kind == "marrow" else 1
            if not has_label(label, value):
                continue
            pr = predictor(f"age_{kind}_{lv}.pt")
            if pr is None:
                continue
            r = pr.predict(gray, label)
            key = f"age_{kind}_{lv}"
            out.update({key: r["mean"], key + "_sigma": r["sigma"],
                        key + "_lo95": r["lo95"], key + "_hi95": r["hi95"]})
            mus.append(r["mean"])
            ws.append(1.0 / r["sigma"] ** 2)
        if ws:
            # the models are not independent: age_combination.json holds the
            # scale for sqrt(1 / sum w) fitted on the validation split
            w = np.asarray(ws)
            m = float(np.sum(w * np.asarray(mus)) / w.sum())
            scale = 1.0
            comb = model_dir / "age_combination.json"
            if comb.exists():
                scale = float(json.loads(comb.read_text())["combined_sigma_scale"])
            s = scale * float(np.sqrt(1.0 / w.sum()))
            out.update(age=m, age_sigma=s, age_lo95=m - 1.96 * s,
                       age_hi95=m + 1.96 * s, age_n_models=len(ws))

    def phase(out):
        probs = []
        for lv in LEVELS:
            mask = inputs[lv][1]
            if not has_label(mask, 1):
                continue
            pr = predictor(f"phase_{lv}.pt")
            if pr is None:
                continue
            r = pr.predict(gray, mask)
            out[f"phase_{lv}"] = r["pred"]
            out[f"phase_{lv}_conf"] = r["conf"]
            probs.append(r["probs"])
        if len({len(p) for p in probs}) > 1:
            print("[estimate] the L1 and L3 phase models have different classes "
                  f"({[len(p) for p in probs]}); no combined phase")
            return
        if probs:
            prob = np.mean(probs, axis=0)
            k = int(np.argmax(prob))
            names = PHASE_SCHEMES.get(len(prob), {})
            out.update(phase=k, phase_name=names.get(k, str(k)),
                       phase_conf=float(prob[k]), phase_n_levels=len(probs),
                       phase_n_classes=len(prob))
            out.update({f"phase_prob_{i}": float(v) for i, v in enumerate(prob)})

    def body(out):
        path = model_dir / "bodysize.pt"
        if not path.exists():
            print(f"[estimate] {path} not found; skipped")
            return
        labels = [inputs[lv][0] for lv in LEVELS]
        present = [lab is not None and bool(np.any(sitk.GetArrayViewFromImage(lab) > 0))
                   for lab in labels]
        from ct25d.bodysize import (BodySizeGeometry, level_z_mm, load_bodysize,
                                    predict_bodysize, prepare_case)
        from ct25d.geometry import with_direction
        model, stds, cfg, scales, _ = load_bodysize(path, map_location=device)
        # a model trained with level dropping (p_drop) also works with one level
        one_level_ok = float(cfg.get("p_drop", 0.0)) > 0
        if not all(present) and not (one_level_ok and any(present)):
            print("[estimate] body size needs " + ("L1 or L3" if one_level_ok else
                  "both L1 and L3") + "; skipped")
            return
        g = cfg["geometry"]
        geometry = BodySizeGeometry(**{k: (tuple(v) if k == "window" else v)
                                       for k, v in g.items()})
        # the slices are head first here (orientation.py), so the frame is
        # set from the index order rather than from the header's z direction
        z = [None, None]
        for k, lab in enumerate(labels):
            if present[k]:
                z[k] = level_z_mm(with_direction(gray, lab, INDEX_ORDER)[1])
        g_im, _ = with_direction(gray, labels[present.index(True)], INDEX_ORDER)
        mask = prepcache.cached(args.cache_dir, "trunk_mask",
                                {"gray": prepcache.file_signature(args.gray)},
                                lambda: {"mask": extract_trunk.trunk_mask(gray)})["mask"]
        case = prepare_case(g_im, mask.astype(bool), z[0], z[1], geometry=geometry)
        out["body_n_levels"] = int(sum(present))
        res = predict_bodysize(model, stds, scales, [case], cfg["targets"], device=device)
        for t, (m, s, lo, hi) in res.items():
            out.update({t: float(m[0]), t + "_sigma": float(s[0]),
                        t + "_lo95": float(lo[0]), t + "_hi95": float(hi[0])})

    def corrected(out):
        path = model_dir / "phase_correction.json"
        if "phase" not in out or not path.exists():
            print("[estimate] no phase estimate or no phase_correction.json; "
                  "no phase-0 values")
            return
        table = json.loads(path.read_text())
        # the formulas are per phase index: the file must use the same phases
        # (a file without the list is the 5-phase v8 one)
        n_file = len(table.get("phases", PHASE_SCHEMES[5]))
        if n_file != int(out["phase_n_classes"]):
            print(f"[estimate] phase_correction.json has {n_file} phases, the phase "
                  f"models {int(out['phase_n_classes'])}; no phase-0 values")
            return
        coef = table["coefficients"]
        forms = table.get("forms", {})
        k = int(out["phase"])
        for lv in LEVELS:
            row = inputs[lv][2]
            if row is None:
                continue
            for v in CORRECTED + AREAS:
                if v in AREAS and v not in coef.get(lv, {}):
                    continue          # a file without area formulas (v8.5 and earlier)
                x = row.get(f"{lv}_{v}")
                if x is None or not np.isfinite(x):
                    continue
                if x == 0 and (v not in FRACTIONS or not row.get(f"{lv}_SMA", 0) > 0):
                    continue          # 0 is sarco's value for an empty result
                if v in AREAS and x < 0:
                    continue
                if k == 0:
                    y = float(x)
                else:
                    c = coef.get(lv, {}).get(v, {}).get(str(k))
                    if c is None:
                        continue
                    if forms.get(v, "linear") == "power":
                        y = float(np.exp(c["a"]) * float(x) ** c["b"])
                    else:
                        y = c["a"] + c["b"] * float(x)
                if v in FRACTIONS:
                    y = min(max(y, 0.0), 1.0)
                out[f"{lv}_{v}_phase0"] = y

    orient = resultdir / "__orientation.json"
    if orient.exists():
        o = json.loads(orient.read_text())
        out["input_feet_first"] = int(bool(o.get("flipped")))
        if "prob_feet_first" in o:
            out["input_feet_first_prob"] = float(o["prob_feet_first"])

    for name, fn in (("age", age), ("phase", phase), ("body size", body),
                     ("phase-0 values", corrected)):
        run(name, fn, out)

    print(json.dumps(out, indent=1, default=float))
    pd.DataFrame([out]).to_csv(resultdir / "__estimates.csv", index=False,
                               encoding="shift_jis")


if __name__ == "__main__":
    sys.exit(main())
