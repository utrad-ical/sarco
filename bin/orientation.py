"""
Craniocaudal orientation of the input volume: is slice 0 at the head end?

The whole pipeline (lumbar.py onwards) works in array index order and expects
the slices to run from head to feet. Headers cannot be trusted for this (one
of our cohorts has every z direction reversed), so the orientation is
judged from the image: a small CNN on coronal and sagittal projections in
index order, trained on head-first volumes flipped at random and cut to
random craniocaudal ranges (so that a short upper-abdominal scan works too).

  check    write the input in head-first order (the slices reversed if the
           volume runs feet first; the header is kept) and a report
  restore  put the result labels in <resultdir> back into the input's order
           and fix the slice numbers and paths in the result tables
  train    train the classifier from a list of feature files

  python bin/orientation.py check --input in/0.mhd --output /temp/0.mhd \\
      --report out/__orientation.json --model model/orientation.pt
  python bin/orientation.py restore --report out/__orientation.json --resultdir out
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROW_MM = 5.0          # projection rows (index-order z)
COL_MM = 4.0          # projection columns
COLS = 128            # 512 mm
WINDOW_ROWS = 128     # 640 mm per network window
HU = (-1000.0, 1000.0)


def features(image : sitk.Image) -> np.ndarray:
    """
    (4, rows, COLS) float32 in [0, 1]: mean and maximum projections along y
    and along x, in index order (row 0 = slice 0), whatever the header says.
    """
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    while arr.ndim > 3:
        arr = arr.mean(axis=-1)
    sp = image.GetSpacing()
    plain = sitk.GetImageFromArray(arr)
    plain.SetSpacing(sp)                         # identity direction, origin 0
    nz, ny, nx = arr.shape
    rows = max(1, int(round(nz * sp[2] / ROW_MM)))
    f = sitk.ResampleImageFilter()
    f.SetOutputSpacing((COL_MM, COL_MM, ROW_MM))
    f.SetSize((COLS, COLS, rows))
    f.SetOutputOrigin(((nx - 1) * sp[0] / 2 - (COLS - 1) * COL_MM / 2,
                       (ny - 1) * sp[1] / 2 - (COLS - 1) * COL_MM / 2,
                       (sp[2] - ROW_MM) / 2))
    f.SetDefaultPixelValue(HU[0])
    f.SetInterpolator(sitk.sitkLinear)
    if sp[2] < ROW_MM:
        plain = sitk.SmoothingRecursiveGaussian(plain, (0.01, 0.01, ROW_MM / 2))
    v = sitk.GetArrayFromImage(f.Execute(plain))
    v = np.clip((v - HU[0]) / (HU[1] - HU[0]), 0.0, 1.0)
    return np.stack([v.mean(1), v.mean(2), v.max(1), v.max(2)]).astype(np.float32)


def window_input(feat, start, length):
    """(5, WINDOW_ROWS, COLS): the rows [start, start + length) and a validity channel."""
    x = np.zeros((5, WINDOW_ROWS, COLS), np.float32)
    part = feat[:, start:start + length]
    x[:4, :part.shape[1]] = part
    x[4, :part.shape[1]] = 1.0
    return x


def build_model():
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                             nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
                             nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                             nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
                             nn.MaxPool2d(2))
    return nn.Sequential(block(5, 16), block(16, 32), block(32, 64), block(64, 128),
                         block(128, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                         nn.Dropout(0.2), nn.Linear(128, 1))


def load_model(path, device="cpu"):
    import torch
    ck = torch.load(str(path), map_location=device, weights_only=True)
    model = build_model()
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def prob_feet_first(model, feat, device="cpu") -> float:
    """Probability that slice 0 is at the feet end; windows of the whole volume, averaged."""
    import torch
    n = feat.shape[1]
    starts = [0] if n <= WINDOW_ROWS else sorted(set(
        list(range(0, n - WINDOW_ROWS + 1, WINDOW_ROWS // 2)) + [n - WINDOW_ROWS]))
    x = torch.from_numpy(np.stack([window_input(feat, s, WINDOW_ROWS) for s in starts]))
    with torch.no_grad():
        logits = model(x.to(device)).float().cpu().numpy().ravel()
    return float(1.0 / (1.0 + np.exp(-logits.mean())))


def check(args):
    image = sitk.ReadImage(args.input)
    report = {"input": str(Path(args.input).resolve()), "output": args.output,
              "n_slices": image.GetSize()[2], "flipped": False}
    if Path(args.model).exists():
        import torch
        device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
        p = prob_feet_first(load_model(args.model, device), features(image), device)
        report["prob_feet_first"] = p
        report["threshold"] = args.threshold
        report["flipped"] = bool(p > args.threshold)
    else:
        print(f"[orientation] {args.model} not found; the input is used as it is")
    if report["flipped"]:
        arr = sitk.GetArrayFromImage(image)[::-1].copy()
        out = sitk.GetImageFromArray(arr, isVector=image.GetNumberOfComponentsPerPixel() > 1)
        out.CopyInformation(image)        # header kept: only the slice order changes
        print(f"[orientation] feet first (p={report['prob_feet_first']:.3f}): slices reversed")
    else:
        out = image
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, args.output)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


def restore(args):
    import pandas as pd
    report = json.loads(Path(args.report).read_text())
    resultdir = Path(args.resultdir)
    n = report["n_slices"]
    for csv in resultdir.glob("__result_*.csv"):
        df = pd.read_csv(csv, encoding="shift_jis")
        if "{volume_name}" in df:
            df["{volume_name}"] = report["input"]
        if report["flipped"]:
            for col in ("{z_estimated}", "{z_modified}"):
                if col in df:
                    z = pd.to_numeric(df[col], errors="coerce")
                    df[col] = np.where((z >= 0) & (z < n), n - 1 - z, z)
        df.to_csv(csv, encoding="shift_jis", index=False)
    if not report["flipped"]:
        return
    for path in list(resultdir.glob("*.result.nii.gz")) + list(resultdir.glob("*.visceral.nii.gz")):
        im = sitk.ReadImage(str(path))
        out = sitk.GetImageFromArray(sitk.GetArrayFromImage(im)[::-1].copy())
        out.CopyInformation(im)
        sitk.WriteImage(out, str(path), True)
        print(f"[orientation] {path.name}: slices put back in the input's order")


def train(args):
    """Feature list CSV with columns feature, split (train/val/test); all head first."""
    import pandas as pd
    import torch
    import torch.nn.functional as F

    df = pd.read_csv(args.list)
    df = df[df.feature.notna() & (df.feature != "")]
    feats = {s: [np.load(p).astype(np.float32) for p in df.feature[df.split == s]]
             for s in ("train", "val", "test")}
    print({s: len(v) for s, v in feats.items()})
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    model = build_model().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    min_rows = int(round(args.min_mm / ROW_MM))

    def sample(feat, flip, length=None):
        """Random craniocaudal crop; optionally a narrower field of view (columns
        outside a random band set to air, as in a smaller reconstruction) and the
        crop placed at a random row of the window, the rest filled as empty
        (validity 0), so the model does not rely on the scan starting at row 0."""
        n = feat.shape[1]
        length = min(n, length or int(rng.integers(min_rows, WINDOW_ROWS + 1)))
        start = int(rng.integers(0, n - length + 1))
        part = feat[:, start:start + length].copy()
        if flip:
            part = part[:, ::-1]
        if args.fov_crop_prob and rng.random() < args.fov_crop_prob:
            keep = int(rng.integers(int(COLS * args.fov_min), COLS + 1))
            c0 = int(rng.integers(0, COLS - keep + 1))
            part[:, :, :c0] = 0.0
            part[:, :, c0 + keep:] = 0.0
        offset = int(rng.integers(0, WINDOW_ROWS - length + 1)) if args.z_fill else 0
        x = np.zeros((5, WINDOW_ROWS, COLS), np.float32)
        x[:4, offset:offset + length] = part
        x[4, offset:offset + length] = 1.0
        if rng.random() < 0.5:
            x = x[:, :, ::-1]                       # left-right: irrelevant here
        return x.copy()

    def evaluate(split, length):
        model.eval()
        r = np.random.default_rng(1)
        correct = []
        with torch.no_grad():
            for feat in feats[split]:
                for flip in (0, 1):
                    n = feat.shape[1]
                    L = min(n, length)
                    s = int(r.integers(0, n - L + 1))
                    part = feat[:, s:s + L]
                    if flip:
                        part = part[:, ::-1]
                    p = prob_feet_first(model, np.ascontiguousarray(part), args.device)
                    correct.append((p > 0.5) == bool(flip))
        model.train()
        return float(np.mean(correct))

    best, best_state = -1.0, None
    for epoch in range(args.epochs):
        order = rng.permutation(len(feats["train"]))
        total = 0.0
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            flips = rng.integers(0, 2, len(idx))
            x = torch.from_numpy(np.stack([sample(feats["train"][j], f) for j, f in zip(idx, flips)]))
            y = torch.from_numpy(flips.astype(np.float32))
            loss = F.binary_cross_entropy_with_logits(model(x.to(args.device)).view(-1),
                                                      y.to(args.device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        sched.step()
        acc = evaluate("val", int(round(200 / ROW_MM)))
        mark = ""
        if acc >= best:
            best, mark = acc, " *"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d}  loss {total / len(order):.4f}  val acc (200 mm windows) {acc:.4f}{mark}",
              flush=True)
    model.load_state_dict(best_state)
    report = {}
    for mm in (100, 150, 200, 300, 2000):
        report[f"test_acc_{mm}mm"] = evaluate("test", int(round(mm / ROW_MM)))
        print(f"test accuracy, {mm} mm ranges: {report[f'test_acc_{mm}mm']:.4f}")
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": {"row_mm": ROW_MM, "col_mm": COL_MM, "cols": COLS,
                           "window_rows": WINDOW_ROWS, "hu": list(HU)},
                "metrics": report}, args.out)
    print(f"wrote {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--input", required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--report", required=True)
    c.add_argument("--model", default="model/orientation.pt")
    c.add_argument("--device", default="cuda")
    c.add_argument("--threshold", type=float, default=0.7,
                   help="reverse only above this probability; most inputs are head "
                        "first, and reversing a correct one is the costlier mistake")
    r = sub.add_parser("restore")
    r.add_argument("--report", required=True)
    r.add_argument("--resultdir", required=True)
    t = sub.add_parser("train")
    t.add_argument("--list", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--min-mm", type=float, default=100.0)
    t.add_argument("--fov-crop-prob", type=float, default=0.0,
                   help="probability of a narrower field of view (augmentation)")
    t.add_argument("--fov-min", type=float, default=0.6,
                   help="narrowest field of view kept, as a fraction of the 512 mm columns")
    t.add_argument("--z-fill", action="store_true",
                   help="place each crop at a random row of the window, the rest empty")
    t.add_argument("--device", default="cuda")
    t.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    {"check": check, "restore": restore, "train": train}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
