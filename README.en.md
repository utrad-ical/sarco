# Sarcopenia analysis tool (sarco)

[日本語](README.md)

This repository contains a tool for sarcopenia analysis of trunk CT images. It is designed to run as a CAD plug-in of [CIRCUS](https://circus-project.net/).

## Features

*   **Automatic skeletal muscle segmentation:** a U-Net segments skeletal muscle (and the vertebral body) on the L1 and L3 slices.
*   **Quantitative analysis:** the following sarcopenia-related measures are computed.
    *   Skeletal muscle area (SMA)
    *   Mean skeletal muscle density (MSMD)
    *   Visceral adipose tissue area (VAT)
    *   Subcutaneous adipose tissue area (SAT)
*   **Estimates:** age, contrast phase, and body height and weight are estimated, and the attenuation measures are converted to their non-contrast (phase 0) equivalents (using [ct25d-regressor](https://github.com/hanaokalog/ct25d-regressor)).
*   **Automatic orientation correction:** input whose craniocaudal order is reversed (feet first) is detected from the image and analysed in the correct order.
*   **CIRCUS integration:** runs as a CIRCUS CAD plug-in and displays the results there.

## How it works

`processsingle.sh` runs the following steps.

1.  **Orientation:** `orientation.py check` decides the craniocaudal orientation of the input; if the feet come first, it writes a working copy with the slices reordered (all later steps use this copy).
2.  **L1/L3 localisation:** `lumbar.py` estimates the levels from projection images, and `lumbarmodifier.py --mode passthrough` applies the scan-edge rule and writes `l1mod.txt`/`l3mod.txt` (see "L1/L3 localisation and scan range" below).
3.  **Segmentation:** `sarcosegmentation.py` segments muscle and vertebral body on the L1/L3 slices with the pretrained U-Net models.
4.  **Calculation:** `sarcocalculate.py` computes the sarcopenia measures and the visceral region mask from the segmentation and the CT image.
5.  **Estimation:** `estimate.py` estimates age, contrast phase, height and weight, and the phase-0 equivalent values.
6.  **Restore the orientation:** `orientation.py restore` puts the result labels and slice numbers back into the original order of the input.
7.  **Summary:** `resultuniter_circus.py` collects the result tables into `results.json`.

The computed measures and images are shown in the CIRCUS interface.

## Usage

The tool is meant to be run as a CIRCUS plug-in. The `Dockerfile` in this repository builds a Docker image with the dependencies and the analysis scripts.

When the plug-in is started from CIRCUS, `processsingle.sh` analyses a single case.

## Scripts

*   `bin/orientation.py`: decides the craniocaudal orientation of the input (`check`), restores the order of the results (`restore`), trains the classifier (`train`).
*   `bin/lumbar.py`, `bin/lumbarmodifier.py`: estimate the L1/L3 levels (the classifier of `lumbarmodifier.py` is not used for inference).
*   `bin/sarcosegmentation.py`: muscle segmentation with the U-Net models.
*   `bin/sarcocalculate.py`: sarcopenia measures from the segmentation.
*   `bin/estimate.py`: age, contrast phase, height and weight, and phase-0 equivalent values.
*   `bin/resultuniter_circus.py`: collects the result tables into `results.json`.
*   `bin/tools/make_visceral_masks.py`: rebuilds the visceral region masks from existing results.
*   `processsingle.sh`: main script that runs the analysis of a single case.
*   `Dockerfile`: builds the Docker image of the plug-in.
*   `plugin.json`: CIRCUS plug-in settings and display definitions.

## Training data handling

In training (`--mode train`), cases whose `--labelused` label has no voxel at all are excluded automatically, and the label files of the excluded cases are printed (`readfiles.drop_empty_labels`). Without this, lumbarmodifier and sarcosegmentation would learn slice 0 with an empty label as the answer. Nothing is excluded at inference.

## Visceral region mask

Next to the result label `<name>_<L1|L3>.result.nii.gz`, `sarcosegmentation.py` writes the visceral region mask `<name>_<L1|L3>.visceral.nii.gz` and records its file name in the `{<L1|L3>_visceral_path}` column of `__result_<L1|L3>.csv` (`sarcocalculate.visceral_mask`).

*   The mask is the region enclosed by muscle (label 1) (`get_visceral_region`) minus muscle and vertebral body (labels 2, 3). Only the one slice with the most muscle is set to 1. The vertebral arch and the spinal canal are not part of the vertebral body label, so they remain in the mask.
*   The image geometry (origin, spacing, direction) is that of the original volume. `tricolor<i>.nii.gz` also carries the geometry.
*   To rebuild masks from existing results, use `bin/tools/make_visceral_masks.py`; it needs neither a GPU nor a new segmentation. Given a CSV with the columns `gray`, `label`, `name`, it writes the masks, check images (CT, muscle outline in red, vertebral body in blue, visceral region in green) and `manifest.csv` with check values per case. Check cases with a low `muscle_coverage` (share of directions from the muscle centre in which muscle is found): their muscle ring is broken.

## Age, contrast phase, height and weight, and phase-0 (non-contrast) equivalent values

After the L1/L3 analysis, `processsingle.sh` runs `bin/estimate.py`. The results are written to `__estimates.csv` and to `results.estimates` in `results.json`. The estimates use [ct25d-regressor](https://github.com/hanaokalog/ct25d-regressor) (installed at a fixed commit in the Dockerfile). Estimates whose checkpoint is missing from `model/` are skipped; the rest still run.

*   **Age** (`age`): four models on the vertebral medulla (label 3) and on muscle (label 1) at L1/L3 (`model/age_{marrow,muscle}_L{1,3}.pt`) are combined by inverse-variance weighting. The width of the combined interval is corrected with the coefficients in `model/age_combination.json` (the four models are not independent). Each model's value and 95% interval are also given as `age_marrow_L1`, `age_marrow_L3`, `age_muscle_L1`, `age_muscle_L3` (with `_sigma`, `_lo95`, `_hi95`) and shown in CIRCUS.
*   **Contrast phase** (`phase`, `phase_name`): estimated at L1/L3 from the visceral region mask (`model/phase_L{1,3}.pt`); the probabilities of the available levels are averaged and the most probable phase is chosen. Since v8.4.0 there are six phases: 0 non-contrast, 1 early arterial, 2 late arterial, 3 portal venous, 4 venous, 5 equilibrium (up to v8.3 five phases, with 3 portal venous and 4 equilibrium). The phase names follow the number of classes of the model (`estimate.PHASE_SCHEMES`). If the number of `phases` in `phase_correction.json` differs from the number of classes of the model, no phase-0 equivalent values are produced. **If only one of L1 and L3 is in the scan range** (e.g. upper-abdominal dynamic CT), the phase is estimated from that level alone (`phase_n_levels` is 1). If neither is in range, no phase is estimated.
*   **Height and weight** (`height`, `weight`): estimated from the projection of the trunk (arms and table removed) and the L1/L3 slices (`model/bodysize.pt`, `ct25d.bodysize`). If only one level is in the scan range, only that level is used (`body_n_levels` is 1; training included examples with one level dropped).
*   **Phase-0 equivalent values** (`<L1|L3>_<measure>_phase0`): MSMD, meanVAT, meanSAT, cortical_density, medullary_density and SASHI are converted to their non-contrast equivalents with a linear equation for the estimated phase, `value0 = a + b × value_k` (`model/phase_correction.json`; per level, measure and phase; phase 0 is left as it is). The equations were fitted by least squares on pairs of phase 0 and phase k from the same examination (phases 1–3 and 5: Dataset L; phases 1, 2 and 4: Dataset O; phases 4 and 5: Dataset P). SASHI is a proportion, so 0 counts as a value, and the converted value is clipped to 0–1 (since v8.4.0). Since v8.6.0 the areas SMA, VAT and SAT (mm²) are converted as well, with a power equation `value0 = exp(a) × value_k^b` (through the origin, always positive; fitted on the same pairs by weighted non-linear least squares with the variance proportional to the area; `forms` in `phase_correction.json`), because VAT appears about one fifth smaller on contrast-enhanced series (median over pairs from one examination, phases 2–5). On the test-split pairs the median error for L3 VAT is 4–11% (11–32% without conversion), but in patients with little VAT the individual converted values are uncertain. With a `phase_correction.json` without area equations (up to v8.5), areas are not converted.

Training and evaluation (2026-10-02, held-out test split):

| Estimate | Test performance |
|---|---|
| Age, marrow L1 / L3 | MAE 7.2 / 7.7 years |
| Age, muscle L1 / L3 | MAE 7.9 / 7.5 years |
| Age, combined | MAE 6.9 years (Dataset L), 6.2 years (Dataset main); coverage of the 95% interval 0.98 / 0.97 |
| Contrast phase (6 phases, v8.4.0) L1 / L3 / average | Accuracy 0.85 / 0.81 / 0.88 (all datasets, 409 series). Dataset L only: average 0.93 (the 5-phase model of v8.3 on the same series: 0.92). Recall of the venous phase is low (0.31); most venous series are estimated as portal venous or equilibrium |
| Height / weight | MAE 3.5 cm / 2.7 kg (Dataset main). L1 only 4.2 cm / 3.4 kg, L3 only 4.2 cm / 3.2 kg |

The training data are Dataset L (1675 series, 321 patients, with contrast phases) and Dataset main (2039 patients, labels checked by radiologists, with height and weight). For the contrast phase (v8.4.0), Dataset P (pancreatobiliary dynamic CT: non-contrast, venous and equilibrium phases), Dataset HC (whole-body non-contrast CT; at most 100 patients used for training) and Dataset O (phases taken from the series descriptions) were added (see "Training data" below). The split is by patient, train/val/test = 70/15/15 (random seed 0). Series in which L3 lies cranial to L1 (swapped L1/L3 labels, volumes joined from several acquisitions, etc.; 16 series) were left out of training.

## L1/L3 localisation and scan range

In some examinations, such as upper-abdominal dynamic CT, L1 or L3 is outside the scan range.

*   The `lumbar.py` model is trained on images cut at random craniocaudal ranges (`--online`; 120 mm to the whole volume). If the peak likelihood is below the threshold in `model/l<1|3>_lumbar.ckpt.json`, the level is reported as out of range (z = -1).
*   `lumbarmodifier.py --mode passthrough` takes the z of `lumbar.py` as it is and reports the level as out of range if it lies within 10 mm (`--edge_mm`) of the first or last slice. When a level is just outside the scan range, the neighbouring vertebra tends to be found near the edge.
*   The former `lumbarmodifier.py` classifier (refinement with a DenseNet) is not used. On the test split it moved a correct z by one slice in 11–17% of series, which lowered accuracy. Training (`--mode train`) and `--mode test` are kept.

Whole pipeline (`lumbar.py` → `lumbarmodifier.py`) on images cut from the test split (about 300 series, 5 mm slices):

| | L1 | L3 |
|---|---|---|
| In range: found | 0.99–1.00 | 1.00 |
| In range: same slice / within ±1 | 0.96–0.97 / 0.97–0.98 | 0.98–1.00 / 0.99–1.00 |
| Out of range (5–30 mm beyond the edge): falsely found | 0.06 | 0.06 |
| Out of range (30–100 mm beyond the edge): falsely found | 0.00 | 0.00 |

The former models (up to v8.2) always found a level somewhere even when it was out of range, and took another vertebra for L3 in 24–46% of the images with L3 out of range. In range, the same slice was chosen in 0.89 (L1) and 0.83 (L3).

The `lumbar.py` models of v8.5.0 were trained further from the v8.4 models, adding to the training data above whole-body non-contrast CT (Dataset HC) series in which sarco's result had been visually checked and found correct (training L1 297 / L3 318 series; feet-first volumes are read head-first via `{flip}`). On the test split above they perform almost as v8.4 (L1 found with the whole volume as input: 0.997 → 1.000). Compared with v8.4 on 2544 series not used for training (test splits of Datasets L, P, O and HC):

| | L1 | L3 |
|---|---|---|
| Series with the level found (v8.4 → v8.5) | 2282 → 2327 | 2123 → 2197 |
| Series found by both: chosen slices within 5 mm | 2269 / 2281 | 2107 / 2121 |
| Disagreements (found by one only, or more than 5 mm apart) checked visually: correct v8.4 / v8.5 | 6 / 48 (of 59) | 5 / 82 (of 92) |

Of the series found only by v8.5 (L1 46, L3 76), 90% (L1 41, L3 72) were at the correct position. New errors of v8.5, such as levels falsely found out of range, were L1 5 and L3 4. In Dataset HC series that v8.4 had reported as out of range (not used for training), L1 is now found correctly in 24 of 32 series and L3 in 4 of 8.

## Input orientation

The whole pipeline (from `lumbar.py` on) works in array order and assumes that slice 0 is cranial. The z direction in the header cannot be trusted (it is reversed in all of Dataset L), so the orientation is decided from the image (`bin/orientation.py`).

*   Classifier: a small CNN on coronal and sagittal mean and maximum intensity projections (5 mm × 4 mm grid, in array order) (`model/orientation.pt`). The training data are all series of Datasets L and main plus whole-body scans (Dataset HC) and Dataset P (since v8.3.1; the classifier up to v8.3.0 got the orientation of many whole-body scans wrong). Training used random craniocaudal flips, crops of 100–640 mm placed at a random position of the input window (the rest left empty), and, in half of the samples, a narrowed field of view (so that short upper-abdominal scans can be classified). Errors on the test split (threshold 0.7, whole volume): correctly oriented input flipped 0; feet-first input missed Dataset L 0 / main 0 / P 0 / HC 1/71.
*   The input is flipped only when the probability of feet first exceeds 0.7 (`--threshold`; flipping a correctly oriented input is the worse error).
*   When the input is flipped, the result labels (`*.result.nii.gz`, `*.visceral.nii.gz`) and the slice numbers in the tables are put back into the original order of the input. The decision is kept in `__orientation.json` and in `input_feet_first`, `input_feet_first_prob` of `results.estimates`.
*   Without the classifier (no `model/orientation.pt`) the input is used as it is.
*   Height and weight are estimated in physical coordinates, so the coordinate system is taken from the array order after orientation (not from the z direction of the header).

Error rates on the test split (567 series, each as it is and flipped, cropped to the scan ranges below):

| Scan range | Correct orientation flipped | Reversed orientation missed |
|---|---|---|
| 150 mm | 1.4% | 0.9% |
| 200 mm | 0.2% | 0.5% |
| 300 mm | 0.2% | 0.4% |
| Whole volume | 0% | 0% |

(Threshold 0.7. Of all 3698 series at the whole range, most of the errors were volumes without a defined orientation, such as volumes joined from several acquisitions.)

## Tests

```bash
pip install pytest   # ct25d-regressor, torch, SimpleITK, pandas, scipy are also needed
python -m pytest
```

The tests use synthetic phantoms (a trunk with a muscle ring and a vertebral body) and small untrained ct25d checkpoints, so they need no real data, no trained models and no GPU. They check:

*   `sarcocalculate`: SMA, MSMD, VAT and bone values match the known areas and attenuation of the phantom; empty results give 0.
*   Visceral region mask: inside of the ring without muscle and vertebral body, the slice with the most muscle, kept geometry, output of `make_visceral_masks.py` and the appended manifest.
*   Orientation: features follow the array order, not the header; feet-first input is flipped and the results are restored (labels, slice numbers, paths); no flip below the threshold; input used as it is without the classifier; training runs.
*   `estimate.py`: the age combination (including the width correction), contrast phase as the maximum of the averaged probabilities, L1 only, L3 only and neither, the phase-0 equations, skipping only the estimates whose checkpoint is missing.
*   `resultuniter_circus.py`: every value shown by `plugin.json` can be read from `results.json`; missing or empty estimates can still be summarised.

L1/L3 localisation and segmentation (which need the trained models) are checked by running real cases in the Docker image.

## Preprocessing cache

Reading volumes, resampling and arm removal take time and are repeated often (the same volume is processed for L1 and L3, the same cases are preprocessed in every training run, ...). With `--cache_dir`, `lumbar.py`, `lumbarmodifier.py` and `sarcosegmentation.py` therefore store the preprocessing result per case and reuse it (`bin/prepcache.py`).

*   Cached: the lumbar projection images (for the whole volume; augmentation shifts are reproduced by cropping rows), the modifier search range, the segmentation target slice, the trunk mask of the calculation step.
*   The cache key is the path, size and modification time of the input file (for `.mhd` also of the data file) and the preprocessing settings. If any of them changes, the entry is rebuilt, so stale results are never used (old files just stay).
*   Results are bit-identical to those without the cache.
*   The shell scripts use `$TEMPDIR/cache` for inference and `./train_temp/cache` for training. The environment variable `SARCO_CACHE_DIR` changes the location. In Docker, `/temp` is cleared after every run; to reuse the cache across runs, mount a persistent volume.
*   If you change preprocessing code so that its output changes, increase the corresponding number in `CACHE_VERSIONS` in `bin/prepcache.py`.

## Building the Docker image

Run the following in the repository root.

```bash
docker build -t sarcopenia-plugin .
```

ct25d-regressor, used for the estimates, is installed at the commit fixed by `CT25D_COMMIT` in the `Dockerfile` (`--no-deps`, keeping the torch of the image). Update this value when ct25d is updated.

## Training data

All data are CT examinations of The University of Tokyo Hospital, used with the approval of the institutional review board (single centre).

| Name | Contents |
|---|---|
| Dataset main | Trunk CT (2039 patients; the cohort of earlier prognosis studies). L1/L3 muscle and vertebral body labels checked and corrected by radiologists; with height and weight |
| Dataset L | Abdominal dynamic CT whose contrast phases are fixed by the protocol (321 patients, 1675 series; five phases including non-contrast) |
| Dataset P | Pancreatobiliary dynamic CT (non-contrast, venous and equilibrium phases) |
| Dataset HC | Whole-body non-contrast CT of health-screening PET/CT |
| Dataset O | Other CT (contrast phases taken from the DICOM series descriptions and protocol names) |

## Intended use and limitations

*   **For research use only.** This is not a medical device for diagnosis or treatment decisions and has no regulatory approval. Check the results before using them.
*   Intended for adult trunk CT that includes the abdomen. Children are not in the training data.
*   Trained on data from a single centre. Performance may differ at other centres, scanners and acquisition settings.
*   The level may be wrong when L1 or L3 is close to the edge of the scan range, or when the number of vertebrae varies (e.g. transitional vertebrae).
*   The venous phase is estimated poorly; most venous series are estimated as portal venous or equilibrium.
*   Age estimates regress towards the mean (young people come out older, old people younger).

## Citation

The reference of the paper will be added here after acceptance. To cite the software, use the Zenodo DOI of the release you used (`CITATION.cff`).

## Licence

The source code and the model weights (`model/`) are both licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). Commercial use is not permitted. See `LICENSE`.

[ct25d-regressor](https://github.com/hanaokalog/ct25d-regressor), used for the estimates, is a separate repository under its own licence.
