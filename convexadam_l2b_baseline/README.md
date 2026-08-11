# Learn2Breath ANTs Affine + ConvexAdam Baseline

Standalone (non-Docker) baseline for the Learn2Reg 2026 Learn2Breath task. It estimates
moving(EXP)-to-fixed(INSP) displacement fields named `disp_<subject>.nii.gz` and packages
them into a Codabench-style submission zip. This is the same registration logic as
`../Test_Phase_Docker_Example/infer_convexadam.py`, batched over a dataset manifest — use this to
generate the organizers' own baseline validation/test predictions, or as a reference when
building your own batch-runner outside Docker.

The method includes:

1. CT window normalization, default `[-1000, 200] -> [0, 1]` (lung window; no bed/table
   removal, since Learn2Breath CTs are already cropped to the lung field of view).
2. ANTsPy affine registration, `moving` (EXP) CT to `fixed` (INSP) CT.
3. Optional ConvexAdam MIND/SVF residual registration on the affine-warped moving CT.
4. Composition into one forward moving-to-fixed voxel-displacement field, upsampled back to
   the full `(224, 192, 224)` grid.

## Official Submission Convention

- File names: `disp_<subject>.nii.gz`, e.g. `disp_NLST_0001.nii.gz`.
- Files live at the root of the submission zip, with no nested directory.
- Field shape: `(3, 224, 192, 224)`, full resolution (not downsampled).
- Channel order: displacement components first, then spatial dimensions.
- Direction: forward moving(EXP)-to-fixed(INSP), so warping EXP with the field aligns it to
  INSP.
- Header spacing is the fixed image spacing, `(1.5, 1.5, 1.5)` mm.

## Requirements

```bash
pip install -r requirements.txt
```

The script also requires the [MIR package](https://github.com/junyuchen245/MIR) (the same
package bundled in `../Test_Phase_Docker_Example/MIR`), either installed in the Python environment or
supplied with:

```bash
--mir-src /path/to/Test_Phase_Docker_Example/MIR/src
```

## Run

```bash
python3 estimate_displacements.py \
  --reference-dir /path/to/learn2breath_validation \
  --output-dir ./outputs/ants_affine_convexadam \
  --mir-src /path/to/Test_Phase_Docker_Example/MIR/src \
  --stage affine-convex \
  --ants-transform Affine \
  --affine-field-transform fwd \
  --downsample-factor 2 \
  --convex-grid-sp 4 \
  --convex-niter 80 \
  --convex-smooth 3 \
  --dtype float32 \
  --overwrite

python3 zip_submission.py \
  --prediction-dir ./outputs/ants_affine_convexadam/predictions \
  --zip-path ./outputs/ants_affine_convexadam/predictions.zip
```

Or run both plus validation in one step:

```bash
bash run_baseline.sh /path/to/learn2breath_validation ./outputs/ants_affine_convexadam /path/to/Test_Phase_Docker_Example/MIR/src
```

Submit `predictions.zip`. Validate before submission:

```bash
python3 validate_submission.py \
  --zip-path ./outputs/ants_affine_convexadam/predictions.zip \
  --dataset-json /path/to/learn2breath_validation/Learn2Breath_dataset.json
```

`--reference-dir` expects a `Learn2Breath_dataset.json` reference manifest (organizer-side,
not included in this repo — see the Codabench scoring program) alongside `ct_data/` (and,
for organizer use,
`seg_net/`).

## Notes

- Use `--affine-field-transform fwd`; this is the calibrated forward moving-to-fixed
  convention for this baseline.
- `--stage affine` runs affine-only. `--stage affine-convex` runs affine plus ConvexAdam.
- By default, no intermediate warped images or dense debug fields are saved. Keep this
  behavior for disk-light challenge runs.
- Add `--save-previews` if you want compact PNG diagnostics.
- Hyperparameters (`--ct-window`, `--convex-*`) reproduce the PSMAReg 2026 baseline recipe,
  adapted to single-modality lung CT — retune if lung breathing motion (much larger than
  longitudinal tumor motion) needs a different `--convex-grid-sp` / `--convex-niter` /
  `--convex-lambda` to converge well.
