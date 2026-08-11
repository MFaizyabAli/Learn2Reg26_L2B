# Learn2Breath (Learn2Reg 2026) — Docker Submission Instructions & Example

This folder is a **ready-to-run template** for containerizing a Learn2Breath registration
method with Docker. It packages a working baseline — **ANTs Affine + ConvexAdam-MIND
(SVF)** — that you can build, run, and adapt into your own submission.

> The test set (COPDGene lung CT pairs) will **not** be shared with participants. Instead
> you submit a Docker container; the organizers run it internally on the hidden test set.
> Your code is used only for evaluation and is never shared.

## 0. What the container must do (interface contract)

Your container registers **one INSP/EXP CT pair per run**. The organizers invoke it once
per test subject with **three positional arguments** (absolute paths inside the container):

```
<your_image>  <fixed_insp>  <moving_exp>  <output_disp>
```

| # | Argument      | Meaning                                                     |
|---|---------------|--------------------------------------------------------------|
| 1 | `fixed_insp`  | Fixed (inspiration) CT                                       |
| 2 | `moving_exp`  | Moving (expiration) CT                                       |
| 3 | `output_disp` | Where to write the displacement field (path is chosen by us) |

- Registration direction is fixed: **register EXP (moving) to INSP (fixed)**.
- Input images are mounted under `/app/input` (read-only) and the output directory under
  `/app/output`. **Write the displacement field to exactly the 3rd-argument path** — do not
  invent your own filename; the organizers pick it.
- The container must **not** assume any dataset layout, filenames, or the number of pairs.
  It receives explicit paths and processes exactly that one pair.

### Output format

- A single NIfTI displacement field, **channel-first `(3, X, Y, Z)`**, `float32`.
- **Original / full input resolution** — the same spatial grid as the fixed image
  (`224 × 192 × 224` at `1.5 mm` isotropic spacing — this is the shared NLST/COPDGene
  target grid; see `COPDGene_PROCESSING.md` for how the evaluation data was normalized to
  it), not a downsampled grid.
- Values are **displacements in voxels** on the fixed-image grid that map the **moving
  (EXP)** image onto the **fixed (INSP)** image (i.e. warping `moving` by the field aligns
  it to `fixed`).

Both training (NLST) and evaluation (COPDGene-derived) images share this exact geometry —
shape `(224, 192, 224)`, spacing `(1.5, 1.5, 1.5)` mm, `float32`, unnormalized HU values, the
simplified NLST-style affine `diag(1.5, 1.5, 1.5, 1)`. There is a single modality (CT); no
modality code is needed in filenames.

---

## 1. Files in this template

```
Test_Phase_Docker_Example/
├── Dockerfile                     # builds the image (base: anibali/pytorch 2.0.1 / CUDA 11.8)
├── requirements.txt               # Python dependencies
├── infer_convexadam.py            # the inference entrypoint (one INSP/EXP CT pair -> displacement)
├── MIR/                           # the MIR package (bundled, pip-installed inside the image)
├── build.sh                       # docker build helper
├── export.sh                      # docker save -> .tar.gz helper
├── test.sh                        # iterate the validation set locally (see §5)
├── Learn2Breath_val_dataset.json  # validation manifest test.sh loops over
└── README.md                      # this file
```

You will typically **replace `infer_convexadam.py`** (and `requirements.txt` / `MIR/`) with
your own method, keeping the same 3-argument interface and output format.

---

## 2. The baseline method (what the example does)

`infer_convexadam.py` reproduces the `ants_affine_convexadam_svf` baseline, adapted from the
PSMAReg 2026 baseline for single-modality lung CT:

1. Window-normalize the CT (lung window, `[-1000, 200] HU`) and downsample by a factor of 2.
   Learn2Breath images are already cropped to the lung field of view during preprocessing, so
   — unlike a whole-body PET/CT baseline — **no scanner-bed/table removal step is needed**.
2. **ANTs affine** registration on the low-resolution CT.
3. **ConvexAdam-MIND** deformable registration (SVF parameterization) on the affine-warped
   low-resolution CT.
4. **Compose** the deformable field *after* the affine field into a single transform.
5. **Up-sample** the composed field back to the **original resolution** and save it.

It is optimization-based (no learned weights, no GPU training).

> **[TODO — organizers]** Run the baseline end-to-end on `learn2breath_validation/` and
> record an actual per-pair runtime here (the PSMAReg analogue ran ~15 s/pair on GPU; lung
> CT at `224×192×224` with larger breathing-motion displacement may run slower or need
> retuned ConvexAdam hyperparameters — `CONVEX_NITER`, `CONVEX_GRID_SP`, `CONVEX_LAMBDA` in
> `infer_convexadam.py`).

---

## 3. Build your Docker image

Install [Docker](https://docs.docker.com/get-docker/) first. Then, from this directory:

```bash
docker build -f Dockerfile -t learn2breath_convexadam .
# or:
bash build.sh
```

Key points if you build your own:

- Use a **GPU-based PyTorch** base image (recommended for reasonable runtime).
- Use **`ENTRYPOINT`** (not `CMD`) so the three positional path arguments are forwarded to
  your script. Test it: `docker run --rm your_image --help` should reach your argument
  parser.
- If your method needs learned weights, `COPY` them into the image (they are not downloaded
  at runtime — evaluation containers run **offline**, see §4).

---

## 4. How the organizers run your container

Once per test subject, using a command of this form (paths are examples):

```bash
docker run --rm \
    --ipc=host \
    --memory 60g \
    --gpus "device=0" \
    --user $(id -u):$(id -g) \
    --network=none \
    --mount type=bind,source=[test image dir],target=/app/input,readonly \
    --mount type=bind,source=[predictions dir],target=/app/output \
    [your image name] \
        /app/input/SUBJECTID_INSP.nii.gz \
        /app/input/SUBJECTID_EXP.nii.gz \
        /app/output/disp_SUBJECTID.nii.gz
```

Notes:

- `--network=none`: containers run **offline**. Bundle everything you need at build time.
- `--gpus "device=0"`: one GPU is available.
- Your container must exit cleanly and leave the displacement field at the given output
  path.

### Runtime budget

> **[TODO — organizers]** The evaluation cohort is still being finalized — see
> `COPDGene_PROCESSING.md` (as of last processing run: 165 of a 225-subject target
> complete, plus a 35-subject backfill list for FOV-gate failures). Once the final test-set
> size and evaluation hardware are locked in, set a concrete total-runtime budget and
> per-pair average here, in the same form as PSMAReg's:
> *"N pairs, T hours total, ~S seconds/pair average, baseline reference runtime X."*

### Reference evaluation environment

> The evaluation hardware/software below is provided as a reference; confirm the exact
> specification on the official challenge page.

- CPU: Intel(R) Xeon(R) Silver 4410Y
- GPU: NVIDIA H100 (80 GB VRAM)
- RAM: 60 GB
- Driver 545.23.06, CUDA 12.3, Docker 26.1.3, Rocky Linux 8

---

## 5. Test your container on the validation set

The container itself does **one** pair per run, so validation is done by looping over the
validation subjects. `test.sh` does exactly this: it builds the image, then reads
`Learn2Breath_val_dataset.json` and calls the container once per subject.

```bash
bash test.sh
```

Edit the paths at the top of `test.sh` first:

- `DATA_DIR` — the local path to `learn2breath_validation/` (contains `ct_data/`), mounted
  at `/app/input`.
- `DATASET_JSON` — `Learn2Breath_val_dataset.json` (the validation manifest).
- `OUTPUT_DIR` — where predictions are written (mounted at `/app/output`).

`Learn2Breath_val_dataset.json` lists a held-out subset of the NLST training subjects — since
the COPDGene hidden test set is never shared with participants, this is what you use for
local validation before submitting. Each entry defines one subject:

```json
{
  "subject": "NLST_0001",
  "Fixed INSP":  "./ct_data/NLST_0001_INSP.nii.gz",
  "Moving EXP":  "./ct_data/NLST_0001_EXP.nii.gz"
}
```

`test.sh` maps each entry to the container arguments as:

```
fixed  = Fixed INSP  ->  ./ct_data/<subject>_INSP.nii.gz
moving = Moving EXP  ->  ./ct_data/<subject>_EXP.nii.gz
output =  disp_<subject>.nii.gz
```

**Only `ct_data/` is mounted into the container.** The `seg_net/` lobe and fissure masks
that ship alongside `ct_data/` in `learn2breath_validation/` are organizer-side scoring
material — they are never exposed to the container, matching how lung/lobe/fissure masks are
kept out of the COPDGene evaluation data (see `COPDGene_PROCESSING.md`).

After it finishes, `OUTPUT_DIR` contains one `disp_<subject>.nii.gz` per validation subject —
these are your **validation predictions** for submission.

---

## 6. Export your image

```bash
docker save learn2breath_convexadam | gzip -c > learn2breath_convexadam.tar.gz
# or:
bash export.sh
```

---

## 7. Submission checklist

Package a single `.zip`:

```
Learn2Breath_[your Grand Challenge username]_TestPhase.zip
├── [your image name].tar.gz     <-- your exported Docker container (docker save | gzip)
├── README.txt                    <-- requirements to run your model: #CPUs, RAM,
│                                     estimated time/pair, GPU VRAM, and the exact
│                                     `docker run` argument order you expect
└── validation_predictions.zip    <-- your validation displacement fields
    ├── disp_NLST_0001.nii.gz
    ├── disp_NLST_0002.nii.gz
    └── ...
```

Because the organizers reproduce your validation predictions locally, please make the
container **deterministic** where possible and document any nondeterminism in `README.txt`.

Finally, please email a **downloadable link** to your `.zip` file to the organizer at
**mfaizyabali@gmail.com**.

---

## 8. Paper submission (Springer LNCS proceedings)

> **[TODO — organizers]** Confirm whether Learn2Breath follows the same Learn2Reg 2026 paper
> track as PSMAReg (Springer LNCS proceedings, OpenReview peer review, 4-page minimum,
> originality gating) or has its own process. The PSMAReg language is reproduced below as a
> starting point — replace the OpenReview link/venue if Learn2Breath uses a different one.

This year the challenge runs **full paper proceedings**. To be eligible for evaluation and
prizes, submit a full method description prepared with the **Springer LNCS templates** (LaTeX or
Word), available from the
[Springer LNCS author guidelines](https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines).
The paper must be ***at least four pages, excluding references.***

Your paper should give a clear, reproducible description of the method, the computing
infrastructure used (hardware and software), and validation-leaderboard results
(mean ± std per metric).

**Peer review and originality gating.** Papers are submitted through OpenReview and undergo
**peer review**. Submissions are gated for originality: a paper found to have **significant
overlap with existing publications will not be accepted** into the proceedings, and its
authors will be **ineligible to compete for prizes** and to be included in any challenge
summary papers.

> Deadlines, the OpenReview submission link, and any required citations are announced on the
> official Learn2Reg 2026 / Learn2Breath challenge page — follow those for the authoritative
> dates and requirements.

---

## Questions

For questions about the Docker submission, contact the organizers at
**mfaizyabali@gmail.com**.
