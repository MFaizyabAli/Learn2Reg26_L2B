#!/usr/bin/env python3
"""Estimate Learn2Breath displacement fields with ANTs affine + ConvexAdam MIND/SVF.

Batches the same registration used by ../Docker_Example/infer_convexadam.py over every
subject in a Learn2Breath_dataset.json manifest, writing full-resolution
(3, 224, 192, 224) predictions named ``disp_<subject>.nii.gz`` directly under
``<output-dir>/predictions``.
"""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import ants
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom


SpatialTransformer = None
convex_adam_pt_svf = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--dataset-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mir-src", type=Path, default=None, help="Path to MIR/src if MIR is not installed.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--stage", default="affine-convex", choices=["affine", "affine-convex"])
    parser.add_argument("--ct-window", nargs=2, type=float, default=[-1000.0, 200.0])
    parser.add_argument("--ants-transform", default="Affine", choices=["Rigid", "Affine", "TRSAA"])
    parser.add_argument("--affine-field-transform", default="fwd", choices=["fwd", "inverse"])
    parser.add_argument("--aff-metric", default="mattes")
    parser.add_argument("--aff-sampling", type=int, default=64)
    parser.add_argument("--convex-mind-r", type=int, default=1)
    parser.add_argument("--convex-mind-d", type=int, default=2)
    parser.add_argument("--convex-lambda", type=float, default=2.0)
    parser.add_argument("--convex-grid-sp", type=int, default=4)
    parser.add_argument("--convex-disp-hw", type=int, default=4)
    parser.add_argument("--convex-niter", type=int, default=80)
    parser.add_argument("--convex-smooth", type=int, default=3)
    parser.add_argument("--convex-grid-sp-adam", type=int, default=2)
    parser.add_argument("--svf-steps", type=int, default=7)
    parser.add_argument("--no-ic", action="store_true")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--preview-subjects", nargs="*", default=None)
    return parser.parse_args()


def ensure_mir(mir_src):
    if mir_src is not None:
        sys.path.insert(0, str(mir_src))
    global SpatialTransformer, convex_adam_pt_svf
    from MIR.models.convexAdam.convex_adam_MIND_SVF import convex_adam_pt_svf as convex_fn
    from MIR.models.registration_utils import SpatialTransformer as transformer_cls

    convex_adam_pt_svf = convex_fn
    SpatialTransformer = transformer_cls


def resolve_reference_path(reference_dir, relative_path):
    if relative_path.startswith("./"):
        relative_path = relative_path[2:]
    return (reference_dir / relative_path).resolve()


def ct_window_normalize(volume, window):
    lo, hi = [float(value) for value in window]
    if hi <= lo:
        raise ValueError("Invalid CT window: {}".format(window))
    volume = np.asarray(volume, dtype=np.float32)
    out = (np.clip(volume, lo, hi) - lo) / (hi - lo)
    out[~np.isfinite(volume)] = 0.0
    return out.astype(np.float32)


def downsample_volume(volume, factor, order=1):
    if factor == 1:
        return np.asarray(volume, dtype=np.float32)
    scale = tuple(1.0 / factor for _ in range(3))
    return zoom(volume, zoom=scale, order=order).astype(np.float32)


def make_lowres_ants_image(volume, spacing, factor, ct_window):
    lowres = downsample_volume(ct_window_normalize(volume, ct_window), factor=factor, order=1)
    image = ants.from_numpy(lowres)
    image.set_spacing(tuple(float(s) * factor for s in spacing))
    image.set_origin((0.0, 0.0, 0.0))
    image.set_direction(np.diag([-1.0, -1.0, 1.0]))
    return image


def ants_physical_delta_to_fullres_voxel_disp(disp, direction, fullres_spacing):
    spacing = np.asarray(fullres_spacing, dtype=np.float32)
    flat = disp.reshape(-1, 3)
    index_delta = flat.dot(direction) / spacing
    return index_delta.reshape(disp.shape).astype(np.float32)


def ants_affine_to_fullres_voxel_disp(transform, reference_image, fullres_spacing):
    params = np.asarray(transform.parameters, dtype=np.float32)
    fixed_params = np.asarray(transform.fixed_parameters, dtype=np.float32)
    if params.size != 12 or fixed_params.size < 3:
        raise ValueError("Expected 3D affine transform with 12 parameters and center fixed parameters.")
    matrix = params[:9].reshape(3, 3)
    translation = params[9:12]
    center = fixed_params[:3]
    shape = tuple(int(v) for v in reference_image.shape)
    spacing_lr = np.asarray(reference_image.spacing, dtype=np.float32)
    direction = np.asarray(reference_image.direction, dtype=np.float32)
    origin = np.asarray(reference_image.origin, dtype=np.float32)
    grids = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    index = np.stack(grids, axis=-1).reshape(-1, 3)
    physical = origin + (index * spacing_lr).dot(direction.T)
    moved = (physical - center).dot(matrix.T) + center + translation
    disp = (moved - physical).reshape(shape + (3,)).astype(np.float32)
    return ants_physical_delta_to_fullres_voxel_disp(disp, direction, fullres_spacing)


def channel_last_to_channel_first(field):
    if field.shape[-1] == 3:
        return np.moveaxis(field, -1, 0)
    if field.shape[0] == 3:
        return field
    raise ValueError("Expected displacement field with 3 channels.")


def compose_deform_after_affine(deform_fullres_vox, affine_fullres_vox):
    deform = torch.from_numpy(channel_last_to_channel_first(deform_fullres_vox)).unsqueeze(0).float()
    affine = torch.from_numpy(channel_last_to_channel_first(affine_fullres_vox)).unsqueeze(0).float()
    transformer = SpatialTransformer(tuple(deform.shape[2:]), mode="bilinear")
    with torch.no_grad():
        sampled_affine = transformer(affine, deform, padding_mode="border")
        total = deform + sampled_affine
    return total.squeeze(0).numpy().astype(np.float32)


def upsample_field_to(field_ch_first, target_shape, factor):
    field = torch.from_numpy(np.ascontiguousarray(field_ch_first)).unsqueeze(0).float()
    with torch.no_grad():
        up = F.interpolate(field, size=tuple(int(s) for s in target_shape), mode="trilinear", align_corners=False)
    return (up * float(factor)).squeeze(0).numpy().astype(np.float32)


def save_disp(path, field_ch_first, affine, spacing):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(field_ch_first.astype(np.float32), affine)
    image.header["pixdim"][1:4] = [float(s) for s in spacing]
    nib.save(image, str(path))


def warp_volume(volume, field_ch_first, mode="bilinear"):
    field = torch.from_numpy(np.ascontiguousarray(field_ch_first)).unsqueeze(0).float()
    src = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    transformer = SpatialTransformer(tuple(field.shape[2:]), mode=mode)
    with torch.no_grad():
        warped = transformer(src, field, padding_mode="border").squeeze().numpy()
    return warped.astype(np.float32)


def mid_slices(volume):
    x, y, z = [int(dim) // 2 for dim in volume.shape[:3]]
    return [
        ("sag", np.rot90(volume[x, :, :])),
        ("cor", np.rot90(volume[:, y, :])),
        ("axi", np.rot90(volume[:, :, z])),
    ]


def save_preview_png(path, fixed, moving, final_warped, ct_window, title):
    rows = [
        ("fixed", ct_window_normalize(fixed, ct_window)),
        ("moving", ct_window_normalize(moving, ct_window)),
        ("warped", ct_window_normalize(final_warped, ct_window)),
    ]
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 2.4 * len(rows)), squeeze=False)
    fig.suptitle(title, fontsize=10)
    for r, (row_name, volume) in enumerate(rows):
        for c, (view_name, image) in enumerate(mid_slices(volume)):
            axes[r, c].imshow(image, cmap="gray", origin="lower", vmin=0.0, vmax=1.0)
            axes[r, c].set_title("{} {}".format(row_name, view_name), fontsize=8)
            axes[r, c].axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(path), dpi=160, bbox_inches="tight")
    plt.close(fig)


def should_preview(subject, args):
    if not args.save_previews:
        return False
    if not args.preview_subjects:
        return True
    return subject in set(args.preview_subjects)


def select_entries(dataset, subjects, limit):
    entries = list(dataset.get("validation", []))
    if subjects:
        subject_set = set(subjects)
        entries = [entry for entry in entries if entry["subject"] in subject_set]
    if limit is not None:
        entries = entries[:limit]
    return entries


def run_case(entry, args, prediction_dir, preview_dir):
    subject = entry["subject"]
    prediction_path = prediction_dir / "disp_{}.nii.gz".format(subject)
    if prediction_path.exists() and args.skip_existing:
        return {"subject": subject, "status": "skipped", "prediction": str(prediction_path)}
    if prediction_path.exists() and not args.overwrite:
        raise FileExistsError("Prediction exists: {}".format(prediction_path))

    t0 = time.time()
    fixed_path = resolve_reference_path(args.reference_dir, entry["Fixed INSP"])
    moving_path = resolve_reference_path(args.reference_dir, entry["Moving EXP"])
    fixed_nii = nib.load(str(fixed_path))
    moving_nii = nib.load(str(moving_path))
    fixed = fixed_nii.get_fdata(dtype=np.float32)
    moving = moving_nii.get_fdata(dtype=np.float32)
    fullres_shape = tuple(int(s) for s in fixed.shape[:3])
    fullres_spacing = tuple(float(s) for s in fixed_nii.header.get_zooms()[:3])

    fixed_ants = make_lowres_ants_image(fixed, fullres_spacing, args.downsample_factor, args.ct_window)
    moving_ants = make_lowres_ants_image(moving, fullres_spacing, args.downsample_factor, args.ct_window)

    affine_start = time.time()
    tx = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform=args.ants_transform,
        aff_metric=args.aff_metric,
        aff_sampling=args.aff_sampling,
        verbose=False,
    )
    affine_seconds = time.time() - affine_start

    affine_transform = ants.read_transform(tx["fwdtransforms"][0])
    if args.affine_field_transform == "inverse":
        affine_transform = ants.invert_ants_transform(affine_transform)
    affine_disp_fullres_vox = ants_affine_to_fullres_voxel_disp(
        affine_transform, fixed_ants, fullres_spacing=fullres_spacing,
    )

    fixed_low = fixed_ants.numpy().astype(np.float32)
    warped_moving_low = tx["warpedmovout"].numpy().astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "float16" and device.type == "cuda" else torch.float32

    convex_seconds = 0.0
    deform_fullres_vox = np.zeros_like(affine_disp_fullres_vox, dtype=np.float32)
    if args.stage == "affine-convex":
        convex_start = time.time()
        deform_fwd, _ = convex_adam_pt_svf(
            img_fixed=torch.from_numpy(fixed_low),
            img_moving=torch.from_numpy(warped_moving_low),
            mind_r=args.convex_mind_r,
            mind_d=args.convex_mind_d,
            lambda_weight=args.convex_lambda,
            grid_sp=args.convex_grid_sp,
            disp_hw=args.convex_disp_hw,
            selected_niter=args.convex_niter,
            selected_smooth=args.convex_smooth,
            grid_sp_adam=args.convex_grid_sp_adam,
            ic=not args.no_ic,
            svf_steps=args.svf_steps,
            dtype=dtype,
            device=device,
            verbose=False,
            save_disp=True,
        )
        convex_seconds = time.time() - convex_start
        deform_fullres_vox = np.asarray(deform_fwd, dtype=np.float32) * float(args.downsample_factor)
        total_field_low = compose_deform_after_affine(deform_fullres_vox, affine_disp_fullres_vox)
    else:
        total_field_low = channel_last_to_channel_first(affine_disp_fullres_vox).astype(np.float32)

    total_field = upsample_field_to(total_field_low, fullres_shape, args.downsample_factor)
    save_disp(prediction_path, total_field, fixed_nii.affine, fullres_spacing)

    if should_preview(subject, args):
        final_warped = warp_volume(moving, total_field)
        save_preview_png(
            preview_dir / "{}.png".format(subject),
            fixed=fixed, moving=moving, final_warped=final_warped,
            ct_window=args.ct_window,
            title="{} {} {}".format(subject, args.stage, args.affine_field_transform),
        )

    return {
        "subject": subject,
        "status": "ok",
        "prediction": str(prediction_path),
        "affine_seconds": affine_seconds,
        "convex_seconds": convex_seconds,
        "total_seconds": time.time() - t0,
        "stage": args.stage,
        "field_abs_mean": float(np.mean(np.abs(total_field))),
        "field_abs_max": float(np.max(np.abs(total_field))),
    }


def write_rows(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    ensure_mir(args.mir_src)

    dataset_json = args.dataset_json or args.reference_dir / "Learn2Breath_dataset.json"
    with dataset_json.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    prediction_dir = args.output_dir / "predictions"
    preview_dir = args.output_dir / "previews"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    if args.save_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    entries = select_entries(dataset, args.subjects, args.limit)
    rows = []
    for index, entry in enumerate(entries, start=1):
        try:
            row = run_case(entry, args, prediction_dir, preview_dir)
        except Exception as error:
            row = {"subject": entry.get("subject", ""), "status": "error", "error": repr(error)}
            print("[{}/{}] ERROR {}".format(index, len(entries), json.dumps(row)), flush=True)
            traceback.print_exc()
            raise
        rows.append(row)
        write_rows(args.output_dir / "case_results.csv", rows)
        print("[{}/{}] {}".format(index, len(entries), json.dumps(row)), flush=True)

    print("Done. Predictions: {}".format(prediction_dir))


if __name__ == "__main__":
    main()
