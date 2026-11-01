"""
End-to-end preprocessing pipeline orchestrator for FCD lesion segmentation.

Processes each subject through:
1. Input validation & fingerprinting
2. Resume / overwrite check
3. FLAIR skull stripping (HD-BET)
4. T1 skull stripping (HD-BET)
5. Registration (T1 moving -> FLAIR fixed, rigid SimpleITK)
6. Grid resampling and discrete label preservation
7. Robust brain-masked intensity normalization
8. Writing to three nnU-Net v2 datasets (FLAIR, T1, Fusion)
9. QC visual capture (embedded base64) and intermediate cleanup
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from fcd_segmentation.preprocessing.dataset import (
    create_nnunet_dataset_json,
    validate_nnunet_subject_outputs,
    write_json_atomic,
    write_subject_to_nnunet,
)
from fcd_segmentation.preprocessing.discovery import (
    inventory_cohort,
)
from fcd_segmentation.preprocessing.geometry import (
    same_grid,
)
from fcd_segmentation.preprocessing.normalization import normalize_intensity
from fcd_segmentation.preprocessing.qc_reports import (
    generate_final_dataset_qc_html,
    generate_input_inventory_html,
    generate_registration_html,
    generate_skull_stripping_html,
)
from fcd_segmentation.preprocessing.registration import (
    create_registration_qc_visuals,
    register_to_reference,
)
from fcd_segmentation.preprocessing.resampling import resample_to_reference
from fcd_segmentation.preprocessing.skull_stripping import (
    create_visual_qc,
    run_hd_bet,
)

logger = logging.getLogger("fcd_preprocessing")


def compute_file_fingerprint(path: Path | str) -> str:
    """Fast fingerprint of file based on size and mtime."""
    p = Path(path)
    if not p.is_file():
        return "missing"
    stat = p.stat()
    return f"{stat.st_size}_{stat.st_mtime_ns}"


def compute_subject_fingerprint(t1_path: str, flair_path: str, roi_path: str | None, config_hash: str) -> str:
    """Compute combined fingerprint for subject inputs and pipeline configuration."""
    parts = [
        compute_file_fingerprint(t1_path),
        compute_file_fingerprint(flair_path),
        compute_file_fingerprint(roi_path) if roi_path else "no_roi",
        config_hash,
    ]
    return hashlib.sha256(";".join(parts).encode("utf-8")).hexdigest()[:16]


def process_subject_pipeline(
    subject_record: dict[str, Any],
    config: dict[str, Any],
    config_hash: str,
    output_dirs: dict[str, Path],
    logger_instance: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Process one subject end-to-end.

    Returns comprehensive record of stages, QC, outputs, and status.
    """
    log = logger_instance or logger
    subject_id = subject_record["subject_id"]
    split = subject_record.get("split", "train")
    log.info(f"--- Starting subject: {subject_id} (split: {split}) ---")

    flair_input_path = subject_record["flair_path"]
    t1_input_path = subject_record["t1_path"]
    roi_input_path = subject_record.get("roi_path")

    held_out_target = config.get("splits", {}).get("held_out_target", "imagesTr")
    resume = config.get("execution", {}).get("resume", False)
    overwrite = config.get("execution", {}).get("overwrite", False)
    keep_failed = config.get("execution", {}).get("keep_failed_intermediates", False)

    flair_ds_dir = output_dirs["flair_dataset"]
    t1_ds_dir = output_dirs["t1_dataset"]
    fusion_ds_dir = output_dirs["fusion_dataset"]

    # 1. Check resume condition
    if resume and not overwrite:
        val_check = validate_nnunet_subject_outputs(
            subject_id=subject_id,
            flair_dataset_dir=flair_ds_dir,
            t1_dataset_dir=t1_ds_dir,
            fusion_dataset_dir=fusion_ds_dir,
            split=split,
            held_out_target=held_out_target,
        )
        if val_check["status"] in {"PASS", "WARN"}:
            log.info(f"Subject {subject_id} outputs exist and validate. Skipping due to --resume.")
            val_check["status"] = "SKIPPED"
            val_check["duration_seconds"] = 0.0
            return {
                "subject_id": subject_id,
                "status": "SKIPPED",
                "stages": {"all": "SKIPPED"},
                "dataset_validation": val_check,
                "error": None,
                "warnings": [],
            }

    subject_fingerprint = compute_subject_fingerprint(
        t1_input_path, flair_input_path, roi_input_path, config_hash
    )

    result_record: dict[str, Any] = {
        "subject_id": subject_id,
        "split": split,
        "fingerprint": subject_fingerprint,
        "status": "FAILED",
        "stages": {},
        "durations": {},
        "warnings": [],
        "errors": [],
        "qc_flair_skullstrip": None,
        "qc_t1_skullstrip": None,
        "qc_registration": None,
        "dataset_validation": None,
    }

    start_time = time.time()
    temp_dir_obj = tempfile.TemporaryDirectory(prefix=f"{subject_id}_fcd_work_")
    temp_work_dir = Path(temp_dir_obj.name)
    success = False

    try:
        # Stage 1: Skull-strip FLAIR
        t0 = time.time()
        log.info(f"[{subject_id}] Stage 1: Skull-stripping FLAIR")
        device_opt = config.get("skull_stripping", {}).get("device", "cuda")
        tta_opt = config.get("skull_stripping", {}).get("tta", False)

        flair_stripped_path = temp_work_dir / f"{subject_id}_FLAIR_skullstripped.nii.gz"
        flair_mask_path = temp_work_dir / f"{subject_id}_FLAIR_brain_mask.nii.gz"

        run_hd_bet(
            input_path=flair_input_path,
            output_dir=temp_work_dir,
            subject_id=subject_id,
            device=device_opt,
            tta=tta_opt,
            overwrite=True,
            modality="FLAIR",
            output_image_path=flair_stripped_path,
            output_mask_path=flair_mask_path,
        )
        result_record["stages"]["flair_skull_stripping"] = "SUCCESS"
        result_record["durations"]["flair_skull_stripping"] = round(time.time() - t0, 2)

        # FLAIR skull strip QC
        qc_flair = create_visual_qc(
            original_flair_path=flair_input_path,
            skullstripped_path=flair_stripped_path,
            brain_mask_path=flair_mask_path,
            roi_path=roi_input_path,
            subject_id=subject_id,
            modality="FLAIR",
            return_base64=True,
        )
        result_record["qc_flair_skullstrip"] = qc_flair
        if qc_flair.get("review_flags"):
            result_record["warnings"].append(f"FLAIR skull strip QC flags: {qc_flair['review_flags']}")

        # Stage 2: Skull-strip T1
        t0 = time.time()
        log.info(f"[{subject_id}] Stage 2: Skull-stripping T1")
        t1_stripped_path = temp_work_dir / f"{subject_id}_T1_skullstripped.nii.gz"
        t1_mask_path = temp_work_dir / f"{subject_id}_T1_brain_mask.nii.gz"

        run_hd_bet(
            input_path=t1_input_path,
            output_dir=temp_work_dir,
            subject_id=subject_id,
            device=device_opt,
            tta=tta_opt,
            overwrite=True,
            modality="T1",
            output_image_path=t1_stripped_path,
            output_mask_path=t1_mask_path,
        )
        result_record["stages"]["t1_skull_stripping"] = "SUCCESS"
        result_record["durations"]["t1_skull_stripping"] = round(time.time() - t0, 2)

        # T1 skull strip QC
        qc_t1 = create_visual_qc(
            original_flair_path=t1_input_path,
            skullstripped_path=t1_stripped_path,
            brain_mask_path=t1_mask_path,
            roi_path=None,
            subject_id=subject_id,
            modality="T1",
            return_base64=True,
        )
        result_record["qc_t1_skullstrip"] = qc_t1
        if qc_t1.get("review_flags"):
            result_record["warnings"].append(f"T1 skull strip QC flags: {qc_t1['review_flags']}")

        # Stage 3: Registration T1 -> FLAIR
        t0 = time.time()
        log.info(f"[{subject_id}] Stage 3: Registering T1 (moving) to FLAIR (fixed reference)")
        reg_cfg = config.get("registration", {})
        method_opt = reg_cfg.get("method", "rigid")
        n_iters = reg_cfg.get("number_of_iterations", 200)
        n_bins = reg_cfg.get("number_of_bins", 50)
        sampling_pct = reg_cfg.get("sampling_percentage", 0.20)

        # Estimate transform on full-head images for robust anatomical alignment
        tfm_path = temp_work_dir / f"{subject_id}_t1_to_flair.tfm"
        registered_t1_full_path = temp_work_dir / f"{subject_id}_T1_registered_full.nii.gz"

        reg_res = register_to_reference(
            moving_path=t1_input_path,
            reference_path=flair_input_path,
            output_path=registered_t1_full_path,
            method=method_opt,
            transform_output_path=tfm_path,
            number_of_iterations=n_iters,
            number_of_bins=n_bins,
            sampling_percentage=sampling_pct,
        )
        result_record["stages"]["registration"] = "SUCCESS"
        result_record["durations"]["registration"] = round(time.time() - t0, 2)

        # Apply estimated transform to skull-stripped T1 image
        final_tfm = reg_res["transform"]
        fixed_sitk = sitk.ReadImage(str(flair_input_path), sitk.sitkFloat32)
        moving_stripped_sitk = sitk.ReadImage(str(t1_stripped_path), sitk.sitkFloat32)

        resampled_t1_stripped = sitk.Resample(
            moving_stripped_sitk,
            fixed_sitk,
            final_tfm,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
        registered_t1_stripped_path = temp_work_dir / f"{subject_id}_T1_registered_skullstripped.nii.gz"
        sitk.WriteImage(resampled_t1_stripped, str(registered_t1_stripped_path), True)

        # Registration QC Visuals (ROI shown — both images are now in FLAIR grid space)
        qc_reg = create_registration_qc_visuals(
            fixed_path=flair_stripped_path,
            resampled_moving_path=registered_t1_stripped_path,
            subject_id=subject_id,
            return_base64=True,
            roi_path=roi_input_path,
        )
        qc_reg["final_metric"] = reg_res.get("final_metric")
        qc_reg["stop_condition"] = reg_res.get("stop_condition")
        result_record["qc_registration"] = qc_reg
        if qc_reg.get("flags"):
            result_record["warnings"].append(f"Registration QC flags: {'; '.join(qc_reg['flags'])}")

        # Stage 4: Load, Resample Label onto FLAIR grid, and Normalize Intensities
        t0 = time.time()
        log.info(f"[{subject_id}] Stage 4: Intensity normalization & label preservation")
        flair_nib = nib.load(str(flair_stripped_path))
        t1_nib = nib.load(str(registered_t1_stripped_path))

        # Lesion ROI: ensure exact grid agreement
        if roi_input_path and Path(roi_input_path).is_file():
            roi_raw_nib = nib.load(roi_input_path)
            if same_grid(flair_nib, roi_raw_nib, tolerance_mm=0.01):
                roi_nib = roi_raw_nib
            else:
                log.warning(f"[{subject_id}] Resampling ROI to reference FLAIR grid using nearest-neighbor.")
                roi_nib = resample_to_reference(roi_raw_nib, flair_nib, is_mask=True)
                result_record["warnings"].append("Resampled ROI onto reference grid.")
        else:
            raise FileNotFoundError(f"Missing required ROI file for {subject_id}")

        # Brain mask for normalization: use FLAIR brain mask
        flair_mask_nib = nib.load(str(flair_mask_path))
        brain_mask_arr = np.asarray(flair_mask_nib.dataobj) > 0

        # Normalization
        norm_cfg = config.get("normalization", {})
        clip_p = norm_cfg.get("clip_percentiles", [0.5, 99.5])
        eps_val = float(norm_cfg.get("eps", 1e-8))

        flair_arr = np.asarray(flair_nib.dataobj, dtype=np.float32)
        flair_norm_arr = normalize_intensity(
            flair_arr,
            mask=brain_mask_arr,
            clip_percentiles=clip_p,
            eps=eps_val,
            allow_constant=True,
        )

        t1_arr = np.asarray(t1_nib.dataobj, dtype=np.float32)
        # Apply FLAIR brain mask to registered T1 to ensure identical brain coverage
        t1_arr[~brain_mask_arr] = 0.0
        t1_norm_arr = normalize_intensity(
            t1_arr,
            mask=brain_mask_arr,
            clip_percentiles=clip_p,
            eps=eps_val,
            allow_constant=True,
        )

        # Label: integer uint8 preservation
        roi_arr = np.asarray(roi_nib.dataobj)
        roi_uint8_arr = np.rint(roi_arr).astype(np.uint8)
        # Ensure only 0 and 1
        roi_uint8_arr = np.where(roi_uint8_arr > 0, 1, 0).astype(np.uint8)

        # Create normalized nibabel images
        flair_header = flair_nib.header.copy()
        flair_header.set_data_dtype(np.float32)
        norm_flair_img = nib.Nifti1Image(flair_norm_arr, flair_nib.affine, header=flair_header)

        t1_header = flair_nib.header.copy()
        t1_header.set_data_dtype(np.float32)
        norm_t1_img = nib.Nifti1Image(t1_norm_arr, flair_nib.affine, header=t1_header)

        roi_header = flair_nib.header.copy()
        roi_header.set_data_dtype(np.uint8)
        norm_roi_img = nib.Nifti1Image(roi_uint8_arr, flair_nib.affine, header=roi_header)

        # Stage 5: Write to three nnU-Net v2 datasets
        log.info(f"[{subject_id}] Stage 5: Writing to FLAIR, T1, and Fusion nnU-Net datasets")
        write_subject_to_nnunet(
            subject_id=subject_id,
            flair_image=norm_flair_img,
            t1_image=norm_t1_img,
            roi_image=norm_roi_img,
            split=split,
            flair_dataset_dir=flair_ds_dir,
            t1_dataset_dir=t1_ds_dir,
            fusion_dataset_dir=fusion_ds_dir,
            overwrite=overwrite,
            held_out_target=held_out_target,
        )
        result_record["stages"]["dataset_writing"] = "SUCCESS"
        result_record["durations"]["normalization_and_writing"] = round(time.time() - t0, 2)

        # Stage 6: Validate output files on disk
        val_res = validate_nnunet_subject_outputs(
            subject_id=subject_id,
            flair_dataset_dir=flair_ds_dir,
            t1_dataset_dir=t1_ds_dir,
            fusion_dataset_dir=fusion_ds_dir,
            split=split,
            held_out_target=held_out_target,
        )
        result_record["dataset_validation"] = val_res
        if val_res["status"] == "FAIL":
            raise RuntimeError(f"Output dataset validation failed for {subject_id}: {val_res['errors']}")

        result_record["status"] = val_res["status"]
        success = True
        log.info(f"[{subject_id}] Pipeline completed successfully in {time.time() - start_time:.2f}s!")

    except Exception as exc:
        result_record["status"] = "FAILED"
        result_record["errors"].append(f"{type(exc).__name__}: {exc}")
        log.error(f"[{subject_id}] Pipeline failed: {exc}", exc_info=True)

    finally:
        result_record["duration_seconds"] = round(time.time() - start_time, 2)
        # Safe cleanup of temporary intermediate images
        if not success and keep_failed:
            log.warning(f"[{subject_id}] Preserving intermediate failed files at {temp_work_dir}")
        else:
            try:
                temp_dir_obj.cleanup()
            except Exception as clean_err:
                log.warning(f"Failed to cleanup temp dir {temp_work_dir}: {clean_err}")

    return result_record


def run_pipeline(
    config_path: Path | str,
    raw_data_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    inventory_only: bool = False,
    limit: int | None = None,
    resume: bool = False,
    overwrite: bool = False,
    fail_fast: bool = False,
    keep_failed_intermediates: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Execute the preprocessing pipeline over the cohort according to configuration.
    """
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    # Allow CLI overrides
    if raw_data_dir:
        config.setdefault("paths", {})["raw_data_dir"] = str(raw_data_dir)
    if processed_dir:
        config.setdefault("paths", {})["processed_data_dir"] = str(processed_dir)
    if reports_dir:
        config.setdefault("paths", {})["reports_dir"] = str(reports_dir)
    if device:
        config.setdefault("skull_stripping", {})["device"] = device

    config.setdefault("execution", {})
    if resume:
        config["execution"]["resume"] = True
        config["execution"]["overwrite"] = overwrite
    elif overwrite:
        config["execution"]["overwrite"] = True
    if fail_fast:
        config["execution"]["fail_fast"] = True
    if keep_failed_intermediates:
        config["execution"]["keep_failed_intermediates"] = True
    if limit is not None:
        config["execution"]["limit"] = limit

    # Resolve paths
    raw_root = Path(config["paths"]["raw_data_dir"]).resolve()
    processed_root = Path(config["paths"]["processed_data_dir"]).resolve()
    reports_root = Path(config["paths"]["reports_dir"]).resolve()
    participants_tsv = config.get("paths", {}).get("participants_tsv")
    if participants_tsv:
        participants_tsv = Path(participants_tsv).resolve()

    # If user provided a path ending in 'qc', treat that as qc_dir directly
    if reports_root.name == "qc":
        qc_dir = reports_root
        manifests_dir = reports_root.parent / "manifests"
        logs_dir = reports_root.parent / "logs"
    else:
        qc_dir = reports_root / "qc"
        manifests_dir = reports_root / "manifests"
        logs_dir = reports_root / "logs"

    qc_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = logs_dir / "preprocessing.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info("=======================================================")
    logger.info("STARTING FCD PREPROCESSING PIPELINE")
    logger.info(f"Raw data root: {raw_root}")
    logger.info(f"Processed datasets root: {processed_root}")
    logger.info(f"Reports root: {reports_root}")
    logger.info(f"Config path: {config_path}")
    logger.info("=======================================================")

    # Dataset IDs
    flair_id = config.get("datasets", {}).get("flair_id", "Dataset101_FLAIR")
    t1_id = config.get("datasets", {}).get("t1_id", "Dataset102_T1")
    fusion_id = config.get("datasets", {}).get("fusion_id", "Dataset103_Fusion")

    output_dirs = {
        "flair_dataset": processed_root / flair_id,
        "t1_dataset": processed_root / t1_id,
        "fusion_dataset": processed_root / fusion_id,
    }

    # Inventory cohort
    logger.info("Running cohort inventory...")
    exp_lesion_lbl = config.get("label", {}).get("expected_lesion_value", 1)
    subject_records, inventory_summary = inventory_cohort(
        data_root=raw_root,
        participants_tsv=participants_tsv,
        require_roi=True,
        expected_lesion_label=exp_lesion_lbl,
    )

    # Write input inventory HTML
    inventory_html_path = qc_dir / "input_inventory.html"
    generate_input_inventory_html(subject_records, inventory_summary, inventory_html_path)
    logger.info(f"Input inventory written to {inventory_html_path}")

    # Save active config
    config_saved_path = manifests_dir / "preprocessing_config.json"
    write_json_atomic(config, config_saved_path)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    if inventory_only:
        logger.info("Inventory-only run requested. Exiting without processing.")
        return {
            "mode": "inventory_only",
            "inventory_summary": inventory_summary,
            "inventory_html": str(inventory_html_path),
            "config_path": str(config_saved_path),
        }

    # Filter eligible subjects
    eligible_subjects = [r for r in subject_records if r["status"] == "ELIGIBLE"]
    total_eligible = len(eligible_subjects)
    logger.info(f"Total eligible labeled subjects: {total_eligible}")

    limit_val = config.get("execution", {}).get("limit")
    if limit_val is not None:
        eligible_subjects = eligible_subjects[: int(limit_val)]
        logger.info(f"Limit applied: processing {len(eligible_subjects)} subjects.")

    # Shared subject splits tracking
    split_assignments: dict[str, list[str]] = {"train": [], "test": []}
    for r in eligible_subjects:
        s_id = r["subject_id"]
        s_split = r.get("split", "train")
        if s_split not in split_assignments:
            split_assignments[s_split] = []
        split_assignments[s_split].append(s_id)

    subject_splits_path = manifests_dir / "subject_splits.json"
    write_json_atomic(split_assignments, subject_splits_path)

    # Process subjects sequentially
    pipeline_results: list[dict[str, Any]] = []
    qc_flair_records: list[dict[str, Any]] = []
    qc_t1_records: list[dict[str, Any]] = []
    qc_reg_records: list[dict[str, Any]] = []
    qc_final_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    dataset_meta = {
        "FLAIR Dataset": str(output_dirs["flair_dataset"]),
        "T1 Dataset": str(output_dirs["t1_dataset"]),
        "Fusion Dataset": str(output_dirs["fusion_dataset"]),
        "Reference Space": "FLAIR original grid",
        "Registration": f"{config.get('registration', {}).get('method', 'rigid')} (SimpleITK)",
        "Skull Stripping": f"{config.get('skull_stripping', {}).get('backend', 'hd-bet')} ({config.get('skull_stripping', {}).get('device', 'cuda')})",
        "Normalization": f"{config.get('normalization', {}).get('method', 'foreground_zscore')} {config.get('normalization', {}).get('clip_percentiles', [0.5, 99.5])}",
    }

    def _save_current_reports() -> None:
        if qc_flair_records:
            generate_skull_stripping_html(qc_flair_records, "FLAIR", qc_dir / "skull_stripping_FLAIR.html")
        if qc_t1_records:
            generate_skull_stripping_html(qc_t1_records, "T1", qc_dir / "skull_stripping_T1.html")
        if qc_reg_records:
            generate_registration_html(qc_reg_records, qc_dir / "registration_T1_to_FLAIR.html")
        if qc_final_records:
            generate_final_dataset_qc_html(qc_final_records, dataset_meta, qc_dir / "final_dataset_qc.html")

    try:
        for idx, subj_rec in enumerate(eligible_subjects, start=1):
            s_id = subj_rec["subject_id"]
            logger.info(f"Processing [{idx}/{len(eligible_subjects)}]: {s_id}")

            res = process_subject_pipeline(
                subject_record=subj_rec,
                config=config,
                config_hash=config_hash,
                output_dirs=output_dirs,
                logger_instance=logger,
            )
            pipeline_results.append(res)

            if res.get("qc_flair_skullstrip"):
                qc_flair_records.append(res["qc_flair_skullstrip"])
            if res.get("qc_t1_skullstrip"):
                qc_t1_records.append(res["qc_t1_skullstrip"])
            if res.get("qc_registration"):
                qc_reg_records.append(res["qc_registration"])
            if res.get("dataset_validation"):
                qc_final_records.append(res["dataset_validation"])

            if res["status"] == "FAILED":
                err_msg = "; ".join(res.get("errors", [])) or "Unknown failure"
                failures.append({
                    "subject_id": s_id,
                    "split": subj_rec.get("split", "train"),
                    "error": err_msg,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                })
                if config.get("execution", {}).get("fail_fast", False):
                    logger.error(f"Fail-fast enabled: aborting pipeline due to failure on {s_id}.")
                    break

            # Progressively update HTML reports after each subject
            _save_current_reports()

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user (KeyboardInterrupt). Saving reports for completed subjects...")

    # Final reports save
    _save_current_reports()

    # Write failures CSV
    failures_csv_path = manifests_dir / "failures.csv"
    with failures_csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["subject_id", "split", "error", "timestamp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fail in failures:
            writer.writerow(fail)

    # Write nnU-Net dataset.json files for each dataset if any subjects succeeded
    num_tr = sum(r.get("split") == "train" and r.get("status") in {"PASS", "WARN"} for r in pipeline_results)
    labels_map = config.get("label", {}).get("mapping", {"background": 0, "lesion": 1})

    create_nnunet_dataset_json(
        output_dir=output_dirs["flair_dataset"],
        channel_names={"0": "FLAIR"},
        labels=labels_map,
        num_training=num_tr,
        dataset_name=flair_id,
        description="FCD Lesion Segmentation — FLAIR only representation",
    )

    create_nnunet_dataset_json(
        output_dir=output_dirs["t1_dataset"],
        channel_names={"0": "T1"},
        labels=labels_map,
        num_training=num_tr,
        dataset_name=t1_id,
        description="FCD Lesion Segmentation — T1 only representation registered to FLAIR",
    )

    create_nnunet_dataset_json(
        output_dir=output_dirs["fusion_dataset"],
        channel_names={"0": "FLAIR", "1": "T1"},
        labels=labels_map,
        num_training=num_tr,
        dataset_name=fusion_id,
        description="FCD Lesion Segmentation — Multimodal Fusion (FLAIR + registered T1)",
    )

    # Save preprocessing manifest
    manifest_data = {
        "config_hash": config_hash,
        "config": config,
        "qc_dir": str(qc_dir),
        "manifests_dir": str(manifests_dir),
        "inventory_summary": inventory_summary,
        "split_assignments": split_assignments,
        "results": pipeline_results,
        "summary": {
            "attempted": len(pipeline_results),
            "passed": sum(r["status"] == "PASS" for r in pipeline_results),
            "warned": sum(r["status"] == "WARN" for r in pipeline_results),
            "skipped": sum(r["status"] == "SKIPPED" for r in pipeline_results),
            "failed": len(failures),
        },
    }
    manifest_path = manifests_dir / "preprocessing_manifest.json"
    write_json_atomic(manifest_data, manifest_path)

    logger.info("=======================================================")
    logger.info("PREPROCESSING RUN COMPLETE")
    logger.info(f"Attempted: {len(pipeline_results)}, Passed: {manifest_data['summary']['passed']}, "
                f"Warned: {manifest_data['summary']['warned']}, Skipped: {manifest_data['summary']['skipped']}, "
                f"Failed: {manifest_data['summary']['failed']}")
    logger.info(f"Reports: {qc_dir}")
    logger.info(f"Manifests: {manifests_dir}")
    logger.info("=======================================================")

    return manifest_data
