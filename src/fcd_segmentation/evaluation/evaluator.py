"""Cohort-level evaluation engine, aggregation, and modality comparison for FCD segmentation."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from fcd_segmentation.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class SubjectMetricRecord:
    subject_id: str
    metrics: dict[str, float]
    group: str = "unknown"
    split: str = "unknown"
    fold: int | None = None

    def to_dict(self) -> dict[str, Any]:
        res = {
            "subject_id": self.subject_id,
            "group": self.group,
            "split": self.split,
        }
        if self.fold is not None:
            res["fold"] = self.fold
        res.update(self.metrics)
        return res


@dataclass
class EvaluationReport:
    name: str
    split: str
    num_cases: int
    expected_num_cases: int
    subject_records: list[SubjectMetricRecord]
    fold: int | None = None
    summary_statistics: dict[str, dict[str, float]] = field(default_factory=dict)
    subgroup_statistics: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    scientific_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "split": self.split,
            "num_cases": self.num_cases,
            "expected_num_cases": self.expected_num_cases,
            "scientific_notes": self.scientific_notes,
            "summary_statistics": self.summary_statistics,
            "subgroup_statistics": self.subgroup_statistics,
            "per_subject": [r.to_dict() for r in self.subject_records],
        }
        if self.fold is not None:
            data["fold"] = self.fold
        return data

    def save_json(self, path: Path | str) -> None:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_csv(self, path: Path | str) -> None:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        if not self.subject_records:
            return

        rows = [r.to_dict() for r in self.subject_records]
        fieldnames = list(rows[0].keys())
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def compute_summary_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, std, median, 25th, and 75th percentiles.

    Includes explicit reporting of:
    - finite_count
    - infinite_count
    - infinite_percentage
    - finite_only_mean
    """
    arr = np.array(values, dtype=float)
    total_count = len(arr)
    valid_mask = np.isfinite(arr)
    valid = arr[valid_mask]
    finite_count = len(valid)
    inf_count = total_count - finite_count
    inf_pct = round(100.0 * inf_count / float(total_count), 2) if total_count > 0 else 0.0

    finite_mean = float(np.mean(valid)) if finite_count > 0 else float("nan")

    return {
        "mean": finite_mean,
        "finite_only_mean": finite_mean,
        "std": float(np.std(valid)) if finite_count > 0 else float("nan"),
        "median": float(np.median(valid)) if finite_count > 0 else float("nan"),
        "iqr_25": float(np.percentile(valid, 25)) if finite_count > 0 else float("nan"),
        "iqr_75": float(np.percentile(valid, 75)) if finite_count > 0 else float("nan"),
        "count": float(finite_count),
        "finite_count": float(finite_count),
        "infinite_count": float(inf_count),
        "infinite_percentage": float(inf_pct),
        "total_count": float(total_count),
    }


class DatasetEvaluator:
    """Evaluates predicted segmentation masks against ground-truth labels."""

    def __init__(self, participants_tsv: Path | str | None = None):
        self.participants_info: dict[str, dict[str, str]] = {}
        if participants_tsv and Path(participants_tsv).is_file():
            with Path(participants_tsv).open("r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    sub_id = row.get("participant_id", "").strip()
                    if sub_id:
                        self.participants_info[sub_id] = {
                            "group": row.get("group", "unknown").strip().lower(),
                            "split": row.get("split", "unknown").strip().lower(),
                        }

    @classmethod
    def evaluate_predictions_dir(
        cls,
        name: str,
        predictions_dir: Path | str,
        ground_truth_dir: Path | str,
        participants_tsv: Path | str | None = None,
        split: str = "test",
        fold: int | str | None = None,
        splits_file: Path | str | None = None,
        file_ending: str = ".nii.gz",
        threshold: float = 0.5,
    ) -> EvaluationReport:
        """Strictly evaluate a predictions directory against ground-truth labels.

        Enforces:
        - Fold-aware evaluation for split='val': loads splits_final.json and strictly
          evaluates the validation subjects of that fold against labelsTr.
        - Held-out test evaluation for split='test': strictly evaluates test subjects against labelsTs.
        - Rejection of predictions outside the expected cohort.
        - Detection of missing predictions and missing ground-truth files.
        - Shape and affine matrix compatibility between prediction and ground truth.
        - Validation of discrete binary or bounded continuous probability predictions.
        - Complete HD95 reporting including finite-only mean and infinite-case statistics.
        """
        pred_path = Path(predictions_dir).resolve()
        gt_path = Path(ground_truth_dir).resolve()

        if not pred_path.is_dir():
            raise FileNotFoundError(f"Predictions directory does not exist: {pred_path}")
        if not gt_path.is_dir():
            raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_path}")

        evaluator = cls(participants_tsv)
        expected_subjects: set[str] = set()
        fold_idx: int | None = None

        # 1. Determine expected subjects based on split and fold
        if split.lower() == "val":
            if fold is None:
                raise ValueError(
                    f"Fold index (0..4) is required when evaluating validation split for {name}. "
                    f"Specify --fold <index>."
                )
            try:
                fold_idx = int(fold)
            except ValueError as e:
                raise ValueError(f"Invalid fold index '{fold}': must be integer 0..4") from e

            # Locate splits_final.json
            resolved_splits_file: Path | None = None
            if splits_file:
                candidate = Path(splits_file).resolve()
                if candidate.is_file():
                    resolved_splits_file = candidate
            if resolved_splits_file is None:
                # Default location: data/nnunet_preprocessed/<name>/splits_final.json
                default_cand = gt_path.parents[1] / "nnunet_preprocessed" / name / "splits_final.json"
                if default_cand.is_file():
                    resolved_splits_file = default_cand

            if resolved_splits_file is None or not resolved_splits_file.is_file():
                raise FileNotFoundError(
                    f"splits_final.json for {name} not found at {resolved_splits_file or splits_file}. "
                    f"Required to determine fold {fold_idx} validation subjects. "
                    f"Run 'python scripts/train.py --split-folds --dataset all' first."
                )

            with resolved_splits_file.open("r", encoding="utf-8") as f:
                splits_data = json.load(f)

            if fold_idx < 0 or fold_idx >= len(splits_data):
                raise ValueError(
                    f"Fold index {fold_idx} out of range for {resolved_splits_file} "
                    f"(contains {len(splits_data)} folds)."
                )

            expected_subjects = set(splits_data[fold_idx]["val"])
            logger.info(
                "Loaded fold %d validation cohort for %s from %s: %d subjects",
                fold_idx,
                name,
                resolved_splits_file,
                len(expected_subjects),
            )

        elif split.lower() == "test":
            if participants_tsv and Path(participants_tsv).is_file():
                expected_subjects = {
                    pid
                    for pid, info in evaluator.participants_info.items()
                    if info.get("split", "").lower() == "test"
                }
        elif split.lower() == "train":
            if participants_tsv and Path(participants_tsv).is_file():
                expected_subjects = {
                    pid
                    for pid, info in evaluator.participants_info.items()
                    if info.get("split", "").lower() == "train"
                }

        # Discover all prediction files
        pred_files = sorted(pred_path.glob(f"*{file_ending}"))
        if not pred_files:
            raise FileNotFoundError(f"No prediction files matching '*{file_ending}' found in {pred_path}")

        # Map discovered predictions by subject ID
        pred_by_subject: dict[str, Path] = {}
        for pf in pred_files:
            raw_stem = pf.name[: -len(file_ending)]
            case_id = raw_stem[:-5] if raw_stem.endswith("_0000") else raw_stem

            if case_id in pred_by_subject:
                raise ValueError(
                    f"Duplicate prediction detected for subject '{case_id}': "
                    f"'{pf.name}' and '{pred_by_subject[case_id].name}'"
                )
            pred_by_subject[case_id] = pf

        discovered_subjects = set(pred_by_subject.keys())

        # If expected subjects are defined, enforce exact coverage
        if expected_subjects:
            missing_preds = expected_subjects - discovered_subjects
            extra_preds = discovered_subjects - expected_subjects

            cohort_label = f"fold {fold_idx} validation" if split.lower() == "val" else f"{split} split"

            if missing_preds:
                raise ValueError(
                    f"Incomplete prediction coverage for {cohort_label} on {name}!\n"
                    f"Missing predictions for {len(missing_preds)} expected case(s): {sorted(missing_preds)}"
                )

            if extra_preds:
                raise ValueError(
                    f"Predictions found for {len(extra_preds)} case(s) outside the expected {cohort_label} cohort on {name}:\n"
                    f"{sorted(extra_preds)}"
                )

            subjects_to_evaluate = sorted(expected_subjects)
        else:
            subjects_to_evaluate = sorted(discovered_subjects)

        records: list[SubjectMetricRecord] = []

        for case_id in subjects_to_evaluate:
            pf = pred_by_subject[case_id]
            gt_file = gt_path / f"{case_id}{file_ending}"

            if not gt_file.is_file():
                raise FileNotFoundError(
                    f"Ground-truth label file missing for subject '{case_id}' in {gt_path}: {gt_file}"
                )

            try:
                pred_nii = nib.load(pf)
                gt_nii = nib.load(gt_file)
            except Exception as e:
                raise RuntimeError(f"Failed to read NIfTI file for subject '{case_id}': {e}") from e

            # Verify shape compatibility
            if pred_nii.shape != gt_nii.shape:
                raise ValueError(
                    f"Shape mismatch for subject '{case_id}': "
                    f"prediction shape {pred_nii.shape} != ground truth shape {gt_nii.shape}"
                )

            # Verify affine compatibility
            if not np.allclose(pred_nii.affine, gt_nii.affine, atol=1e-3, rtol=1e-3):
                raise ValueError(
                    f"Affine matrix mismatch for subject '{case_id}': "
                    f"prediction affine is not aligned with ground truth affine"
                )

            pred_data = np.asarray(pred_nii.dataobj)
            gt_data = np.asarray(gt_nii.dataobj)

            # Validate prediction data values
            if np.issubdtype(pred_data.dtype, np.floating):
                pred_min, pred_max = float(np.min(pred_data)), float(np.max(pred_data))
                if pred_min < -1e-4 or pred_max > 1.0 + 1e-4:
                    raise ValueError(
                        f"Invalid continuous prediction values for subject '{case_id}': "
                        f"min={pred_min}, max={pred_max}. Expected probabilities in [0.0, 1.0]."
                    )
                pred_binary = pred_data >= threshold
            else:
                unique_vals = set(np.unique(pred_data).tolist())
                if not unique_vals.issubset({0, 1}):
                    raise ValueError(
                        f"Invalid discrete prediction values for subject '{case_id}': {unique_vals}. "
                        f"Expected binary mask with values in {{0, 1}}."
                    )
                pred_binary = pred_data > 0

            gt_binary = gt_data > 0

            spacing = tuple(float(x) for x in pred_nii.header.get_zooms()[:3])
            metrics = compute_all_metrics(pred_binary, gt_binary, voxel_spacing=spacing)

            part_info = evaluator.participants_info.get(case_id, {})
            group = part_info.get("group", "unknown")
            sub_split = part_info.get("split", split)

            records.append(SubjectMetricRecord(
                subject_id=case_id,
                metrics=metrics,
                group=group,
                split=sub_split,
                fold=fold_idx,
            ))

        # Compute aggregate summary statistics
        metric_names = ["dice", "sensitivity", "specificity", "precision", "hd95"]
        summary_stats = {}
        for m in metric_names:
            vals = [r.metrics[m] for r in records if m in r.metrics]
            summary_stats[m] = compute_summary_stats(vals)

        # Compute subgroup statistics by group (fcd vs hc)
        subgroup_stats = {}
        groups = {r.group for r in records}
        for g in groups:
            subgroup_stats[g] = {}
            g_records = [r for r in records if r.group == g]
            for m in metric_names:
                vals = [r.metrics[m] for r in g_records if m in r.metrics]
                subgroup_stats[g][m] = compute_summary_stats(vals)

        # Scientific reporting notes
        scientific_notes = []
        if split.lower() == "test":
            scientific_notes.append(
                "Held-out test cohort consists strictly of 28 FCD lesion subjects and 0 healthy controls. "
                "Specificity on this test cohort reflects non-lesion brain tissue within FCD subjects, "
                "not independent healthy control false-positive rates."
            )
            scientific_notes.append(
                "Test results must be reported independently and not combined with cross-validation results."
            )
        elif split.lower() == "val":
            scientific_notes.append(
                f"Cross-validation fold {fold_idx} evaluation consisting of {len(records)} validation cases "
                f"evaluated against ground-truth labels from labelsTr."
            )

        return EvaluationReport(
            name=name,
            split=split,
            fold=fold_idx,
            num_cases=len(records),
            expected_num_cases=len(expected_subjects) if expected_subjects else len(records),
            subject_records=records,
            summary_statistics=summary_stats,
            subgroup_statistics=subgroup_stats,
            scientific_notes=scientific_notes,
        )


def compare_modalities(reports: list[EvaluationReport]) -> dict[str, Any]:
    """Generate side-by-side comparison across FLAIR, T1, and Fusion models."""
    comparison: dict[str, Any] = {
        "modalities": [r.name for r in reports],
        "splits": [r.split for r in reports],
        "folds": [r.fold for r in reports],
        "metrics_summary": {},
    }

    metric_names = ["dice", "sensitivity", "specificity", "precision", "hd95"]
    for m in metric_names:
        comparison["metrics_summary"][m] = {}
        for r in reports:
            stats = r.summary_statistics.get(m, {})
            comparison["metrics_summary"][m][r.name] = {
                "mean": stats.get("mean"),
                "finite_only_mean": stats.get("finite_only_mean"),
                "std": stats.get("std"),
                "median": stats.get("median"),
                "iqr_25": stats.get("iqr_25"),
                "iqr_75": stats.get("iqr_75"),
                "count": stats.get("count"),
                "finite_count": stats.get("finite_count"),
                "infinite_count": stats.get("infinite_count"),
                "infinite_percentage": stats.get("infinite_percentage"),
            }

    return comparison
