"""Unit tests for HTML QC reports generation."""


from fcd_segmentation.preprocessing.qc_reports import (
    generate_final_dataset_qc_html,
    generate_input_inventory_html,
    generate_registration_html,
    generate_skull_stripping_html,
)


def test_generate_input_inventory_html(tmp_path):
    out = tmp_path / "inventory.html"
    records = [
        {
            "subject_id": "sub-001",
            "group": "fcd",
            "split": "train",
            "status": "ELIGIBLE",
            "t1_path": "/path/t1.nii.gz",
            "flair_path": "/path/flair.nii.gz",
            "roi_path": "/path/roi.nii.gz",
            "flair_meta": {"shape": [160, 256, 256], "spacing": [1.0, 1.0, 1.0]},
            "roi_meta": {"unique_labels": [0.0, 1.0]},
            "corner_displacement_flair_roi_mm": 0.00004,
            "exclusion_reasons": [],
            "warnings": [],
        },
        {
            "subject_id": "sub-002",
            "group": "hc",
            "split": "train",
            "status": "EXCLUDED",
            "t1_path": "/path/t1.nii.gz",
            "flair_path": "/path/flair.nii.gz",
            "roi_path": None,
            "exclusion_reasons": ["Missing required lesion ROI mask (e.g. Healthy Control)"],
            "warnings": [],
        },
    ]
    summary = {"total_subjects": 2, "eligible_subjects": 1, "excluded_subjects": 1, "roi_on_flair_grid_count": 1}

    p = generate_input_inventory_html(records, summary, out)
    assert p.is_file()
    content = p.read_text(encoding="utf-8")
    assert "sub-001" in content
    assert "sub-002" in content
    assert "Missing required lesion ROI mask" in content
    assert "badge-pass" in content
    assert "badge-fail" in content


def test_generate_skull_stripping_html(tmp_path):
    out = tmp_path / "ss.html"
    records = [
        {
            "subject_id": "sub-001",
            "status": "success",
            "qc_status": "PASS",
            "brain_fraction": 0.12,
            "brain_voxels": 150000,
            "roi_inside_brain_fraction": 1.0,
            "review_flags": "",
            "base64_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==",
        }
    ]
    p = generate_skull_stripping_html(records, "FLAIR", out)
    assert p.is_file()
    content = p.read_text(encoding="utf-8")
    assert "HD-BET Skull Stripping QC — FLAIR" in content
    assert "sub-001" in content
    assert "data:image/png;base64," in content


def test_generate_registration_html(tmp_path):
    out = tmp_path / "reg.html"
    records = [
        {
            "subject_id": "sub-001",
            "status": "PASS",
            "final_metric": -0.523,
            "foreground_correlation": 0.75,
            "size_match": True,
            "spacing_match": True,
            "roi_lesion_voxels": 450,
            "roi_overlay": True,
            "flags": [],
            "base64_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==",
        }
    ]
    p = generate_registration_html(records, out)
    assert p.is_file()
    content = p.read_text(encoding="utf-8")
    assert "T1-to-FLAIR Registration Quality Control" in content
    assert "-0.5230" in content
    assert "ROI Lesion Voxels" in content
    assert "450" in content
    assert "red contour" in content
    assert "data:image/png;base64," in content


def test_generate_final_dataset_qc_html(tmp_path):
    out = tmp_path / "final.html"
    records = [
        {
            "subject_id": "sub-001",
            "split": "train",
            "status": "PASS",
            "flair_dataset_valid": True,
            "t1_dataset_valid": True,
            "fusion_dataset_valid": True,
            "fusion_geometry_match": True,
            "target_grid_shape": [160, 256, 256],
            "target_grid_spacing": [1.0, 1.0, 1.0],
            "lesion_voxels": 450,
            "flair_mean": 0.05,
            "t1_mean": -0.02,
            "warnings": [],
            "errors": [],
        }
    ]
    meta = {"FLAIR Dataset": "Dataset101_FLAIR"}
    p = generate_final_dataset_qc_html(records, meta, out)
    assert p.is_file()
    content = p.read_text(encoding="utf-8")
    assert "Final nnU-Net Datasets Quality Control" in content
    assert "sub-001" in content
    assert "Dataset101_FLAIR" in content
