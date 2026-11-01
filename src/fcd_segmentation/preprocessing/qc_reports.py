"""
HTML Quality Control (QC) Report Generation for FCD MRI Preprocessing.

Generates self-contained HTML reports with embedded base64 images:
1. input_inventory.html
2. skull_stripping_FLAIR.html
3. skull_stripping_T1.html
4. registration_T1_to_FLAIR.html
5. final_dataset_qc.html
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

# Shared CSS for clean, responsive, self-contained reports
REPORT_CSS = """
  :root {
    --bg: #f8fafc;
    --card-bg: #ffffff;
    --text: #0f172a;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --pass: #16a34a;
    --pass-bg: #dcfce7;
    --review: #d97706;
    --review-bg: #fef3c7;
    --fail: #dc2626;
    --fail-bg: #fee2e2;
    --info: #2563eb;
    --info-bg: #dbeafe;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 24px;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.5;
  }
  .container { max-width: 1400px; margin: 0 auto; }
  header { margin-bottom: 24px; }
  h1 { margin: 0 0 8px 0; font-size: 1.75rem; font-weight: 700; color: #1e293b; }
  .subtitle { color: var(--text-muted); font-size: 0.95rem; margin-bottom: 16px; }
  .summary-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin: 20px 0;
  }
  .stat-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 18px;
    min-width: 130px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
  }
  .stat-card .label { font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); font-weight: 600; }
  .stat-card .value { font-size: 1.4rem; font-weight: 700; margin-top: 4px; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 20px;
  }
  .card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    overflow: hidden;
  }
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
  }
  .card-title { font-size: 1.1rem; font-weight: 600; margin: 0; }
  .badge {
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
  }
  .badge-pass { background: var(--pass-bg); color: var(--pass); }
  .badge-review { background: var(--review-bg); color: var(--review); }
  .badge-fail { background: var(--fail-bg); color: var(--fail); }
  .badge-info { background: var(--info-bg); color: var(--info); }
  .meta-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 8px; }
  .meta-table td { padding: 4px 6px; border-bottom: 1px solid #f1f5f9; }
  .meta-table td:first-child { color: var(--text-muted); font-weight: 500; width: 40%; }
  .qc-img { width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--border); margin-top: 12px; display: block; background: #000; }
  table.data-table {
    width: 100%;
    border-collapse: collapse;
    background: var(--card-bg);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    font-size: 0.85rem;
  }
  table.data-table th, table.data-table td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); }
  table.data-table th { background: #f1f5f9; font-weight: 600; color: #475569; }
  table.data-table tr:hover { background: #f8fafc; }
  .reasons { color: var(--fail); font-size: 0.8rem; margin: 4px 0 0 0; }
  .warnings { color: var(--review); font-size: 0.8rem; margin: 4px 0 0 0; }
"""


def generate_input_inventory_html(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path | str,
) -> Path:
    """Generate input_inventory.html report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = summary.get("total_subjects", len(records))
    eligible = summary.get("eligible_subjects", sum(r.get("status") == "ELIGIBLE" for r in records))
    excluded = total - eligible

    rows: list[str] = []
    for r in records:
        sid = html.escape(str(r.get("subject_id", "")))
        grp = html.escape(str(r.get("group", "unknown")))
        splt = html.escape(str(r.get("split", "unknown")))
        status = r.get("status", "EXCLUDED")
        badge_cls = "badge-pass" if status == "ELIGIBLE" else "badge-fail"

        t1_ok = "✓" if r.get("t1_path") else "✗"
        flair_ok = "✓" if r.get("flair_path") else "✗"
        roi_ok = "✓" if r.get("roi_path") else "✗"

        f_meta = r.get("flair_meta") or {}
        shape_str = f"{f_meta.get('shape')}" if f_meta.get("shape") else "N/A"
        spc = f_meta.get("spacing")
        spc_str = f"{spc[0]:.2f}x{spc[1]:.2f}x{spc[2]:.2f}" if spc else "N/A"

        r_meta = r.get("roi_meta") or {}
        labels_str = str(r_meta.get("unique_labels", []))
        disp = r.get("corner_displacement_flair_roi_mm")
        disp_str = f"{disp:.4f} mm" if disp is not None else "N/A"

        reasons = "<br>".join(html.escape(s) for s in r.get("exclusion_reasons", []))
        warns = "<br>".join(html.escape(s) for s in r.get("warnings", []))
        notes = []
        if reasons:
            notes.append(f"<div class='reasons'>{reasons}</div>")
        if warns:
            notes.append(f"<div class='warnings'>{warns}</div>")
        notes_str = "".join(notes) or "—"

        rows.append(f"""
        <tr>
          <td><strong>{sid}</strong></td>
          <td>{grp}</td>
          <td>{splt}</td>
          <td><span class="badge {badge_cls}">{html.escape(status)}</span></td>
          <td style="text-align:center;">{flair_ok}</td>
          <td style="text-align:center;">{t1_ok}</td>
          <td style="text-align:center;">{roi_ok}</td>
          <td>{shape_str}</td>
          <td>{spc_str}</td>
          <td>{labels_str}</td>
          <td>{disp_str}</td>
          <td>{notes_str}</td>
        </tr>
        """)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Input Inventory QC Report</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<div class="container">
  <header>
    <h1>FCD Preprocessing — Input Cohort Inventory</h1>
    <div class="subtitle">Cohort input verification, geometry validation, modality pairing, and exclusion reasons.</div>
    <div class="summary-bar">
      <div class="stat-card">
        <div class="label">Total Cohort</div>
        <div class="value">{total}</div>
      </div>
      <div class="stat-card">
        <div class="label">Eligible (Labeled)</div>
        <div class="value" style="color: var(--pass);">{eligible}</div>
      </div>
      <div class="stat-card">
        <div class="label">Excluded (e.g. Controls)</div>
        <div class="value" style="color: var(--fail);">{excluded}</div>
      </div>
      <div class="stat-card">
        <div class="label">ROI On FLAIR Grid</div>
        <div class="value">{summary.get("roi_on_flair_grid_count", 0)}</div>
      </div>
    </div>
  </header>

  <main>
    <table class="data-table">
      <thead>
        <tr>
          <th>Subject ID</th>
          <th>Group</th>
          <th>Split</th>
          <th>Status</th>
          <th>FLAIR</th>
          <th>T1</th>
          <th>ROI</th>
          <th>FLAIR Shape</th>
          <th>Spacing (mm)</th>
          <th>ROI Labels</th>
          <th>Corner Disp</th>
          <th>Exclusions & Warnings</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
  </main>
</div>
</body>
</html>
"""
    output_path.write_text(doc, encoding="utf-8")
    return output_path


def generate_skull_stripping_html(
    records: list[dict[str, Any]],
    modality: str,
    output_path: Path | str,
) -> Path:
    """Generate skull_stripping_FLAIR.html or skull_stripping_T1.html."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    modality = modality.upper()
    total = len(records)
    passed = sum(r.get("qc_status") == "PASS" for r in records)
    review = sum(r.get("qc_status") == "REVIEW" for r in records)
    failed = sum(r.get("status") == "failed" for r in records)

    cards: list[str] = []
    for r in records:
        sid = html.escape(str(r.get("subject_id", "unknown")))
        qc_status = r.get("qc_status", "NOT_AVAILABLE")
        status = r.get("status", "success" if qc_status in {"PASS", "REVIEW"} else "failed")

        if status != "success":
            cards.append(f"""
            <div class="card">
              <div class="card-header">
                <span class="card-title">{sid}</span>
                <span class="badge badge-fail">FAILED</span>
              </div>
              <p style="color:var(--fail); font-size:0.85rem;">{html.escape(str(r.get("error", "Unknown error")))}</p>
            </div>
            """)
            continue

        badge_cls = "badge-pass" if qc_status == "PASS" else "badge-review"
        b_fraction = r.get("brain_fraction")
        bf_str = f"{float(b_fraction):.4f}" if b_fraction is not None else "N/A"
        roi_frac = r.get("roi_inside_brain_fraction")
        rf_str = f"{float(roi_frac):.4f}" if roi_frac is not None else "N/A"
        flags = html.escape(str(r.get("review_flags", "None") or "None"))

        img_html = ""
        b64 = r.get("base64_image")
        if b64:
            img_html = f'<img class="qc-img" src="{b64}" alt="QC {sid}">'

        cards.append(f"""
        <div class="card">
          <div class="card-header">
            <span class="card-title">{sid}</span>
            <span class="badge {badge_cls}">{html.escape(qc_status)}</span>
          </div>
          <table class="meta-table">
            <tr><td>Brain Fraction</td><td>{bf_str}</td></tr>
            <tr><td>Brain Voxels</td><td>{r.get('brain_voxels', 'N/A')}</td></tr>
            <tr><td>ROI Overlap</td><td>{rf_str}</td></tr>
            <tr><td>Review Flags</td><td>{flags}</td></tr>
          </table>
          {img_html}
        </div>
        """)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{modality} Skull Stripping QC</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<div class="container">
  <header>
    <h1>HD-BET Skull Stripping QC — {modality}</h1>
    <div class="subtitle">Review axial, coronal, and sagittal orthogonal slices before and after brain extraction. Green: brain mask boundary. Red: lesion ROI.</div>
    <div class="summary-bar">
      <div class="stat-card">
        <div class="label">Total Processed</div>
        <div class="value">{total}</div>
      </div>
      <div class="stat-card">
        <div class="label">QC Passed</div>
        <div class="value" style="color: var(--pass);">{passed}</div>
      </div>
      <div class="stat-card">
        <div class="label">Review Flagged</div>
        <div class="value" style="color: var(--review);">{review}</div>
      </div>
      <div class="stat-card">
        <div class="label">Failed</div>
        <div class="value" style="color: var(--fail);">{failed}</div>
      </div>
    </div>
  </header>

  <main class="grid">
    {"".join(cards)}
  </main>
</div>
</body>
</html>
"""
    output_path.write_text(doc, encoding="utf-8")
    return output_path


def generate_registration_html(
    records: list[dict[str, Any]],
    output_path: Path | str,
) -> Path:
    """Generate registration_T1_to_FLAIR.html report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(records)
    passed = sum(r.get("status") == "PASS" for r in records)
    review = sum(r.get("status") == "REVIEW" for r in records)
    failed = sum(r.get("status") == "FAIL" for r in records)

    cards: list[str] = []
    for r in records:
        sid = html.escape(str(r.get("subject_id", "unknown")))
        status = r.get("status", "FAIL")

        if status == "FAIL":
            cards.append(f"""
            <div class="card">
              <div class="card-header">
                <span class="card-title">{sid}</span>
                <span class="badge badge-fail">FAIL</span>
              </div>
              <p style="color:var(--fail); font-size:0.85rem;">{html.escape(str(r.get("error", "Registration error")))}</p>
            </div>
            """)
            continue

        badge_cls = "badge-pass" if status == "PASS" else "badge-review"
        corr = r.get("foreground_correlation")
        corr_str = f"{float(corr):.4f}" if corr is not None else "N/A"
        metric = r.get("final_metric")
        metric_str = f"{float(metric):.4f}" if metric is not None else "N/A"
        flags = html.escape("; ".join(r.get("flags", [])) or "None")
        roi_voxels = r.get("roi_lesion_voxels", 0)
        roi_voxels_str = f"{int(roi_voxels):,}" if roi_voxels else "N/A (HC or missing)"
        roi_overlay = r.get("roi_overlay", False)
        roi_overlay_str = "Yes — red contour shown" if roi_overlay else "No"

        img_html = ""
        b64 = r.get("base64_image")
        if b64:
            img_html = f'<img class="qc-img" src="{b64}" alt="Reg QC {sid}">'

        cards.append(f"""
        <div class="card">
          <div class="card-header">
            <span class="card-title">{sid}</span>
            <span class="badge {badge_cls}">{html.escape(status)}</span>
          </div>
          <table class="meta-table">
            <tr><td>Mutual Info Metric</td><td>{metric_str}</td></tr>
            <tr><td>Foreground Correlation</td><td>{corr_str}</td></tr>
            <tr><td>Geometry Agreement</td><td>Size: {'✓' if r.get('size_match') else '✗'}, Spacing: {'✓' if r.get('spacing_match') else '✗'}</td></tr>
            <tr><td>ROI Lesion Voxels</td><td>{roi_voxels_str}</td></tr>
            <tr><td>ROI Contour Overlay</td><td>{roi_overlay_str}</td></tr>
            <tr><td>Review Flags</td><td>{flags}</td></tr>
          </table>
          {img_html}
        </div>
        """)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>T1-to-FLAIR Registration QC</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<div class="container">
  <header>
    <h1>T1-to-FLAIR Registration Quality Control</h1>
    <div class="subtitle">Reference space: FLAIR. Moving: T1 (registered). Views: Fixed FLAIR, Resampled T1, Color Overlay (FLAIR=green, T1=magenta), and 3D Checkerboard across axial, coronal, and sagittal planes. <strong>Red contour = lesion ROI</strong> (shown where available — both images are in FLAIR grid space after registration).</div>
    <div class="summary-bar">
      <div class="stat-card">
        <div class="label">Total Processed</div>
        <div class="value">{total}</div>
      </div>
      <div class="stat-card">
        <div class="label">Passed</div>
        <div class="value" style="color: var(--pass);">{passed}</div>
      </div>
      <div class="stat-card">
        <div class="label">Review Required</div>
        <div class="value" style="color: var(--review);">{review}</div>
      </div>
      <div class="stat-card">
        <div class="label">Failed</div>
        <div class="value" style="color: var(--fail);">{failed}</div>
      </div>
    </div>
  </header>

  <main class="grid">
    {"".join(cards)}
  </main>
</div>
</body>
</html>
"""
    output_path.write_text(doc, encoding="utf-8")
    return output_path


def generate_final_dataset_qc_html(
    records: list[dict[str, Any]],
    dataset_metadata: dict[str, Any],
    output_path: Path | str,
) -> Path:
    """Generate final_dataset_qc.html report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(records)
    passed = sum(r.get("status") == "PASS" for r in records)
    warned = sum(r.get("status") == "WARN" for r in records)
    failed = sum(r.get("status") == "FAIL" for r in records)

    rows: list[str] = []
    for r in records:
        sid = html.escape(str(r.get("subject_id", "")))
        splt = html.escape(str(r.get("split", "train")))
        status = r.get("status", "FAIL")
        badge_cls = "badge-pass" if status == "PASS" else ("badge-review" if status == "WARN" else "badge-fail")

        f_ok = "✓" if r.get("flair_dataset_valid") else "✗"
        t1_ok = "✓" if r.get("t1_dataset_valid") else "✗"
        fus_ok = "✓" if r.get("fusion_dataset_valid") else "✗"
        geom_ok = "✓" if r.get("fusion_geometry_match") else "✗"

        shape_str = str(r.get("target_grid_shape", "N/A"))
        spc_str = str(r.get("target_grid_spacing", "N/A"))

        l_vox = r.get("lesion_voxels", "N/A")
        f_mean = f"{float(r.get('flair_mean', 0)):.2f}" if r.get("flair_mean") is not None else "N/A"
        t1_mean = f"{float(r.get('t1_mean', 0)):.2f}" if r.get("t1_mean") is not None else "N/A"

        notes = html.escape("; ".join(r.get("warnings", []) + r.get("errors", [])) or "—")

        rows.append(f"""
        <tr>
          <td><strong>{sid}</strong></td>
          <td>{splt}</td>
          <td><span class="badge {badge_cls}">{html.escape(status)}</span></td>
          <td style="text-align:center;">{f_ok}</td>
          <td style="text-align:center;">{t1_ok}</td>
          <td style="text-align:center;">{fus_ok}</td>
          <td style="text-align:center;">{geom_ok}</td>
          <td>{shape_str}</td>
          <td>{spc_str}</td>
          <td>{l_vox}</td>
          <td>FLAIR: {f_mean}, T1: {t1_mean}</td>
          <td>{notes}</td>
        </tr>
        """)

    d_info = "<br>".join(f"<strong>{html.escape(k)}:</strong> {html.escape(str(v))}" for k, v in dataset_metadata.items())

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Final nnU-Net Datasets QC</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Final nnU-Net Datasets Quality Control</h1>
    <div class="subtitle">Validation across FLAIR-only, T1-only, and Fusion datasets: identical split, fusion channel grid alignment, discrete labels, and finite normalized intensities.</div>
    <div class="summary-bar">
      <div class="stat-card">
        <div class="label">Total Processed</div>
        <div class="value">{total}</div>
      </div>
      <div class="stat-card">
        <div class="label">Datasets PASS</div>
        <div class="value" style="color: var(--pass);">{passed}</div>
      </div>
      <div class="stat-card">
        <div class="label">Warnings</div>
        <div class="value" style="color: var(--review);">{warned}</div>
      </div>
      <div class="stat-card">
        <div class="label">Failures</div>
        <div class="value" style="color: var(--fail);">{failed}</div>
      </div>
    </div>
    <div class="card" style="margin-top:12px; font-size:0.85rem;">
      {d_info}
    </div>
  </header>

  <main>
    <table class="data-table">
      <thead>
        <tr>
          <th>Subject ID</th>
          <th>Split</th>
          <th>Status</th>
          <th>FLAIR DS</th>
          <th>T1 DS</th>
          <th>Fusion DS</th>
          <th>Fusion Grid Match</th>
          <th>Grid Shape</th>
          <th>Spacing (mm)</th>
          <th>Lesion Voxels</th>
          <th>Normalized Means</th>
          <th>Notes</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
  </main>
</div>
</body>
</html>
"""
    output_path.write_text(doc, encoding="utf-8")
    return output_path
