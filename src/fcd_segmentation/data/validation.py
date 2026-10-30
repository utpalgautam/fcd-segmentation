"""Initial file-level audit interfaces."""

from dataclasses import dataclass, field
from pathlib import Path

from fcd_segmentation.data.discovery import discover_flair_files


@dataclass
class DatasetAudit:
    root: Path
    flair_files: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def audit_dataset(data_root):
    root = Path(data_root).expanduser().resolve()
    result = DatasetAudit(root=root)
    try:
        result.flair_files = discover_flair_files(root)
    except (NotADirectoryError, FileNotFoundError) as exc:
        result.errors.append(str(exc))
        return result
    counts = {}
    for path in result.flair_files:
        subject = path.parents[1].name
        counts[subject] = counts.get(subject, 0) + 1
    for subject, count in sorted(counts.items()):
        if count > 1:
            result.warnings.append(
                f"{subject} has {count} FLAIR acquisitions; "
                "no automatic selection made."
)
    return result
