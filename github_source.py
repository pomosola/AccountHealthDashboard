"""Download AccountHealthDashboard and use that checkout as the app source.

Git is not required. A public zip of the default branch is cached under
%LOCALAPPDATA%\\AccountHealthDashboard\\github and refreshed when the GitHub
commit changes. Files the repository does not contain are copied from the
local practicum folder when that copy exists, so the pack can still load.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

REPO = "pomosola/AccountHealthDashboard"
BRANCH = "main"
ZIP_URL = f"https://github.com/{REPO}/archive/refs/heads/{BRANCH}.zip"
COMMIT_URL = f"https://api.github.com/repos/{REPO}/commits/{BRANCH}"
LOCAL_FALLBACK = Path(
    r"c:\Users\pomos\OneDrive\Documents\Ga_Tech_Courses\Applied_Analytics_Practicum\Dashboard datasets"
)
PACK_FILES = (
    "product_health_score.csv",
    "product_adoption_score.csv",
    "account_data_state.csv",
    "account_hierarchy.csv",
    "score_context.csv",
    "account_prediction_drivers.csv",
    "opportunity_review_lists.csv",
    "customer_journey_touchpoints.csv",
    "migration_reconciliation.csv",
    "salesforce_handoff_crosswalk.csv",
    "contact_account_bridge.csv",
    "model_performance.csv",
    "review_capacity_metrics.csv",
    "calibration_summary.csv",
    "account_product_relationships.csv",
    "contact_touchpoint_links.csv",
)


def cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "AccountHealthDashboard" / "github"


def _request(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "account-health-dashboard",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _commit_sha() -> str:
    try:
        payload = json.loads(_request(COMMIT_URL, timeout=30).decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return ""
    return str(payload.get("sha") or "")


def _ready(folder: Path) -> bool:
    return (folder / "practicum_view.py").exists() and (folder / "product_health_score.csv").exists()


def _unpack_bundled_archives(folder: Path) -> None:
    """GitHub has the journey extract as a zip beside the other CSVs."""
    archive = folder / "customer_journey_touchpoints.zip"
    destination = folder / "customer_journey_touchpoints.csv"
    if destination.exists() or not archive.exists():
        return
    with zipfile.ZipFile(archive) as bundle:
        members = [
            name
            for name in bundle.namelist()
            if Path(name).name == "customer_journey_touchpoints.csv" and not name.endswith("/")
        ]
        if not members:
            return
        with bundle.open(members[0]) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def _fill_missing(folder: Path) -> list[str]:
    note = folder / ".local_fill.txt"
    recorded = []
    if note.exists():
        recorded = [line.strip() for line in note.read_text(encoding="utf-8").splitlines() if line.strip()]
    filled = [name for name in recorded if (folder / name).exists()]
    for name in PACK_FILES:
        if name in filled:
            continue
        destination = folder / name
        source = LOCAL_FALLBACK / name
        if destination.exists() or not source.exists():
            continue
        shutil.copy2(source, destination)
        filled.append(name)
    note.write_text("\n".join(filled), encoding="utf-8")
    return filled


def _download(destination: Path, sha: str) -> None:
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / "github_staging"
    incoming = parent / "github_incoming"
    if staging.exists():
        shutil.rmtree(staging)
    if incoming.exists():
        shutil.rmtree(incoming)
    staging.mkdir()
    blob = _request(ZIP_URL, timeout=300)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        archive.extractall(staging)
    extracted = next(path for path in staging.iterdir() if path.is_dir())
    shutil.move(str(extracted), str(incoming))
    shutil.rmtree(staging, ignore_errors=True)
    if sha:
        (incoming / ".commit_sha").write_text(sha, encoding="utf-8")
    previous = parent / "github_previous"
    if previous.exists():
        shutil.rmtree(previous)
    if destination.exists():
        destination.rename(previous)
    incoming.rename(destination)
    if previous.exists():
        shutil.rmtree(previous, ignore_errors=True)


def sync() -> tuple[Path, list[str]]:
    """Return the cached repo root and the names filled from the local folder."""
    root = cache_dir()
    sha = _commit_sha()
    marker = root / ".commit_sha"
    current = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    if not (_ready(root) and sha and current == sha):
        try:
            _download(root, sha)
        except (OSError, zipfile.BadZipFile, StopIteration):
            if not _ready(root):
                raise
    _unpack_bundled_archives(root)
    return root, _fill_missing(root)
