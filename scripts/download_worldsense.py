"""Download the WorldSense audio-visual QA dataset (honglyhly/WorldSense).

Tries ModelScope first (honglyhly/WorldSense), falls back to the HuggingFace
mirror (hf-mirror.com). Lands files in data/worldsense_src/:
    worldsense_qa.json, worldsense_subtitles.zip, worldsense_videos_{0..10}.zip
Run from the repo root in the mbe-ingest env. Honors MBE_ROOT.

    conda run -n mbe-ingest python scripts/download_worldsense.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("MBE_ROOT", Path(__file__).resolve().parents[1]))
DEST = ROOT / "data" / "worldsense_src"
REPO = "honglyhly/WorldSense"


def via_modelscope() -> bool:
    try:
        from modelscope import dataset_snapshot_download
    except Exception as exc:  # pragma: no cover
        print(f"[modelscope] import failed: {exc}", flush=True)
        return False
    try:
        print(f"[modelscope] downloading {REPO} -> {DEST}", flush=True)
        dataset_snapshot_download(REPO, local_dir=str(DEST))
        return True
    except Exception as exc:
        print(f"[modelscope] download failed: {exc}", flush=True)
        return False


def via_hf_mirror() -> bool:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        print(f"[hf] import failed: {exc}", flush=True)
        return False
    try:
        print(f"[hf-mirror] downloading {REPO} -> {DEST}", flush=True)
        snapshot_download(
            repo_id=REPO, repo_type="dataset", local_dir=str(DEST),
            max_workers=8,
        )
        return True
    except Exception as exc:
        print(f"[hf-mirror] download failed: {exc}", flush=True)
        return False


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    ok = via_modelscope() or via_hf_mirror()
    if not ok:
        print("ALL_DOWNLOAD_METHODS_FAILED", flush=True)
        return 1
    files = sorted(p.name for p in DEST.glob("*") if p.is_file())
    print(f"WORLDSENSE_DOWNLOAD_DONE files={len(files)}", flush=True)
    for f in files:
        print("  ", f, f"{(DEST / f).stat().st_size/1e6:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
