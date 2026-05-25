"""Download + sample + cleanup helpers for eval datasets.

LongVideoBench: HuggingFace dataset (gated), routed via hf-mirror.com by default.
NExT-GQA: raw GitHub fetch of annotation files.

Heavy CLI orchestration lives in scripts/eval_prepare_datasets.py; the per-dataset
mapping logic lives in scripts/eval_convert_*.py. This module just owns network
I/O and disk cleanup.
"""
from __future__ import annotations

import logging
import os
import random
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

LONGVIDEOBENCH_REPO = "longvideobench/LongVideoBench"
LONGVIDEOBENCH_FILES = ["lvb_val.json"]

NEXTGQA_RAW_BASE = "https://raw.githubusercontent.com/doc-doc/NExT-GQA/main/datasets/nextgqa"
NEXTGQA_FILES = [
    "val.csv",
    "gsub_val.json",
    "frame2time_val.json",
    "map_vid_vidorID.json",
]


def download_longvideobench(raw_dir: Path) -> dict[str, Path]:
    """Download LongVideoBench annotations into raw_dir. Returns mapping of file
    name -> local path. Routes through HF_ENDPOINT (defaults to hf-mirror.com).
    """
    from huggingface_hub import snapshot_download

    endpoint = os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
    os.environ["HF_ENDPOINT"] = endpoint
    token = os.environ.get("HF_TOKEN") or None

    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading LongVideoBench annotations from %s", endpoint)
    try:
        snapshot_path = snapshot_download(
            repo_id=LONGVIDEOBENCH_REPO,
            repo_type="dataset",
            allow_patterns=LONGVIDEOBENCH_FILES,
            local_dir=str(raw_dir),
            endpoint=endpoint,
            token=token,
        )
    except Exception as exc:
        if _longvideobench_access_is_gated(endpoint=endpoint, token=token):
            raise RuntimeError(
                "LongVideoBench is gated on HuggingFace. Add an authorized HF_TOKEN "
                "to .env or the process environment after accepting access at "
                "https://huggingface.co/datasets/longvideobench/LongVideoBench."
            ) from exc
        message = str(exc)
        if "GatedRepo" in message or "restricted" in message or "authenticated" in message:
            raise RuntimeError(
                "LongVideoBench is gated on HuggingFace. Add an authorized HF_TOKEN "
                "to .env or the process environment after accepting access at "
                "https://huggingface.co/datasets/longvideobench/LongVideoBench."
            ) from exc
        raise
    snapshot_dir = Path(snapshot_path)
    result = {}
    for name in LONGVIDEOBENCH_FILES:
        local = snapshot_dir / name
        if not local.exists():
            raise FileNotFoundError(
                f"LongVideoBench download did not produce {name} at {local}. "
                "If the dataset is gated, set HF_TOKEN or click-through the license "
                "at https://huggingface.co/datasets/longvideobench/LongVideoBench."
            )
        result[name] = local
    return result


def _longvideobench_access_is_gated(*, endpoint: str, token: str | None) -> bool:
    """Detect gated-repo failures hidden behind huggingface_hub cache errors."""
    headers = {"Authorization": f"Bearer {token}"} if token else None
    url = f"{endpoint.rstrip('/')}/datasets/{LONGVIDEOBENCH_REPO}/resolve/main/lvb_val.json"
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            response = client.head(url, headers=headers)
    except Exception:
        return False
    message = " ".join(
        str(value)
        for value in (
            response.headers.get("x-error-code"),
            response.headers.get("x-error-message"),
            response.text[:200],
        )
        if value
    ).lower()
    return response.status_code in {401, 403} and (
        "gated" in message or "restricted" in message or "authenticated" in message
    )


def download_nextgqa(raw_dir: Path) -> dict[str, Path]:
    """Download NExT-GQA annotation files from raw GitHub into raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for name in NEXTGQA_FILES:
            url = f"{NEXTGQA_RAW_BASE}/{name}"
            target = raw_dir / name
            logger.info("Downloading %s -> %s", url, target)
            response = client.get(url)
            response.raise_for_status()
            target.write_bytes(response.content)
            result[name] = target
    return result


def cleanup_raw(raw_dir: Path) -> None:
    """Remove a raw download directory if it exists. Idempotent."""
    if not raw_dir.exists():
        return
    logger.info("Cleaning up %s", raw_dir)
    shutil.rmtree(raw_dir, ignore_errors=True)


def cleanup_hf_cache(repo_id: str) -> None:
    """Best-effort delete the HF hub cache entry for a dataset repo."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return
    try:
        cache_info = scan_cache_dir()
    except Exception as exc:
        logger.debug("scan_cache_dir failed: %s", exc)
        return
    revisions_to_delete = []
    for repo in cache_info.repos:
        if repo.repo_id == repo_id and repo.repo_type == "dataset":
            revisions_to_delete.extend(rev.commit_hash for rev in repo.revisions)
    if revisions_to_delete:
        logger.info("Deleting HF cache for %s (%d revisions)", repo_id, len(revisions_to_delete))
        cache_info.delete_revisions(*revisions_to_delete).execute()


def stratified_sample(
    items: list[dict[str, Any]],
    *,
    sample_size: int,
    key: Callable[[dict[str, Any]], str],
    seed: int = 17,
) -> list[dict[str, Any]]:
    """Pick `sample_size` items, allocated across `key` strata proportionally to
    each stratum's share of the population. Random within each stratum.
    """
    if sample_size >= len(items):
        return list(items)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        buckets[key(item)].append(item)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    total = len(items)
    allocations: dict[str, int] = {}
    for stratum, bucket in buckets.items():
        share = len(bucket) / total
        allocations[stratum] = max(1, round(share * sample_size))
    while sum(allocations.values()) > sample_size:
        donor = max(allocations, key=lambda k: allocations[k] / max(len(buckets[k]), 1))
        if allocations[donor] > 1:
            allocations[donor] -= 1
        else:
            break
    while sum(allocations.values()) < sample_size:
        donor = min(allocations, key=lambda k: allocations[k] / max(len(buckets[k]), 1))
        if allocations[donor] < len(buckets[donor]):
            allocations[donor] += 1
        else:
            break
    sampled: list[dict[str, Any]] = []
    for stratum, count in allocations.items():
        sampled.extend(buckets[stratum][:count])
    rng.shuffle(sampled)
    return sampled[:sample_size]


def write_jsonl(path: Path, items: Iterable[dict[str, Any]]) -> int:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False))
            handle.write("\n")
            count += 1
    return count


def write_json(path: Path, payload: Any) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
