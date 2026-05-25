"""Tests for the dataset converters using tiny synthetic fixtures.

These exercise the mapping logic without hitting the network.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT_DIR / "tests" / "fixtures" / "datasets"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lvb_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    return _load_module(
        "eval_convert_longvideobench",
        ROOT_DIR / "scripts" / "eval_convert_longvideobench.py",
    )


@pytest.fixture(scope="module")
def nextgqa_module():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    return _load_module(
        "eval_convert_nextgqa",
        ROOT_DIR / "scripts" / "eval_convert_nextgqa.py",
    )


def _read_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def test_lvb_converter_keeps_all_three_buckets_and_correct_answer(lvb_module, tmp_path, monkeypatch):
    cases_out = tmp_path / "lvb_cases.jsonl"
    manifest_out = tmp_path / "lvb_manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_convert_longvideobench.py",
            "--source", str(FIXTURES_DIR / "lvb_val_tiny.json"),
            "--sample", "3",
            "--seed", "1",
            "--output-cases", str(cases_out),
            "--output-manifest", str(manifest_out),
        ],
    )
    rc = lvb_module.main()
    assert rc == 0

    cases = _read_jsonl(cases_out)
    assert len(cases) == 3
    case_ids = [c["case_id"] for c in cases]
    assert all(cid.startswith("longvideobench-") for cid in case_ids)
    by_id = {c["case_id"]: c for c in cases}
    short = by_id["longvideobench-lvb-short-1"]
    assert short["reference_answer"] == "a blue shirt"
    assert "blue" in short["required_keywords"] or "shirt" in short["required_keywords"]
    assert "Candidates:" in short["question"]
    assert short["video_id"] == ""

    manifest = json.loads(manifest_out.read_text())
    buckets = {entry["duration_bucket"] for entry in manifest}
    assert buckets == {"short", "medium", "long"}


def test_nextgqa_converter_extracts_gold_scenes_and_temporal_action(nextgqa_module, tmp_path, monkeypatch):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    shutil.copy(FIXTURES_DIR / "nextgqa_val_tiny.csv", source_dir / "val.csv")
    shutil.copy(FIXTURES_DIR / "nextgqa_gsub_val_tiny.json", source_dir / "gsub_val.json")
    shutil.copy(FIXTURES_DIR / "nextgqa_map_vid_vidorID_tiny.json", source_dir / "map_vid_vidorID.json")
    # frame2time isn't strictly required by the converter but the script's existence
    # check ignores it; keep parity with the prod source dir by touching the file.
    (source_dir / "frame2time_val.json").write_text("{}")

    cases_out = tmp_path / "nextgqa_cases.jsonl"
    manifest_out = tmp_path / "nextgqa_manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_convert_nextgqa.py",
            "--source-dir", str(source_dir),
            "--sample", "3",
            "--seed", "1",
            "--output-cases", str(cases_out),
            "--output-manifest", str(manifest_out),
        ],
    )
    rc = nextgqa_module.main()
    assert rc == 0

    cases = _read_jsonl(cases_out)
    by_id = {c["case_id"]: c for c in cases}

    q1 = by_id["nextgqa-4882821564-q1"]
    assert q1["reference_answer"] == "to drink water"
    assert q1["gold_scenes"] == [{"start": 1.0, "end": 2.5}]
    assert q1["gold_timestamps"] == [pytest.approx(1.75)]
    assert "expected_action" not in q1  # CW is not temporal

    q2 = by_id["nextgqa-5750463032-q2"]
    assert q2["gold_scenes"] == [
        {"start": 2.0, "end": 3.5},
        {"start": 4.0, "end": 5.0},
    ]
    assert q2["expected_action"] == "build_timeline"  # TN is temporal

    manifest = json.loads(manifest_out.read_text())
    by_case = {entry["case_id"]: entry for entry in manifest}
    assert by_case["nextgqa-5750463032-q2"]["vidor_path"] == "1015/5750463032.mp4"
