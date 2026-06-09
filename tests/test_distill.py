from app.distill.filter_strict import strict_filter_trajectory
from app.distill.build_sft_dataset import build_dataset, sft_row
from app.distill.common import write_json
from app.distill.frames import covers_evidence, uniform_timestamps
from app.distill.filter_consistency import strip_answer_from_cot
from app.distill.rewrite import rewrite_trajectory, validate_cot
from app.eval_harness import EvalCase

import pytest


def test_rewrite_validator_rejects_type2_leakage():
    result = validate_cot(
        "I retrieved top rank evidence with score 0.91. [FRAME:t=1.0]",
        [1.0],
    )

    assert not result.ok
    assert "forbidden_token:retriev" in result.errors
    assert any(error.startswith("forbidden_numeric") for error in result.errors)


def test_rewrite_validator_rejects_non_frame_markers_and_unknown_frames():
    result = validate_cot(
        "I inspect the slide [SLIDE:t=2.0] and frame [FRAME:t=99.0].",
        [1.0],
    )

    assert not result.ok
    assert "forbidden_slide_marker" in result.errors
    assert "unknown_frame_marker:99.0" in result.errors


def test_rewrite_validator_accepts_visible_frame_cot():
    result = validate_cot(
        "I compare the visible red object [FRAME:t=1.0], then answer A.",
        [1.0, 2.0],
    )

    assert result.ok


def test_strict_filter_drops_text_evidence_scope():
    trajectory = {
        "case": {
            "case_id": "scope-1",
            "video_id": "vid",
            "question": "Q",
            "reference_answer": "cat",
            "gold_timestamps": [],
            "gold_scenes": [],
        },
        "tool_steps": [
            {"tool_name": "retrieve_transcript_evidence"},
            {"tool_name": "answer_with_evidence"},
        ],
        "state": {
            "retrieved_frames": [],
            "retrieved_transcripts": [{"text": "cat"}],
            "draft_answer": "cat",
            "grounding_report": {"grounded": True},
        },
    }
    case = EvalCase(case_id="scope-1", video_id="vid", question="Q", reference_answer="cat")

    result = strict_filter_trajectory(trajectory, case_override=case)

    assert not result["passed"]
    assert "out_of_scope_text_or_audio" in result["drop_reasons"]


def test_sft_builder_asserts_video_isolation(tmp_path):
    cot_path = tmp_path / "cot.json"
    write_json(
        cot_path,
        {
            "case_id": "c1",
            "video_id": "video-overlap",
            "question": "Q",
            "frame_paths": ["/tmp/frame.jpg"],
            "shown_frames": [1.0],
            "cot": "I look [FRAME:t=1.0].",
            "answer": "A",
            "validation_report": {"ok": True},
        },
    )
    eval_cases = tmp_path / "eval.jsonl"
    eval_cases.write_text(
        '{"case_id":"e1","video_id":"video-overlap","question":"Q"}\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="leakage"):
        build_dataset(
            cot_paths=[cot_path],
            output=tmp_path / "out.jsonl",
            eval_cases_path=eval_cases,
        )


def test_uniform_timestamps_avoid_clip_edges():
    ts = uniform_timestamps(10.0, 4)

    assert ts == [2.0, 4.0, 6.0, 8.0]
    assert 0.0 not in ts and 10.0 not in ts


def test_covers_evidence_scene_and_point():
    # Sample lands inside a gold scene (with tolerance padding).
    assert covers_evidence([5.0, 9.0], [], [{"start": 8.0, "end": 12.0}])
    # Sample near a gold point timestamp.
    assert covers_evidence([19.0], [20.0], [])
    # Sample misses both -> not covered.
    assert not covers_evidence([1.0, 2.0], [50.0], [{"start": 40.0, "end": 45.0}])
    # No GT at all -> cannot be falsified.
    assert covers_evidence([1.0], [], [])


def test_strip_answer_from_cot_removes_option_and_final_answer():
    question = "Q\n\nCandidates:\nA) a red ball\nB) a blue box\n"
    cot = (
        "I look at the visible object [FRAME:t=1.0]; it is round and red.\n"
        "The correct option is A) a red ball.\n"
        "Final answer: A"
    )

    stripped = strip_answer_from_cot(cot, question, "A")

    assert "Final answer" not in stripped
    assert "a red ball" not in stripped
    assert "[FRAME:t=1.0]" in stripped  # reasoning is preserved


def _trajectory_with_gold(sampler_ts: float, gold_scene: dict) -> dict:
    return {
        "case": {
            "case_id": "g1",
            "video_id": "vidg",
            "question": "Q",
            "reference_answer": "A",
            "gold_timestamps": [],
            "gold_scenes": [gold_scene],
            "question_type": "TC",
        },
        "tool_steps": [{"tool_name": "answer_with_evidence"}],
        "state": {
            "retrieved_frames": [],
            "sampler_frames": [{"timestamp": sampler_ts, "source": "uniform", "path": f"/tmp/t{sampler_ts}.jpg"}],
            "draft_answer": "A",
            "grounding_report": {"grounded": True},
        },
        "source_traj_hash": "hash-g1",
    }


def test_cot_artifact_and_sft_row_carry_grounding_gt():
    trajectory = _trajectory_with_gold(5.0, {"start": 4.0, "end": 6.0})

    cot_payload = rewrite_trajectory(trajectory, dry_run_cot="I see it [FRAME:t=5.0].")

    assert cot_payload["gold_scenes"] == [{"start": 4.0, "end": 6.0}]
    assert cot_payload["frame_paths"] == ["/tmp/t5.0.jpg"]
    row = sft_row(cot_payload)
    assert row["gold_scenes"] == [{"start": 4.0, "end": 6.0}]


def test_strict_filter_drops_evidence_not_in_uniform_sample():
    # Uniform sample at t=2.0 but evidence is at t=40-45 -> uncovered.
    trajectory = _trajectory_with_gold(2.0, {"start": 40.0, "end": 45.0})
    case = EvalCase(
        case_id="g1",
        video_id="vidg",
        question="Q",
        reference_answer="A",
        gold_scenes=[{"start": 40.0, "end": 45.0}],
    )

    result = strict_filter_trajectory(trajectory, case_override=case)

    assert not result["passed"]
    assert "evidence_not_in_uniform_sample" in result["drop_reasons"]
    assert result["coverage_ok"] is False
