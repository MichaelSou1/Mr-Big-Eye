from pathlib import Path
import json
import subprocess
import sys

from app.eval_harness import (
    EvalCase,
    EvalPrediction,
    JudgeCache,
    PredictionCache,
    _parse_judge_json,
    evaluate_agent_loop,
    evaluate_answer,
    evaluate_audiovisual_answer,
    evaluate_case,
    evaluate_retrieval,
    extract_evidence_markers,
    group_by_case_prefix,
    load_cases,
    load_predictions,
    summarize_results,
    write_markdown_report,
)


def test_retrieval_harness_scores_timestamp_and_scene_hits():
    result = evaluate_retrieval(
        gold_timestamps=[10.0, 20.0],
        retrieved_timestamps=[9.5, 19.0, 40.0],
        gold_scenes=[{"start": 8.0, "end": 12.0}],
        scene_hits=[{"start": 9.0, "end": 11.0}],
        tolerance_sec=1.5,
        recall_k=None,
    )

    assert result["passed"]
    assert result["recall_at_k"] == 1.0
    assert result["scene_hit_accuracy"] == 1.0


def test_answer_harness_flags_hallucination_and_bad_citation():
    result = evaluate_answer(
        answer="The person holds a cat.\n[FRAME:t=9.0]",
        retrieved_timestamps=[1.0],
        required_keywords=["person"],
        forbidden_keywords=["cat"],
        requires_uncertainty=False,
    )

    assert not result["passed"]
    assert not result["hallucination_free"]
    assert not result["citation_correct"]


def test_audiovisual_answer_scores_keywords_and_citation_kinds():
    result = evaluate_audiovisual_answer(
        answer=(
            "The lecturer compares REINFORCE and A2C. "
            "[TRANSCRIPT:t=10.0-14.0] The slide shows TD target. [SLIDE:t=72.0]"
        ),
        expected_keywords=["REINFORCE", "A2C", "TD target"],
        forbidden_keywords=["DQN"],
        expected_citation_min=2,
        expected_citation_kinds=["transcript", "slide"],
    )

    assert result["passed"]
    assert result["citation_counts"] == {"frame": 0, "transcript": 1, "slide": 1}
    assert result["citation_kind_coverage"] == {"transcript": True, "slide": True}


def test_audiovisual_answer_accepts_frame_or_slide_citation_kind():
    frame_result = evaluate_audiovisual_answer(
        answer="The speaker wears a green plaid shirt. [FRAME:t=19.0]",
        expected_keywords=["green", "plaid", "shirt"],
        forbidden_keywords=[],
        expected_citation_min=1,
        expected_citation_kinds=["frame_or_slide"],
    )
    slide_result = evaluate_audiovisual_answer(
        answer="The answer is American Express. [SLIDE:t=20.0]",
        expected_keywords=["American", "Express"],
        forbidden_keywords=[],
        expected_citation_min=1,
        expected_citation_kinds=["frame_or_slide"],
    )

    assert frame_result["passed"]
    assert slide_result["passed"]
    assert frame_result["citation_kind_coverage"] == {"frame_or_slide": True}


def test_audiovisual_case_uses_question_id_schema_and_llm_judge():
    case = EvalCase(
        case_id="av-1",
        video_id="v",
        question="Which methods are compared?",
        reference_answer="It compares REINFORCE and A2C.",
        modality_tag="audio",
        expected_keywords=["REINFORCE", "A2C"],
        expected_citation_min=1,
        expected_citation_kinds=["transcript"],
    )
    prediction = EvalPrediction(
        case_id="av-1",
        answer="It compares REINFORCE and A2C. [TRANSCRIPT:t=10.0-14.0]",
    )

    class ExplodingJudge:
        model = "unused"

        def grade(self, *, question, reference, answer):
            return {"correct": True, "score": 5, "justification": "ok"}

    result = evaluate_case(case, prediction, judge=ExplodingJudge())
    assert result["passed"]
    assert result["answer"]["llm_judge"]["correct"] is True


def test_parse_audiovisual_jsonl_schema(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(
            {
                "question_id": "av-schema-1",
                "video_id": "vid",
                "question": "What is on the slide?",
                "modality_tag": "joint",
                "question_type": "open",
                "expected_keywords": ["A2C"],
                "expected_citation_min": 2,
                "expected_citation_kinds": ["transcript", "slide"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    case = load_cases(path)[0]
    assert case.case_id == "av-schema-1"
    assert case.required_keywords == ["A2C"]
    assert case.expected_keywords == ["A2C"]
    assert case.expected_citation_kinds == ["transcript", "slide"]


def test_extract_evidence_markers_preserves_order():
    markers = extract_evidence_markers(
        "see [SLIDE:t=20.0] then [TRANSCRIPT:t=10.0-12.0] and [FRAME:t=11.0]"
    )

    assert [marker["kind"] for marker in markers] == ["slide", "transcript", "frame"]


def test_agent_loop_harness_checks_recommended_action():
    result = evaluate_agent_loop(
        agent_actions=["retrieve_video_evidence", "expand_temporal_evidence"],
        expected_action="retrieve_video_evidence",
        needs_expansion=True,
        evidence_sufficiency={
            "sufficient": False,
            "recommended_next_action": "expand_temporal_evidence",
        },
    )

    assert result["passed"]


def test_eval_case_and_summary_from_jsonl_fixtures():
    cases = load_cases("tests/fixtures/eval_cases.jsonl")
    predictions = load_predictions("tests/fixtures/eval_predictions.jsonl")

    results = [
        evaluate_case(case, predictions[case.case_id], tolerance_sec=2.0)
        for case in cases
    ]
    summary = summarize_results(results)

    assert summary["total"] == 2
    assert summary["passed"] == 2
    assert summary["pass_rate"] == 1.0


def test_summarize_results_grouped_by_prefix():
    results = [
        {
            "case_id": "longvideobench-1",
            "passed": True,
            "retrieval": {"passed": True, "recall_at_k": 1.0, "timestamp_distance": None},
            "answer": {"passed": True},
            "agent_loop": {"passed": True},
        },
        {
            "case_id": "longvideobench-2",
            "passed": False,
            "retrieval": {"passed": False, "recall_at_k": 0.5, "timestamp_distance": 3.0},
            "answer": {"passed": True},
            "agent_loop": {"passed": True},
        },
        {
            "case_id": "nextgqa-x",
            "passed": True,
            "retrieval": {"passed": True, "recall_at_k": 1.0, "timestamp_distance": 0.5},
            "answer": {"passed": True},
            "agent_loop": {"passed": True},
        },
    ]
    summary = summarize_results(results, group_by=group_by_case_prefix)
    assert summary["total"] == 3
    assert summary["passed"] == 2
    assert "groups" in summary
    assert summary["groups"]["longvideobench"]["total"] == 2
    assert summary["groups"]["longvideobench"]["pass_rate"] == 0.5
    assert summary["groups"]["nextgqa"]["pass_rate"] == 1.0


def test_evaluate_case_with_mock_judge_overrides_keyword_failure(monkeypatch):
    case = EvalCase(
        case_id="judge-1",
        video_id="v",
        question="Q",
        reference_answer="The cat sleeps on the couch.",
        required_keywords=["dog"],  # intentionally wrong keyword
    )
    prediction = EvalPrediction(
        case_id="judge-1",
        retrieved_timestamps=[1.0],
        answer="A cat is sleeping on the sofa. [FRAME:t=1.0]",
    )

    class FakeJudge:
        model = "fake"

        def grade(self, *, question, reference, answer):
            return {"correct": True, "score": 5, "justification": "semantic match"}

    result = evaluate_case(case, prediction, judge=FakeJudge())
    assert result["answer"]["llm_judge"]["correct"] is True
    assert result["answer"]["passed"] is True
    assert result["passed"] is True


def test_evaluate_case_persists_prediction_text_and_actions():
    case = EvalCase(case_id="trace-1", video_id="v", question="Q", gold_timestamps=[1.0])
    prediction = EvalPrediction(
        case_id="trace-1",
        retrieved_timestamps=[1.0],
        answer="A short reply [FRAME:t=1.0]",
        agent_actions=["retrieve_video_evidence", "answer_with_evidence"],
    )
    result = evaluate_case(case, prediction)
    assert result["prediction_text"] == "A short reply [FRAME:t=1.0]"
    assert result["retrieved_timestamps"] == [1.0]
    assert result["agent_actions"] == ["retrieve_video_evidence", "answer_with_evidence"]


def test_evaluate_case_adds_failure_tags_for_mcq_and_post_answer_retrieval():
    case = EvalCase(
        case_id="av-fail-tags",
        video_id="v",
        question=(
            "What is the correct order?\n(a) Cats.\n(b) Dogs.\n\n"
            "Candidates:\nA) (a)(b).\nB) (b)(a)."
        ),
        reference_answer="(a)(b).",
        expected_keywords=["(a)(b)"],
        expected_citation_min=1,
        expected_citation_kinds=["frame_or_slide"],
        question_type="temporal reasoning",
    )
    prediction = EvalPrediction(
        case_id="av-fail-tags",
        answer="Answer: B) (b)(a). [TRANSCRIPT:t=1.0-2.0]",
        agent_actions=["retrieve_transcript_evidence", "answer_with_evidence", "retrieve_video_evidence"],
    )

    class FakeJudge:
        model = "fake"

        def grade(self, *, question, reference, answer):
            return {"correct": False, "score": 0, "justification": "wrong option"}

    result = evaluate_case(case, prediction, judge=FakeJudge())
    assert "wrong_temporal_option" in result["failure_tags"]
    assert "temporal_order_error" in result["failure_tags"]
    assert "missing_frame_or_slide" in result["failure_tags"]
    assert "post_answer_retrieval" in result["failure_tags"]
    assert result["selected_option"]["label"] == "B"
    assert result["recommended_option"]["label"] == "A"


def test_summary_and_markdown_include_failure_tags(tmp_path):
    results = [
        {
            "case_id": "vme-1",
            "question": "Q",
            "passed": False,
            "failure_tags": ["wrong_fact_option", "missing_slide"],
            "retrieval": {"passed": None, "recall_at_k": None, "timestamp_distance": None},
            "answer": {"passed": False},
            "agent_loop": {"passed": True},
        }
    ]
    summary = summarize_results(results)
    assert summary["failure_tag_counts"] == {"missing_slide": 1, "wrong_fact_option": 1}

    report = {"summary": summary, "missing_predictions": [], "results": results}
    out = tmp_path / "report.md"
    write_markdown_report(report, out)
    text = out.read_text(encoding="utf-8")
    assert "## Failure tags" in text
    assert "`wrong_fact_option`" in text
    assert "missing_slide" in text


def test_evaluate_case_tags_brand_guess_separately():
    case = EvalCase(
        case_id="brand-1",
        video_id="v",
        question="What is the recommended brand?\n\nCandidates:\nA) SNEAKER LAB.\nB) JASON MARKK.",
        reference_answer="SNEAKER LAB.",
        expected_keywords=["SNEAKER", "LAB"],
        expected_citation_kinds=["slide"],
    )
    prediction = EvalPrediction(
        case_id="brand-1",
        answer="Answer: B) JASON MARKK. This is typical of the standard product. [FRAME:t=1.0]",
    )

    class FakeJudge:
        model = "fake"

        def grade(self, *, question, reference, answer):
            return {"correct": False, "score": 0, "justification": "wrong"}

    result = evaluate_case(case, prediction, judge=FakeJudge())
    assert "unsupported_visual_brand_guess" in result["failure_tags"]
    assert result["recommended_option"]["label"] == "A"


def test_evaluate_case_soft_waives_citation_when_judge_passes():
    case = EvalCase(
        case_id="soft-1",
        video_id="v",
        question="Q",
        reference_answer="The cat sleeps.",
    )
    prediction = EvalPrediction(
        case_id="soft-1",
        retrieved_timestamps=[1.0],
        answer="A cat is sleeping.",  # NO [FRAME:...] marker; would normally fail citation
    )

    class FakeJudge:
        model = "fake"

        def grade(self, *, question, reference, answer):
            return {"correct": True, "score": 5, "justification": "ok"}

    result = evaluate_case(case, prediction, judge=FakeJudge())
    assert result["answer"]["citation_correct"] is False  # raw signal preserved
    assert result["answer"]["citation_soft_waived"] is True
    assert result["answer"]["passed"] is True
    assert result["passed"] is True


def test_evaluate_case_soft_waives_retrieval_when_judge_passes():
    case = EvalCase(
        case_id="ret-soft-1",
        video_id="v",
        question="Q",
        reference_answer="ok",
        gold_timestamps=[10.0, 20.0],  # neither will match — retrieval=FAIL
    )
    prediction = EvalPrediction(
        case_id="ret-soft-1",
        retrieved_timestamps=[50.0, 60.0],  # 30s+ away from both gold
        answer="An answer with citation [FRAME:t=50.0].",
    )

    class FakeJudge:
        model = "fake"

        def grade(self, *, question, reference, answer):
            return {"correct": True, "score": 5, "justification": "ok"}

    result = evaluate_case(case, prediction, judge=FakeJudge(), tolerance_sec=2.0)
    assert result["retrieval"]["passed"] is False  # strict signal preserved
    assert result["retrieval"]["soft_waived"] is True
    assert result["passed"] is True


def test_evaluate_case_distill_strict_disables_soft_waive():
    case = EvalCase(
        case_id="strict-soft-1",
        video_id="v",
        question="Q",
        reference_answer="ok",
        gold_timestamps=[10.0],
    )
    prediction = EvalPrediction(
        case_id="strict-soft-1",
        retrieved_timestamps=[50.0],
        answer="Semantically ok but no valid citation.",
    )

    class FakeJudge:
        model = "fake"

        def grade(self, *, question, reference, answer):
            return {"correct": True, "score": 5, "justification": "ok"}

    result = evaluate_case(
        case,
        prediction,
        judge=FakeJudge(),
        tolerance_sec=2.0,
        distill_strict=True,
    )

    assert result["answer"]["llm_judge"]["correct"] is True
    assert result["answer"]["citation_soft_waived"] is False
    assert result["retrieval"]["soft_waived"] is False
    assert result["passed"] is False


def test_evaluate_case_keeps_retrieval_strict_when_judge_fails():
    case = EvalCase(
        case_id="ret-strict-1",
        video_id="v",
        question="Q",
        reference_answer="ok",
        gold_timestamps=[10.0],
    )
    prediction = EvalPrediction(
        case_id="ret-strict-1",
        retrieved_timestamps=[50.0],
        answer="Wrong answer.",
    )

    class FakeJudge:
        model = "fake"

        def grade(self, *, question, reference, answer):
            return {"correct": False, "score": 1, "justification": "wrong"}

    result = evaluate_case(case, prediction, judge=FakeJudge(), tolerance_sec=2.0)
    assert result["retrieval"]["passed"] is False
    assert result["retrieval"]["soft_waived"] is False
    assert result["passed"] is False


def test_evaluate_retrieval_returns_none_when_no_gold():
    result = evaluate_retrieval(
        gold_timestamps=[],
        retrieved_timestamps=[1.0, 2.0],
        gold_scenes=[],
        scene_hits=[],
        tolerance_sec=1.0,
        recall_k=None,
    )
    assert result["passed"] is None


def test_section_pass_rate_excludes_na_cases():
    # 2 cases: one with retrieval n/a (passed=None), one with retrieval passed=True.
    results = [
        {
            "case_id": "lvb-1",
            "passed": True,
            "retrieval": {"passed": None, "recall_at_k": None, "timestamp_distance": None},
            "answer": {"passed": True},
            "agent_loop": {"passed": True},
        },
        {
            "case_id": "ngqa-1",
            "passed": True,
            "retrieval": {"passed": True, "recall_at_k": 1.0, "timestamp_distance": 0.1},
            "answer": {"passed": True},
            "agent_loop": {"passed": True},
        },
    ]
    summary = summarize_results(results)
    # 1/1 applicable case retrieval-passed → 1.0 (n/a excluded from denominator)
    assert summary["retrieval_pass_rate"] == 1.0


def test_prediction_cache_round_trip(tmp_path):
    cache = PredictionCache(tmp_path / "pred.jsonl")
    key = PredictionCache.make_key(
        case_id="c-1",
        model="m",
        prompt_fingerprint="fp",
        video_id="v",
        agent_code_version="v1",
    )
    assert cache.get(key) is None

    pred = EvalPrediction(
        case_id="c-1",
        retrieved_timestamps=[1.0, 2.5],
        scene_hits=[{"start": 0.0, "end": 3.0}],
        retrieved_transcripts=[{"t_start": 1.0, "t_end": 2.0, "text": "hello"}],
        retrieved_slides=[{"t_start": 2.0, "t_end": 2.0, "text": "slide"}],
        answer="hello [FRAME:t=1.0]",
        agent_actions=["retrieve_video_evidence"],
        evidence_sufficiency={"sufficient": True},
        grounding_report={"grounded": True},
    )
    cache.put(key, PredictionCache.prediction_to_dict(pred))

    reloaded = PredictionCache(tmp_path / "pred.jsonl")
    cached = reloaded.get(key)
    assert cached is not None
    restored = PredictionCache.prediction_from_dict("c-1", cached)
    assert restored.retrieved_timestamps == [1.0, 2.5]
    assert restored.scene_hits == [{"start": 0.0, "end": 3.0}]
    assert restored.retrieved_transcripts == [{"t_start": 1.0, "t_end": 2.0, "text": "hello"}]
    assert restored.retrieved_slides == [{"t_start": 2.0, "t_end": 2.0, "text": "slide"}]
    assert restored.answer == "hello [FRAME:t=1.0]"
    assert restored.agent_actions == ["retrieve_video_evidence"]
    assert restored.evidence_sufficiency == {"sufficient": True}
    assert restored.grounding_report == {"grounded": True}


def test_prediction_cache_key_invalidates_on_prompt_change():
    base = dict(case_id="c", model="m", video_id="v", agent_code_version="v1")
    k1 = PredictionCache.make_key(prompt_fingerprint="aaa", **base)
    k2 = PredictionCache.make_key(prompt_fingerprint="bbb", **base)
    assert k1 != k2


def test_judge_cache_round_trip(tmp_path):
    cache = JudgeCache(tmp_path / "judge.jsonl")
    assert cache.get("case-1", "model", "answer") is None
    cache.put("case-1", "model", "answer", {"correct": True, "score": 5, "justification": "ok"})

    reloaded = JudgeCache(tmp_path / "judge.jsonl")
    cached = reloaded.get("case-1", "model", "answer")
    assert cached is not None
    assert cached["correct"] is True
    assert cached["score"] == 5


def test_parse_judge_json_handles_fenced_and_noisy_output():
    fenced = """```json
{"correct": true, "score": 4, "justification": "close enough"}
```"""
    parsed = _parse_judge_json(fenced)
    assert parsed["correct"] is True
    assert parsed["score"] == 4

    noisy = 'Sure! Here is my grade: {"correct": false, "score": 1, "justification": "off"} thanks!'
    parsed2 = _parse_judge_json(noisy)
    assert parsed2["correct"] is False
    assert parsed2["score"] == 1

    broken = "this is not json at all"
    parsed3 = _parse_judge_json(broken)
    assert parsed3["correct"] is False
    assert "error" in parsed3


def test_write_markdown_report_contains_global_and_groups(tmp_path):
    results = [
        {
            "case_id": "longvideobench-1",
            "question": "What happens?",
            "passed": True,
            "retrieval": {"passed": True, "recall_at_k": None, "timestamp_distance": None},
            "answer": {"passed": True},
            "agent_loop": {"passed": True},
        },
        {
            "case_id": "nextgqa-2",
            "question": "When does X occur?",
            "passed": False,
            "retrieval": {"passed": False, "recall_at_k": 0.0, "timestamp_distance": 5.0},
            "answer": {"passed": False},
            "agent_loop": {"passed": True},
        },
    ]
    summary = summarize_results(results, group_by=group_by_case_prefix)
    report = {"summary": summary, "missing_predictions": [], "results": results}
    out = tmp_path / "report.md"
    write_markdown_report(report, out)
    text = out.read_text(encoding="utf-8")
    assert "# Mr. Big-Eye eval report" in text
    assert "## Global summary" in text
    assert "## Per-group summary" in text
    assert "longvideobench" in text
    assert "nextgqa" in text
    assert "Worst" in text  # failure section header


def test_eval_harness_module_cli_audiovisual_predictions_smoke(tmp_path):
    cases = tmp_path / "cases.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    output = tmp_path / "report.json"
    cases.write_text(
        json.dumps(
            {
                "question_id": "av-cli-1",
                "video_id": "vid",
                "question": "Which algorithms are compared?",
                "modality_tag": "audio",
                "question_type": "open",
                "expected_keywords": ["REINFORCE", "A2C"],
                "expected_citation_min": 1,
                "expected_citation_kinds": ["transcript"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    predictions.write_text(
        json.dumps(
            {
                "case_id": "av-cli-1",
                "answer": "The lecture compares REINFORCE and A2C. [TRANSCRIPT:t=10.0-12.0]",
                "agent_actions": ["retrieve_transcript_evidence", "answer_with_evidence"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.eval_harness",
            "--dataset",
            "audiovisual",
            "--cases",
            str(cases),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
            "--n",
            "20",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["passed"] == 1
