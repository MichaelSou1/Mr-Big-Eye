from app.distill_filter import RUNTIME_GUARDS, classify_tier


def test_tier_1_clean_correct():
    traj = {"judge_correct": True, "agent_terminated": None, "guards_triggered": []}
    assert classify_tier(traj) == "tier_1"


def test_tier_1_allows_forced_calls():
    # forced_answer / forced_verify are normal STOP DISCIPLINE, not runtime guards.
    traj = {"judge_correct": True, "guards_triggered": ["forced_answer", "forced_verify"]}
    assert classify_tier(traj) == "tier_1"


def test_tier_2_runtime_guard():
    for guard in sorted(RUNTIME_GUARDS):
        traj = {"judge_correct": True, "guards_triggered": [guard]}
        # cap/empty are demoted to tier_3 only via agent_terminated; as a bare
        # guard label they still mark tier_2.
        assert classify_tier(traj) == "tier_2", guard


def test_tier_3_judge_wrong():
    assert classify_tier({"judge_correct": False, "guards_triggered": []}) == "tier_3"
    assert classify_tier({"judge_correct": None}) == "tier_3"


def test_tier_3_terminated():
    assert classify_tier({"judge_correct": True, "agent_terminated": "cap"}) == "tier_3"
    assert classify_tier({"judge_correct": True, "agent_terminated": "empty"}) == "tier_3"
