from app.distill.eval_common import grade_textaware, relaxed_match, config_fingerprint
from app.distill.eval_stats import paired_bootstrap_net, min_detectable_net, mcnemar


Q = "Q\n\nCandidates:\nA) red ball\nB) blue box\nC) green hat\n"


def test_grade_textaware_flags_letter_luck():
    # letter says B but the prose describes option A (red ball) -> letter-luck -> incorrect
    r = grade_textaware(Q, "B", "The correct answer is B) red ball, clearly a red ball.")
    assert r["letter_luck"] is True
    assert r["correct"] is False
    assert r["letter_ok"] is True


def test_grade_textaware_accepts_consistent_answer():
    r = grade_textaware(Q, "B", "The answer is B) blue box — a blue box is visible.")
    assert r["correct"] is True
    assert r["letter_luck"] is False


def test_grade_textaware_open_ended_relaxed():
    assert grade_textaware("How many?", "10", "about 10.2 items")["correct"] is True
    assert grade_textaware("How many?", "10", "around 25")["correct"] is False


def test_relaxed_match_numeric_tolerance():
    assert relaxed_match("the value is 103.7", "104") is True   # within 5%
    assert relaxed_match("the value is 80", "104") is False


def test_paired_bootstrap_excludes_zero_for_clear_effect():
    free = [0] * 100
    method = [1] * 100  # method always right, free always wrong -> net=+1, CI excludes 0
    r = paired_bootstrap_net(free, method, B=2000, seed=1)
    assert r["net"] == 1.0 and r["excludes_0"] is True and r["gain"] == 100


def test_paired_bootstrap_within_variance_for_noise():
    free = [1, 0] * 50
    method = [0, 1] * 50  # net 0, lots of discordance -> CI includes 0
    r = paired_bootstrap_net(free, method, B=2000, seed=1)
    assert r["excludes_0"] is False


def test_min_detectable_net_shrinks_with_n():
    assert min_detectable_net(0.65, 70) > min_detectable_net(0.65, 300)


def test_fingerprint_is_stable_and_hashed():
    kw = dict(dataset="next", split_hash="abc", model_id="8b", method="free_form",
              n_frames=16, temperature=0.0, top_p=1.0, max_tokens=512, seed=0)
    a = config_fingerprint(**kw); b = config_fingerprint(**kw)
    assert a["fp"] == b["fp"] and len(a["fp"]) == 12
