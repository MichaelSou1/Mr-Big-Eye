import pytest

from app.config import settings
from app.retrieval import _scene_quota


def test_scene_quota_reserves_third_for_global_when_scenes_present():
    # top_k=12, scenes present → 8 scene-gated, 4 un-gated (filled by fallback)
    assert _scene_quota(top_k=12, time_ranges=[(0.0, 5.0)]) == 8


def test_scene_quota_full_budget_when_no_scenes():
    # No scenes routed → no point reserving; spend the whole budget un-gated
    assert _scene_quota(top_k=12, time_ranges=[]) == 12


def test_scene_quota_floor_at_one_for_tiny_top_k():
    # top_k=1 with scenes → at least one scene-gated frame
    assert _scene_quota(top_k=1, time_ranges=[(0.0, 5.0)]) == 1


def test_scene_quota_uneven_split_for_small_top_k():
    # top_k=4 with scenes → 2 scene-gated, 2 un-gated
    assert _scene_quota(top_k=4, time_ranges=[(0.0, 5.0)]) == 2


@pytest.mark.skipif(
    not (settings.data_dir / "cache" / "test001" / ".done").exists(),
    reason="requires a preprocessed test001 video cache",
)
def test_two_stage_retrieve_returns_sorted_frames():
    from app.retrieval import two_stage_retrieve

    result = two_stage_retrieve("test001", "what is happening")

    assert len(result.frames) == settings.top_k_frames
    assert result.timestamps == sorted(result.timestamps)
