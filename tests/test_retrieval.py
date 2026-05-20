import pytest

from app.config import settings
from app.retrieval import two_stage_retrieve


@pytest.mark.skipif(
    not (settings.data_dir / "cache" / "test001" / ".done").exists(),
    reason="requires a preprocessed test001 video cache",
)
def test_two_stage_retrieve_returns_sorted_frames():
    result = two_stage_retrieve("test001", "what is happening")

    assert len(result.frames) == settings.top_k_frames
    assert result.timestamps == sorted(result.timestamps)
