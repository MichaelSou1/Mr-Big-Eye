"""ASR module using SenseVoice-Small + FSMN-VAD (Chinese-optimized, ~1GB VRAM).

Exposes `transcribe(video_path) -> list[dict]` returning sentence-level
segments with timestamps:
    [{"text": "...", "t_start": float_seconds, "t_end": float_seconds}, ...]

Two-stage pipeline: FSMN-VAD slices the audio into speech segments with
timestamps, then SenseVoice-Small transcribes each slice. We need the
explicit VAD step because SenseVoice-Small does not emit token-level
timestamps on its own — only `text` for the whole input.

CLI smoke test:
    python -m app.asr path/to/video.mp4
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_ffmpeg_on_path() -> None:
    """funasr shells out to `ffmpeg` from PATH. Fall back to the bundled
    imageio-ffmpeg binary when the system has no ffmpeg installed."""
    if shutil.which("ffmpeg") is not None:
        return
    try:
        import imageio_ffmpeg
    except ImportError:
        return
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    bin_dir = os.path.dirname(ffmpeg_path)
    canonical = os.path.join(bin_dir, "ffmpeg")
    if not os.path.exists(canonical):
        try:
            os.symlink(ffmpeg_path, canonical)
        except OSError:
            pass
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


_ensure_ffmpeg_on_path()


_VAD: Any = None
_ASR: Any = None
ASR_MODEL_NAME = "iic/SenseVoiceSmall"

_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+"
)

_TERM_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\ba\s*(?:二|2|two)\s*c\b", re.IGNORECASE), "A2C"),
    (re.compile(r"\bA\s*(?:二|2|two)\s*C\b", re.IGNORECASE), "A2C"),
    (re.compile(r"\brl\b", re.IGNORECASE), "RL"),
    (re.compile(r"\breinforce\b", re.IGNORECASE), "REINFORCE"),
    (re.compile(r"\btd\b", re.IGNORECASE), "TD"),
)


def _load_models() -> tuple[Any, Any]:
    """Lazy-load VAD + ASR. First call downloads ~250MB to ~/.cache/modelscope."""
    global _VAD, _ASR
    if _VAD is not None and _ASR is not None:
        return _VAD, _ASR

    import torch
    from funasr import AutoModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Loading FSMN-VAD on %s ...", device)
    _VAD = AutoModel(
        model="fsmn-vad",
        device=device,
        disable_update=True,
        log_level="ERROR",
    )
    logger.info("Loading SenseVoice-Small on %s ...", device)
    _ASR = AutoModel(
        model=ASR_MODEL_NAME,
        device=device,
        disable_update=True,
        log_level="ERROR",
    )
    return _VAD, _ASR


def _extract_audio(video_path: Path, dest: Path) -> None:
    """Extract 16kHz mono wav using whatever ffmpeg is on PATH."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000", "-vn",
        "-loglevel", "error",
        str(dest),
    ]
    subprocess.run(cmd, check=True)


_MIN_SEGMENT_MS = 200  # drop ultra-short noise blips


def transcribe(
    video_path: str | os.PathLike[str],
    *,
    language: str = "auto",
    hotwords: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Transcribe a video file to sentence-level segments.

    Args:
        video_path: path to video file (any container ffmpeg can read).
        language: 'auto' | 'zh' | 'en' | 'yue' | 'ja' | 'ko'.
        hotwords: Optional course/domain terms forwarded to SenseVoice.

    Returns:
        List of segments [{"text": str, "t_start": float, "t_end": float}, ...]
        in seconds, sorted by t_start. Empty segments and ultra-short
        sub-200ms VAD chunks are dropped.
    """
    import soundfile as sf
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    vad, asr = _load_models()
    hotword_text = " ".join(str(item).strip() for item in (hotwords or []) if str(item).strip())

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        _extract_audio(video_path, wav_path)
        audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    vad_res = vad.generate(input=audio, fs=sr)
    vad_segments: list[list[int]] = []
    for entry in vad_res:
        vad_segments.extend(entry.get("value", []) or [])

    segments: list[dict[str, Any]] = []
    for t_start_ms, t_end_ms in vad_segments:
        if t_end_ms - t_start_ms < _MIN_SEGMENT_MS:
            continue
        start_idx = int(t_start_ms * sr / 1000)
        end_idx = int(t_end_ms * sr / 1000)
        chunk = audio[start_idx:end_idx]
        if chunk.size == 0:
            continue

        kwargs: dict[str, Any] = {
            "input": chunk,
            "fs": sr,
            "cache": {},
            "language": language,
            "use_itn": True,
        }
        if hotword_text:
            kwargs["hotword"] = hotword_text
        asr_res = asr.generate(**kwargs)
        raw_text = (asr_res[0].get("text", "") or "") if asr_res else ""
        text = postprocess_text(rich_transcription_postprocess(raw_text))
        if not text:
            continue
        segments.append({
            "text": text,
            "t_start": round(t_start_ms / 1000.0, 3),
            "t_end": round(t_end_ms / 1000.0, 3),
            "source": "asr",
        })

    segments.sort(key=lambda seg: seg["t_start"])
    return segments


def postprocess_text(text: str) -> str:
    """Normalize SenseVoice text for downstream retrieval."""
    cleaned = _EMOJI_RE.sub("", text or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for pattern, replacement in _TERM_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned.strip()


def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("usage: python -m app.asr <video_path>", file=sys.stderr)
        raise SystemExit(2)
    video = sys.argv[1]
    import time
    t0 = time.time()
    segments = transcribe(video)
    elapsed = time.time() - t0
    for seg in segments:
        print(f"[{seg['t_start']:7.2f} - {seg['t_end']:7.2f}] {seg['text']}")
    duration = segments[-1]["t_end"] if segments else 0.0
    rt = duration / elapsed if elapsed > 0 else 0.0
    print(
        f"\n--- {len(segments)} segments, "
        f"audio={duration:.1f}s, wall={elapsed:.1f}s, RTF={rt:.1f}x ---",
        file=sys.stderr,
    )


if __name__ == "__main__":
    _cli()
