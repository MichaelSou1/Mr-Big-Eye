import logging
import gc
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from app.config import settings

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when a model singleton is accessed before startup loading."""


def _model_source(local_dir: Path, fallback_name: str) -> str:
    return str(local_dir) if local_dir.exists() and any(local_dir.iterdir()) else fallback_name


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return device


def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
    normalized = dtype.lower()
    if normalized == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def _normalize_np(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (values / norms).astype(np.float32)


class BgeM3Wrapper:
    """Wraps FlagEmbedding.BGEM3FlagModel for dense text embeddings."""

    def __init__(self, model_name: str, device: str, batch_size: int = 64) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self.device = _resolve_device(device)
        self.batch_size = batch_size
        source = _model_source(settings.bge_m3_local_dir, model_name)
        use_fp16 = self.device.startswith("cuda")
        logger.info("Loading bge-m3 from %s on %s", source, self.device)
        # Pin a SINGLE device. FlagEmbedding's `devices` kwarg (plural) controls
        # this; when omitted it auto-detects ALL GPUs, making encode() spawn a
        # multi-process pool whose model.share_memory() fails in this env
        # ("random_device could not be read"). Pinning one device keeps encode()
        # on the single-device path. Tolerate older signatures (`device=`/none).
        try:
            self.model = BGEM3FlagModel(source, use_fp16=use_fp16, devices=self.device)
        except TypeError:
            try:
                self.model = BGEM3FlagModel(source, use_fp16=use_fp16, device=self.device)
            except TypeError:
                self.model = BGEM3FlagModel(source, use_fp16=use_fp16)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        output = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"] if isinstance(output, dict) else output
        return _normalize_np(np.asarray(dense, dtype=np.float32))


class Siglip2Wrapper:
    """Wraps a SigLIP2 image-text encoder and returns normalized vectors."""

    def __init__(self, model_name: str, device: str, batch_size: int = 32) -> None:
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.dtype = _resolve_dtype(settings.siglip2_dtype, self.device)
        source = _model_source(settings.siglip2_local_dir, model_name)
        logger.info("Loading SigLIP2 from %s on %s (%s)", source, self.device, self.dtype)
        self.processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            source,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def encode_image(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        chunks: list[torch.Tensor] = []
        for start in range(0, len(images), self.batch_size):
            batch = [img.convert("RGB") for img in images[start : start + self.batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad(), self._autocast_context():
                if hasattr(self.model, "get_image_features"):
                    output = self.model.get_image_features(**inputs)
                else:
                    output = self.model(**inputs)
                features = (
                    output
                    if isinstance(output, torch.Tensor)
                    else self._extract_features(output, "image")
                )
                chunks.append(F.normalize(features.float(), p=2, dim=-1).cpu())
        return torch.cat(chunks, dim=0).numpy().astype(np.float32)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        chunks: list[torch.Tensor] = []
        for start in range(0, len(texts), 64):
            batch = texts[start : start + 64]
            inputs = self.processor(
                text=batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad(), self._autocast_context():
                if hasattr(self.model, "get_text_features"):
                    output = self.model.get_text_features(**inputs)
                else:
                    output = self.model(**inputs)
                features = (
                    output
                    if isinstance(output, torch.Tensor)
                    else self._extract_features(output, "text")
                )
                chunks.append(F.normalize(features.float(), p=2, dim=-1).cpu())
        return torch.cat(chunks, dim=0).numpy().astype(np.float32)

    def _autocast_context(self):
        if self.device.startswith("cuda"):
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @staticmethod
    def _extract_features(outputs, prefix: str) -> torch.Tensor:
        for attr in (f"{prefix}_embeds", "pooler_output"):
            value = getattr(outputs, attr, None)
            if value is not None:
                return value
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is not None:
            return last_hidden[:, 0]
        raise RuntimeError(f"Could not find {prefix} features in model output")


_bge: BgeM3Wrapper | None = None
_siglip: Siglip2Wrapper | None = None


def load_all_models() -> None:
    """Call once at app startup."""
    global _bge, _siglip
    if _bge is None:
        _bge = BgeM3Wrapper(settings.bge_m3_model, settings.models_device)
    if _siglip is None:
        _siglip = Siglip2Wrapper(settings.siglip2_model, settings.models_device)


def get_bge() -> BgeM3Wrapper:
    """Returns loaded bge-m3 wrapper."""
    global _bge
    if _bge is None:
        _bge = BgeM3Wrapper(settings.bge_m3_model, settings.models_device)
    return _bge


def get_siglip() -> Siglip2Wrapper:
    """Returns loaded SigLIP2 wrapper."""
    global _siglip
    if _siglip is None:
        _siglip = Siglip2Wrapper(settings.siglip2_model, settings.models_device)
    return _siglip


def get_text_embed() -> BgeM3Wrapper:
    """Transcript/slide dense search reuses the bge-m3 singleton."""
    return get_bge()


def release_bge() -> None:
    """Drop the bge-m3 singleton for low-memory local runs."""
    global _bge
    _bge = None
    _cleanup_memory()


def release_siglip() -> None:
    """Drop the SigLIP2 singleton for low-memory local runs."""
    global _siglip
    _siglip = None
    _cleanup_memory()


def release_text_embed() -> None:
    """Text embedding shares the bge-m3 singleton; defer to release_bge."""
    release_bge()


def _cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
