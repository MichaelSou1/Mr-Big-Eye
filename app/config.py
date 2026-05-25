from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # VLM API (multimodal Q&A and captioning).
    # Two formats are supported:
    #   - "responses": Doubao / Volcano Ark Responses API
    #     (POST {base_url}/responses with input=[{role, content:[{type:input_image|input_text}]}])
    #   - "chat_completions": OpenAI-compatible Chat Completions
    #     (POST {base_url}/chat/completions, used by Xiaomi MiMo and others)
    vlm_api_provider: str = "doubao"
    vlm_api_format: Literal["responses", "chat_completions"] = "responses"
    vlm_api_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    vlm_api_key: str = ""
    vlm_model_name: str = "doubao-seed-2-0-pro-260215"
    vlm_api_timeout: int = 120

    # Tool-aware text orchestrator. It must speak an OpenAI-compatible
    # Chat Completions API because LangChain binds tools through that surface.
    # Empty values default to the VLM API settings above.
    orchestrator_api_base_url: str = ""
    orchestrator_api_key: str = ""
    orchestrator_model_name: str = ""
    orchestrator_api_timeout: int | None = None
    orchestrator_temperature: float = 0.2
    orchestrator_max_tool_calls: int = 8
    # Some OpenAI-compatible proxies refuse streaming for certain models
    # (e.g. "没有可用的 Provider 支持模型 X 的流式请求"). Default off so eval
    # works against the widest set of providers; UX path doesn't depend on
    # token-by-token streaming from the orchestrator.
    orchestrator_streaming: bool = False

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    max_video_duration_sec: int = 600
    max_upload_size_mb: int = 2048
    load_models_on_startup: bool = True
    unload_models_after_use: bool = False
    app_cuda_visible_devices: str = ""
    session_title_max_chars: int = 48

    bge_m3_model: str = "BAAI/bge-m3"
    bge_m3_local_dir: Path = Path("./models/bge-m3")
    siglip2_model: str = "google/siglip2-so400m-patch14-384"
    siglip2_modelscope_model: str = "google/siglip2-so400m-patch14-384"
    siglip2_local_dir: Path = Path("./models/siglip2-so400m-patch14-384")
    siglip2_dtype: str = "auto"
    models_device: str = "cuda:0"

    top_n_scenes: int = 5
    top_k_frames: int = 12
    planner_max_top_n_scenes: int = 12
    planner_max_top_k_frames: int = 36
    vqa_max_frames: int = 6
    vqa_max_image_side: int = 448
    vqa_image_quality: int = 75
    vqa_max_output_tokens: int = 1024
    scene_detect_threshold: float = 27.0
    dense_fps: float = 1.0

    data_dir: Path = Path("./data")
    database_path: Path | None = None
    graph_checkpoint_path: Path | None = None
    langmem_store_path: Path | None = None
    # LangMem (text-only) defaults to the VLM API. Override to point at a
    # cheaper / smaller chat-completions endpoint if desired.
    langmem_api_base_url: str = ""
    langmem_api_key: str = ""
    langmem_model_name: str = ""
    langmem_query_limit: int = 6
    progress_lang: Literal["zh", "en"] = "zh"
    log_level: str = "INFO"

    # Eval LLM judge (text-only). Used by scripts/eval_harness.py --judge.
    # If judge_api_key is blank, callers should fall back to vlm_api_* with a
    # self-judging-bias warning. If both are blank, judging is disabled.
    judge_api_base_url: str = ""
    judge_api_key: str = ""
    judge_model_name: str = ""
    judge_api_timeout: int = 120
    judge_temperature: float = 0.0
    judge_max_output_tokens: int = 512

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
