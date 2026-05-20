from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    sglang_endpoint: str = "http://127.0.0.1:30000"
    vlm_model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    vlm_model_local_dir: Path = Path("./models/Qwen3-VL-2B-Instruct")
    sglang_timeout: int = 120
    sglang_port: int = 30000
    sglang_tp_size: int = 1
    sglang_mem_fraction_static: float = 0.72
    sglang_served_model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    sglang_disable_cuda_graph: bool = False
    sglang_attention_backend: str = ""
    sglang_disable_overlap_schedule: bool = False

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    max_video_duration_sec: int = 600
    max_upload_size_mb: int = 2048
    load_models_on_startup: bool = True
    unload_models_after_use: bool = False

    bge_m3_model: str = "BAAI/bge-m3"
    bge_m3_local_dir: Path = Path("./models/bge-m3")
    siglip2_model: str = "google/siglip2-so400m-patch14-384"
    siglip2_modelscope_model: str = "google/siglip2-so400m-patch14-384"
    siglip2_local_dir: Path = Path("./models/siglip2-so400m-patch14-384")
    siglip2_dtype: str = "auto"
    models_device: str = "cuda:0"

    top_n_scenes: int = 5
    top_k_frames: int = 12
    vqa_max_frames: int = 6
    vqa_max_image_side: int = 448
    vqa_image_quality: int = 75
    scene_detect_threshold: float = 27.0
    dense_fps: float = 1.0

    data_dir: Path = Path("./data")
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
