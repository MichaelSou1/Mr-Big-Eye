# Audiovisual Eval Dataset

## 来源（4 数据集）

| 数据集 | 视频数 | 题数 | 时长 | 取得方式 |
|---|---|---|---|---|
| [Video-MME (medium)](https://huggingface.co/datasets/lmms-lab/Video-MME) | 50 | 150 | 4-15 min | hf-mirror，per-entry HTTP Range 抽取 |
| [Perception Test (sample)](https://github.com/google-deepmind/perception_test) | 4 | 21 | ~23 s | Google Storage（代理） |
| [CinePile](https://huggingface.co/datasets/tomg-group-umd/cinepile) | 4 | 24 | 1-3 min（YouTube 片段） | yt-dlp（代理 + HF_TOKEN） |
| [AVQA (Yang 2022)](https://github.com/AlyssaYoung/AVQA) | 2 | 8 | 10 s | yt-dlp + ffmpeg cut（代理） |
| **合计** | **60** | **203** | 混合 | — |

## License 备注

- Video-MME: CC-BY-NC-SA-4.0（research-only）
- Perception Test: CC-BY
- CinePile: gated dataset on HuggingFace，需要 accept terms（已通过 .env 里的 `HF_TOKEN` 走 huggingface.co 直连）
- AVQA: ACM MM 2022 paper 提供的 annotation；源视频来自 VGG-Sound（YouTube）

## modality_tag 分布

| modality | 题数 | 占比 |
|---|---|---|
| visual | 137 | 67% |
| joint | 47 | 23% |
| overview | 11 | 5% |
| audio | 8 | 4% |
| **合计** | **203** | 100% |

⚠️ **仍偏视觉**。原 spec 要求 audio 30% / visual 20% / joint 35% / overview 15%。
- Video-MME 无纯 audio 任务类型 → 0 audio
- CinePile 的 Character/Theme 类问题 → 部分 audio
- AVQA 的 `question_relation=Audio` → 全部 audio
- 想进一步拉高 audio%，需要再扩 AVQA 或加 MUSIC-AVQA / 自标对白 QA

## 域 / sub_category 分布（Video-MME 50 视频）

| Domain | Videos |
|---|---|
| Knowledge | 17 |
| Life Record | 11 |
| Sports Competition | 9 |
| Film & Television | 6 |
| Artistic Performance | 5 |
| Multilingual | 2 |

## 视频时长偏差

不同源差异很大：
- Video-MME medium: **4-15 min**（spec 目标段）
- CinePile: **1-3 min**（YouTube 电影场景片段）
- Perception Test: **~23 s**
- AVQA: **10 s**

这是 deliberate trade-off：spec 要"5-10 min 通用域"，但通用域里能凑出 audio/visual modality 标签的公开 benchmark 都是短片。后续做 latency / cost 评测时，建议按 `source` 字段切分对照，避免短片均摊把长视频的延迟掩盖。

## 评分协议

- 问题改造为开放式（剥离 MCQ 选项），`reference_answer` = 原数据集正确选项文本
- `expected_keywords` = 从正确选项提取的 1-3 个 discriminating 名词
- `expected_citation_kinds` 按 modality_tag 设定：
  - visual → `frame` 或 `slide`
  - audio → `transcript`
  - joint → `transcript` + `frame`
  - overview → `transcript`

## 关键产物

- [`questions.jsonl`](questions.jsonl) — EvalCase JSONL，203 行
- [`video_manifest.json`](video_manifest.json) — 各源 ID ↔ 我们 pipeline 的 video_id (sha256[:16] of mp4) 映射，60 entries
- [`../../data/uploads/{video_id}.mp4`](../../data/uploads/) — 60 个视频，已是 content-hash 命名（直接对应 pipeline 内部 video_id）
- 原始视频副本：
  - [`../../data/videomme_videos/`](../../data/videomme_videos/) — 50 个 Video-MME 原文件（YouTube ID 命名）
  - [`../../data/cinepile_videos/`](../../data/cinepile_videos/) — 4 个 CinePile yt-dlp 产物
  - [`../../data/avqa_videos/`](../../data/avqa_videos/) — 2 个 AVQA 10s 片段
  - [`../../data/perception_test/sample_videos/videos/`](../../data/perception_test/sample_videos/videos/) — 8 个 PT 样本（用了 4 个）

## 复现

```bash
# 1. Video-MME 主力 (~2.7 GB via hf-mirror)
HF_ENDPOINT=https://hf-mirror.com python scripts/download_videomme_subset.py
python scripts/build_videomme_eval.py

# 2. 三个补充数据集（PT 无需代理；CinePile/AVQA 需代理 + HF_TOKEN）
https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 \
  python scripts/build_supplementary_eval.py

# 3. Ingest 进 pipeline（GPU 重活，~4h 走完 60 视频）
python -m scripts.ingest_videomme   # 顺手也会处理新加的 supplementary 视频，因 uploads 已就位

# 4. 跑评测（建议先抽 N=20 验通）
python -m app.eval_harness --dataset audiovisual --n 20
```

## yt-dlp 提示

- 必须用 `--js-runtimes node:/usr/bin/node`（默认要 deno，本机无）
- 必须用 `--ffmpeg-location /home/user/miniconda3/envs/mbe-phase2/bin/ffmpeg`（PATH 上没有 ffmpeg）
- AVQA 10s 剪辑用 `--download-sections "*start-end"`
- 见 [scripts/build_supplementary_eval.py](../../scripts/build_supplementary_eval.py) `yt_dlp()` 函数
