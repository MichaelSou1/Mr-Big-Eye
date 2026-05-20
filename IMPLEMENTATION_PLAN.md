# Mr. Big-Eye

> *A long-video QA agent powered by Qwen3-VL, two-stage retrieval (caption + visual), and LangGraph memory*

---

## How to use this document

This is an implementation spec for a coding agent (Claude Code / Codex). Work top-down. Don't skip phases. Within a phase, complete tasks in order. Each task lists deliverables (files to create/modify) and acceptance criteria (how to verify it works before moving on).

**Conventions:**
- Python 3.10+, type hints on all public functions
- One concern per file, file length cap ~250 lines (split if exceeded)
- No global mutable state outside dedicated module(s)
- Use `pathlib.Path`, not `os.path`
- All paths configurable via `.env`, with sane defaults
- Logging via `logging` module, not `print` (except in scripts/)
- Error handling: raise specific exceptions, catch only what you can handle

**Out of scope (don't add unless explicitly listed):**
- Authentication beyond a name-tag user ID
- Multi-tenant deployment
- GPU memory optimization beyond what's listed
- Anything mentioned as "Phase 2" while implementing Phase 1

---

## Project structure (target)

```
mr-big-eye/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app + routes
│   ├── config.py               # Pydantic Settings, loads .env
│   ├── models.py               # bge-m3 / SigLIP2 loader (singletons)
│   ├── vqa.py                  # SGLang client wrapper
│   ├── preprocess.py           # 5-stage video preprocessing
│   ├── retrieval.py            # Two-stage retrieval
│   ├── cache.py                # File + status cache
│   ├── schemas.py              # Pydantic request/response models
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
│
├── scripts/
│   ├── launch_sglang.sh
│   ├── launch_app.sh
│   └── smoke_test.py           # End-to-end CLI test
│
├── data/                       # Runtime, gitignored
│   ├── uploads/
│   └── cache/
│       └── {video_id}/
│           ├── meta.json
│           ├── frames_scene/
│           ├── frames_dense/
│           ├── captions.jsonl
│           ├── caption_index/  # chromadb
│           └── frame_index/    # chromadb
│
└── tests/
    ├── test_cache.py
    ├── test_retrieval.py
    └── fixtures/
        └── short_clip.mp4      # 30-second test video
```

---

# PHASE 1 — MVP

**Goal:** Browser-based demo. User uploads a video (<=10 min), waits for preprocessing, then asks questions and gets answers with relevant keyframes shown below the answer.

**Definition of done:** Upload a 5-minute video, ask 3 questions, each answer arrives within 10 seconds and includes 8-12 relevant keyframes. README has a screenshot.

**Time budget:** 5-7 working days.

---

## Phase 1, Task 0 — Project bootstrap

**Goal:** Repository skeleton, dependencies, configuration.

### Deliverables

1. **`.gitignore`**

   Standard Python ignores + `data/`, `.env`, `*.pyc`, `__pycache__/`, `.pytest_cache/`, `*.egg-info/`.

2. **`requirements.txt`**

   ```
   # Web framework
   fastapi>=0.110.0
   uvicorn[standard]>=0.27.0
   python-multipart>=0.0.9
   aiofiles>=23.2.0

   # Config
   pydantic>=2.5.0
   pydantic-settings>=2.1.0
   python-dotenv>=1.0.0

   # Video & image
   decord>=0.6.0
   scenedetect[opencv]>=0.6.3
   Pillow>=10.0.0
   numpy>=1.24.0

   # Models & retrieval
   torch>=2.1.0
   transformers>=4.45.0
   FlagEmbedding>=1.2.10
   chromadb>=0.4.20
   openai>=1.40.0
   httpx>=0.27.0

   # Testing
   pytest>=8.0.0
   pytest-asyncio>=0.23.0
   ```

3. **`.env.example`**

   ```
   # SGLang inference endpoint
   SGLANG_ENDPOINT=http://127.0.0.1:30000
   VLM_MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
   SGLANG_TIMEOUT=120

   # App
   APP_HOST=0.0.0.0
   APP_PORT=8000
   MAX_VIDEO_DURATION_SEC=600
   MAX_UPLOAD_SIZE_MB=2048

   # Models
   BGE_M3_MODEL=BAAI/bge-m3
   SIGLIP2_MODEL=google/siglip2-so400m-patch14-384
   MODELS_DEVICE=cuda:0   # device for bge-m3 & SigLIP2

   # Retrieval
   TOP_N_SCENES=5
   TOP_K_FRAMES=12
   SCENE_DETECT_THRESHOLD=27.0
   DENSE_FPS=1.0          # 1 fps for SigLIP2 indexing

   # Storage
   DATA_DIR=./data

   # Logging
   LOG_LEVEL=INFO
   ```

4. **`app/config.py`**

   Use `pydantic_settings.BaseSettings` to load from `.env`. Single `Settings` class with all fields above, type-annotated. Expose `settings = Settings()` at module level. No other code in this file.

5. **`README.md`** (skeleton)

   ```markdown
   # Mr. Big-Eye

   *A long-video QA agent powered by Qwen3-VL, two-stage retrieval, and LangGraph memory.*

   ## Quick start

   1. Install: `pip install -r requirements.txt`
   2. Configure: `cp .env.example .env` and edit
   3. Launch SGLang: `bash scripts/launch_sglang.sh`
   4. Launch app: `bash scripts/launch_app.sh`
   5. Open `http://localhost:8000` in browser

   ## Architecture

   (placeholder — fill in after Phase 1 complete)
   ```

### Acceptance

- `pip install -r requirements.txt` succeeds in a fresh venv
- `python -c "from app.config import settings; print(settings.app_port)"` prints `8000`
- `.env.example` copied to `.env` works without errors

---

## Phase 1, Task 1 — SGLang client wrapper (`app/vqa.py`)

**Goal:** Reusable async wrapper around SGLang's OpenAI-compatible API for two call patterns: (a) caption a single image, (b) answer a question given multiple frames.

### Deliverables

1. **`app/vqa.py`** with two public async functions:

   ```python
   async def generate_caption(image: PIL.Image.Image) -> str:
       """Generate a 1-3 sentence English caption for a single frame.
       Used during preprocessing. Should be deterministic-ish
       (low temperature)."""

   async def answer_question(
       question: str,
       frames: list[PIL.Image.Image],
       timestamps: list[float],
       history: list[dict] | None = None,
   ) -> str:
       """Answer a question conditioned on K keyframes (already
       sorted by timestamp). `history` is OpenAI-format chat
       messages from prior turns in the same session.
       Returns plain text answer."""
   ```

2. Internal helpers:
   - `_pil_to_data_url(img, quality=85) -> str` — produces `data:image/jpeg;base64,...`
   - `_build_caption_messages(img) -> list[dict]` — short, fixed prompt
   - `_build_qa_messages(question, frames, timestamps, history) -> list[dict]` — system prompt instructs model to (a) cite which timestamp(s) the answer is based on, (b) answer concisely in user's language

3. Use `openai.AsyncOpenAI` client, base URL from `settings.sglang_endpoint + "/v1"`, api_key `"EMPTY"`.

4. **System prompt for QA** (put in a module-level constant, easy to tweak):

   ```
   You are Mr. Big-Eye, a careful video analyst. You will be shown
   K still frames sampled from a video, each labeled with its
   timestamp in seconds. Answer the user's question using only
   what is visible in these frames. When you reference a specific
   moment, cite the timestamp like [t=29.7s]. If the frames are
   insufficient, say so plainly. Match the user's language.
   ```

5. **System prompt for captioning:**

   ```
   Describe this video frame in one or two concise English
   sentences focused on actions, objects, and setting. No
   speculation. No "the image shows" preamble.
   ```

### Acceptance

- With SGLang running, manually run:
  ```python
  import asyncio
  from PIL import Image
  from app.vqa import generate_caption, answer_question
  img = Image.open("tests/fixtures/some_frame.jpg")
  print(asyncio.run(generate_caption(img)))
  print(asyncio.run(answer_question("What is happening?", [img], [0.0])))
  ```
  Both calls return non-empty strings without errors.

---

## Phase 1, Task 2 — Embedding models (`app/models.py`)

**Goal:** Single source of truth for loading bge-m3 (text) and SigLIP2 (image+text). Models loaded once at app startup, never reloaded.

### Deliverables

1. **`app/models.py`** with:

   ```python
   class BgeM3Wrapper:
       """Wraps FlagEmbedding.BGEM3FlagModel. Exposes encode_text(
       texts: list[str]) -> np.ndarray of shape (N, D)."""

   class Siglip2Wrapper:
       """Wraps transformers SigLIP2 model + processor. Exposes
       encode_image(images: list[PIL.Image]) -> np.ndarray
       and encode_text(texts: list[str]) -> np.ndarray.
       Both return L2-normalized vectors in the same joint space."""

   _bge: BgeM3Wrapper | None = None
   _siglip: Siglip2Wrapper | None = None

   def load_all_models() -> None:
       """Call once at app startup."""

   def get_bge() -> BgeM3Wrapper:
       """Returns loaded wrapper; raises if not loaded."""

   def get_siglip() -> Siglip2Wrapper:
       """Returns loaded wrapper; raises if not loaded."""
   ```

2. Both wrappers should:
   - Accept `device` from settings
   - Use `torch.no_grad()` + `eval()` mode
   - Batch internally if the input list is long (batch_size=32 for SigLIP image, 64 for text)

3. SigLIP2 specifics:
   - Use `AutoModel.from_pretrained(..., torch_dtype=torch.float16)` to halve VRAM
   - L2-normalize outputs (`F.normalize(emb, p=2, dim=-1)`)
   - For image encoding, accept `list[PIL.Image]`, internally call the processor

### Acceptance

- `python -c "from app.models import load_all_models, get_bge, get_siglip; load_all_models(); print(get_bge().encode_text(['hello']).shape); print(get_siglip().encode_text(['hello']).shape)"` runs and prints two shapes (probably `(1, 1024)` for bge-m3 and `(1, 1152)` or similar for SigLIP2)
- VRAM usage after load: bge-m3 ~2GB, SigLIP2-so400m ~2GB. Both fit on one 3080 alongside SGLang on other GPUs.

---

## Phase 1, Task 3 — Cache layer (`app/cache.py`)

**Goal:** Filesystem-based cache for video preprocessing results, plus an in-memory status table for "is this video done preprocessing?"

### Deliverables

1. **`app/cache.py`** with:

   ```python
   def video_id_from_file(path: Path) -> str:
       """SHA256 of file bytes, return first 16 hex chars."""

   def video_cache_dir(video_id: str) -> Path:
       """settings.data_dir / 'cache' / video_id. Does NOT create."""

   def ensure_cache_dirs(video_id: str) -> Path:
       """Create cache dir and subdirs (frames_scene, frames_dense,
       caption_index, frame_index). Return the root cache dir."""

   def get_video_status(video_id: str) -> str:
       """Returns one of: 'absent', 'running', 'done', 'failed:<msg>'.
       Checks both in-memory dict (for running) and on-disk marker file
       (for done)."""

   def set_video_status(video_id: str, status: str) -> None:
       """Sets in-memory; if status='done', also writes
       <cache_dir>/.done marker file."""

   def load_meta(video_id: str) -> dict:
       """Loads <cache_dir>/meta.json."""

   def save_meta(video_id: str, meta: dict) -> None:
       """Writes <cache_dir>/meta.json."""
   ```

2. Status table is a module-level `dict[str, str]` — fine for single-process MVP.

3. The `.done` marker is just an empty file; its presence means preprocessing succeeded. On `get_video_status`, if not in memory but `.done` exists → return `'done'`.

### Acceptance

- Unit test `tests/test_cache.py`:
  - `video_id_from_file` returns 16-hex string and is stable across calls
  - After `ensure_cache_dirs("abc")`, all subdirs exist
  - After `set_video_status("abc", "done")`, `get_video_status("abc")` returns `'done'` even in a fresh process (via marker file)

---

## Phase 1, Task 4 — Preprocessing pipeline (`app/preprocess.py`)

**Goal:** Given a video file, produce all artifacts needed for retrieval and serving.

### Pipeline stages

1. **Probe** — Read fps, duration, total frames via `decord.VideoReader`. Save to `meta.json`.

2. **Scene detection** — Use `scenedetect.detect()` with `ContentDetector(threshold=settings.scene_detect_threshold)`. Returns list of `(start_sec, end_sec)` tuples. If fewer than 3 scenes detected (likely a static video), fall back to uniform 10-second chunking.

3. **Scene middle frames + captions** — For each scene, extract the middle frame, save as `frames_scene/scene_{i:04d}.jpg` (quality=85), and call `vqa.generate_caption()`. Captions are awaited **concurrently** with `asyncio.gather` in batches of 8 (avoid overwhelming SGLang). Save to `captions.jsonl`, one JSON per line: `{"scene_id": i, "start": ..., "end": ..., "t_mid": ..., "caption": ...}`.

4. **Caption index** — Encode all captions with bge-m3, store in Chroma collection at `caption_index/`. Use the caption text as `documents`, scene metadata (start, end, t_mid) as `metadatas`, scene_id as `ids`.

5. **Dense frames + visual index** — Sample frames at `settings.dense_fps` (default 1 fps). For each, save as `frames_dense/t{t:06.1f}.jpg` and encode with SigLIP2. Store embeddings in Chroma collection at `frame_index/` with timestamp as metadata.

### Deliverables

1. **`app/preprocess.py`** with the single public async function:

   ```python
   async def preprocess_video(video_id: str, video_path: Path) -> dict:
       """Run all 5 stages. Returns meta dict. Raises on failure
       (caller is responsible for setting status='failed:<msg>')."""
   ```

2. Internal stage functions, each taking and returning explicit data (no hidden state):

   ```python
   def _probe(video_path: Path) -> dict
   def _detect_scenes(video_path: Path, fps: float, duration: float) -> list[tuple[float, float]]
   def _extract_scene_frame(vr: decord.VideoReader, fps: float, t_mid: float) -> PIL.Image
   async def _caption_scenes(scenes_with_frames: list[...]) -> list[dict]
   def _build_caption_index(cache_dir: Path, captions: list[dict]) -> None
   def _extract_and_index_dense_frames(vr, fps, duration, cache_dir) -> int
   ```

3. Logging: each stage logs start/end with elapsed time and counts.

### Acceptance

- Run `python -c "import asyncio; from pathlib import Path; from app.models import load_all_models; from app.preprocess import preprocess_video; load_all_models(); asyncio.run(preprocess_video('test001', Path('tests/fixtures/short_clip.mp4')))"`
- After completion, `data/cache/test001/` contains: `meta.json`, `frames_scene/*.jpg`, `frames_dense/*.jpg`, `captions.jsonl`, `caption_index/`, `frame_index/`
- `captions.jsonl` has one line per detected scene, each with a non-empty caption
- 5-minute 720p video preprocesses in <3 minutes on the target hardware

---

## Phase 1, Task 5 — Retrieval (`app/retrieval.py`)

**Goal:** Given a question, return K keyframes (PIL images + their timestamps), ordered by timestamp.

### Algorithm

1. Encode question with bge-m3 → query the caption index → get top-N scenes with their `(start, end)` time ranges
2. Encode question with SigLIP2 (text encoder) → query the frame index, filtered to only those time ranges → get top-K frame timestamps
3. Load the K frames from `frames_dense/`, sort by timestamp, return

### Deliverables

1. **`app/retrieval.py`** with:

   ```python
   @dataclass
   class RetrievalResult:
       frames: list[PIL.Image.Image]
       timestamps: list[float]
       scene_hits: list[dict]   # for debugging/UI: [{scene_id, start, end, caption, score}, ...]

   def two_stage_retrieve(
       video_id: str,
       question: str,
       top_n_scenes: int | None = None,
       top_k_frames: int | None = None,
   ) -> RetrievalResult:
       """Defaults read from settings."""
   ```

2. Chroma filter for stage 2:
   ```python
   where = {
       "$or": [
           {"$and": [{"timestamp": {"$gte": s}}, {"timestamp": {"$lte": e}}]}
           for s, e in time_ranges
       ]
   }
   ```
   Note: Chroma's filter syntax requires `$and` for compound conditions on the same field. Verify this works against the installed Chroma version; if not, do client-side filtering after a larger query.

3. Edge cases:
   - If stage 1 returns 0 scenes (empty video?), fall back to top-K frames over the whole video
   - If stage 2's filter returns fewer than K frames, do a second unfiltered query for the remainder

### Acceptance

- Unit test `tests/test_retrieval.py`: against a preprocessed `test001`, ask "what is happening" and verify `len(frames) == top_k_frames` and timestamps are sorted ascending.

---

## Phase 1, Task 6 — Schemas (`app/schemas.py`)

**Goal:** Single file for all Pydantic request/response models.

### Deliverables

```python
class UploadResponse(BaseModel):
    video_id: str
    status: str            # 'running' | 'done'
    cached: bool

class StatusResponse(BaseModel):
    video_id: str
    status: str            # 'absent' | 'running' | 'done' | 'failed:...'

class ChatMessage(BaseModel):
    role: str              # 'user' | 'assistant'
    content: str

class ChatRequest(BaseModel):
    video_id: str
    question: str
    history: list[ChatMessage] = []

class FramePayload(BaseModel):
    timestamp: float
    image_b64: str         # base64-encoded JPEG

class ChatResponse(BaseModel):
    answer: str
    frames: list[FramePayload]
    scene_hits: list[dict] = []
```

---

## Phase 1, Task 7 — FastAPI app (`app/main.py`)

**Goal:** Wire everything together. Single-process FastAPI app, four endpoints + static file serving.

### Deliverables

1. **`app/main.py`** with:

   ```python
   app = FastAPI(title="Mr. Big-Eye")

   @app.on_event("startup")
   async def _startup():
       load_all_models()

   app.mount("/static", StaticFiles(directory="app/static"), name="static")

   @app.get("/", include_in_schema=False)
   async def index():
       return FileResponse("app/static/index.html")

   @app.post("/upload", response_model=UploadResponse)
   async def upload_video(file: UploadFile = File(...)): ...

   @app.get("/status/{video_id}", response_model=StatusResponse)
   async def status(video_id: str): ...

   @app.post("/chat", response_model=ChatResponse)
   async def chat(req: ChatRequest): ...
   ```

2. **Upload handler:**
   - Stream file to a temp path under `data/uploads/`, computing SHA256 as it goes
   - Rename to `data/uploads/{video_id}.mp4` (or original extension)
   - Probe duration; if > `MAX_VIDEO_DURATION_SEC`, delete and return HTTP 400
   - If cache already has `.done` marker, return `status='done', cached=True`
   - Else: set status `'running'`, fire `asyncio.create_task(_run_preprocess(...))`, return `status='running', cached=False`
   - The background task wraps `preprocess_video()` in try/except, on success sets `'done'`, on failure sets `'failed:<msg>'`

3. **Chat handler:**
   - Validate status is `'done'`; else HTTP 409
   - `two_stage_retrieve(...)` → frames + timestamps
   - `answer_question(question, frames, timestamps, history)` → answer
   - Encode frames as base64 JPEG (quality=80 to keep payload small)
   - Return `ChatResponse`

4. **Error handling:** Use FastAPI's `HTTPException` with informative messages. Log full tracebacks server-side.

### Acceptance

- Start server: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
- `curl http://localhost:8000/` returns HTML
- `curl -X POST -F "file=@tests/fixtures/short_clip.mp4" http://localhost:8000/upload` returns JSON with `video_id` and `status=running`
- Poll `/status/{video_id}` until `'done'`
- `curl -X POST -H "Content-Type: application/json" -d '{"video_id":"...", "question":"what happens", "history":[]}' http://localhost:8000/chat` returns answer + frames

---

## Phase 1, Task 8 — Frontend (`app/static/`)

**Goal:** Single-page app, vanilla JS, no build step. Three sections: upload, chat input, message list with keyframe gallery.

### Design

- Header: "🦉 Mr. Big-Eye" + user ID badge (right-aligned)
- Upload section (visible when no video active): file picker, upload button, progress display
- Chat section (visible when video done): message list (above), input bar (below)
- Each assistant message: text on top, horizontal scrollable thumbnail row underneath
- Each thumbnail: click to enlarge in a simple modal (Phase 1 acceptable: just CSS hover zoom)

### Deliverables

1. **`app/static/index.html`** — Semantic HTML5, link to `style.css`, script to `app.js`. Three sections with IDs `upload-section`, `chat-section`, `messages`.

2. **`app/static/app.js`** — Vanilla JS, ~150-200 lines:

   ```javascript
   // User ID
   let userId = localStorage.getItem('mbe_userId');
   if (!userId) {
     userId = crypto.randomUUID().slice(0, 8);
     localStorage.setItem('mbe_userId', userId);
   }
   document.getElementById('user-tag').textContent = userId;

   // State
   let currentVideoId = null;
   const history = [];

   // Upload handler (using XMLHttpRequest for progress events)
   // Status polling (interval=2s, stops on 'done' or 'failed:...')
   // Chat handler (fetch POST, render message + gallery)
   // appendMessage(role, text, frames=[])
   ```

3. **`app/static/style.css`** — Modern minimal:
   - System font stack
   - Centered max-width 800px container
   - Message bubbles: user right-aligned blue-tint, assistant left-aligned gray-tint
   - Gallery: `display: flex; overflow-x: auto; gap: 8px;`
   - Each thumbnail: 120px height, auto width, `border-radius: 6px`
   - Subtle transitions, no animations beyond simple fades

### Acceptance

- Open `http://localhost:8000` in browser
- See logo + user ID (e.g., `a1b2c3d4`)
- Pick a small video, click upload, see progress bar then "video ready"
- Type a question, hit send, see answer text + thumbnails appear
- Refreshing the page resets state (this is fine for Phase 1)

---

## Phase 1, Task 9 — Scripts and smoke test

### Deliverables

1. **`scripts/launch_sglang.sh`**

   ```bash
   #!/usr/bin/env bash
   set -e
   python -m sglang.launch_server \
     --model-path Qwen/Qwen3-VL-8B-Instruct \
     --host 127.0.0.1 \
     --port 30000 \
     --tp-size 1 \
     --mem-fraction-static 0.85 \
     --chat-template qwen2-vl
   ```

2. **`scripts/launch_app.sh`**

   ```bash
   #!/usr/bin/env bash
   set -e
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
   ```

3. **`scripts/smoke_test.py`** — CLI script that:
   - Checks SGLang reachable
   - Uploads `tests/fixtures/short_clip.mp4`
   - Polls for done
   - Sends a hardcoded question
   - Prints answer + number of frames returned
   - Exits 0 on success, nonzero on any failure

### Acceptance

- `bash scripts/launch_sglang.sh` starts SGLang (manually verified)
- `bash scripts/launch_app.sh` starts the app on 8000
- `python scripts/smoke_test.py` exits 0

---

## Phase 1, Task 10 — README and demo capture

### Deliverables

1. **Update `README.md`** with:
   - One-paragraph project description
   - Architecture diagram (ASCII art is fine)
   - Hardware requirements (mention 4×RTX 3080 20GB tested config)
   - Setup steps, run steps
   - Screenshot of the UI showing a query result with thumbnails
   - "Known limitations" section listing what Phase 2 will add

2. **Record a 30-60 second screen capture** showing: upload → wait → ask question → see answer with keyframes. Save as `docs/demo.gif` (use `ffmpeg` to convert from screen recording, target <5MB).

### Acceptance

- A friend can clone the repo, follow the README, and reach a working demo
- The GIF plays in GitHub's README rendering

---

# PHASE 2 — Full version

**Goal:** Replace MVP shortcuts with the production-quality stack you actually want on your resume.

**Definition of done:** LangGraph drives conversation. LangMem persists per-user memory. SSE streams progress and answers. Witch/Queen-style user IDs. Keyframes inline in answers. Session list with restore.

**Time budget:** 7-10 working days after Phase 1.

---

## Phase 2, Task 11 — Add SQLite persistence (`app/db.py`)

**Goal:** Three tables: users, sessions, videos. SQLite, no ORM (just `sqlite3` stdlib) to keep it minimal.

### Deliverables

1. **`app/db.py`** with:

   ```python
   def init_db() -> None:
       """Create tables if not exist. Called at startup."""

   def create_or_get_user(username: str) -> dict:
       """Insert if absent, return {user_id, username, created_at}."""

   def get_user_by_id(user_id: str) -> dict | None: ...

   def create_session(user_id: str, video_id: str | None = None, title: str | None = None) -> str:
       """Returns session_id (uuid4)."""

   def list_sessions(user_id: str) -> list[dict]: ...

   def update_session(session_id: str, **fields) -> None: ...

   def register_video(video_id: str, user_id: str, filename: str, duration: float) -> None: ...

   def list_videos(user_id: str) -> list[dict]: ...
   ```

2. Schema (see earlier spec). All timestamps as `TIMESTAMP DEFAULT CURRENT_TIMESTAMP`.

3. Connection per-request (use a context manager). Don't hold a connection across `await` points.

### Acceptance

- `python -c "from app.db import init_db, create_or_get_user; init_db(); print(create_or_get_user('Witch'))"` prints user dict.

---

## Phase 2, Task 12 — Witch/Queen user ID namespace

**Goal:** Replace UUID-based localStorage IDs with memorable names from a curated pool.

### Deliverables

1. **`app/usernames.py`**

   ```python
   USERNAME_POOL: list[str] = [
       "Witch", "Wizard", "Oracle", "Phantom", "Specter",
       "Queen", "King", "Duke", "Knight", "Baron",
       "Phoenix", "Dragon", "Falcon", "Wolf", "Raven",
       "Ace", "Hero", "Sage", "Bard", "Scout",
       # Optional: add a Chinese-style alternative pool, toggle via .env
   ]

   def pick_username(taken: set[str]) -> str:
       """Pick a random name not in `taken`. If all 20 are taken,
       append _2, _3, etc. to a random one."""
   ```

2. **New endpoint:** `POST /api/login`
   - Body: `{username: str | null}`. If null, pick from pool. If provided, use as-is (let user override).
   - Calls `create_or_get_user`, returns user dict.

3. **Frontend update:** on first visit, call `/api/login` with `null`, store returned username in localStorage. Display in header. Add a "switch user" link that clears localStorage and reloads.

### Acceptance

- First visit to `/` displays a random name like "Witch" in the header
- Refresh keeps the same name
- Clear localStorage → new name on next visit

---

## Phase 2, Task 13 — LangGraph + LangMem integration

**Goal:** Replace the bare `answer_question` call with a LangGraph state machine that supports tool calls, memory, and checkpoint-based session restore.

### Architecture

```
LangGraph State:
  - messages: list[Message]
  - video_id: str | None
  - user_id: str
  - retrieved_frames: list[FramePayload] (set by video_qa tool)
  - retrieved_scene_hits: list[dict]

Nodes:
  - chat_node: LLM call (system prompt + history + memory context)
              decides whether to call video_qa or answer directly
  - tool_node: executes video_qa tool
  - memory_write_node: extracts memory from finished conversation

Edges:
  - START → chat_node
  - chat_node → tool_node (if tool call)
  - chat_node → memory_write_node → END (if direct answer)
  - tool_node → chat_node (loop back with tool result)

Checkpointer: SqliteSaver, thread_id = session_id
```

### Deliverables

1. **`app/graph.py`**

   ```python
   def build_graph(checkpointer: SqliteSaver) -> CompiledGraph: ...

   class GraphState(TypedDict):
       messages: Annotated[list[AnyMessage], add_messages]
       video_id: str | None
       user_id: str
       retrieved_frames: list[dict]
       retrieved_scene_hits: list[dict]
   ```

2. **Tool definition** — `video_qa(question: str)` reads `video_id` from state, runs retrieval + VLM, returns answer + frames. Tool result is added to messages; frames stored in state for the response builder.

3. **Memory** — Use `langmem.create_memory_manager` with namespace `("memories", user_id)`. Inject a few relevant memories into the system prompt at each chat_node call.

4. **`app/main.py` updates:**
   - On startup, build the graph once and store on `app.state.graph`
   - `/chat` endpoint accepts `session_id` (creates if absent), invokes graph with `config={"configurable": {"thread_id": session_id}}`
   - New endpoint `GET /api/sessions` lists user's sessions
   - New endpoint `POST /api/sessions` creates a new one (returns session_id)

### Acceptance

- Ask a question that doesn't require the video ("hi who are you"): graph answers directly, no retrieval triggered
- Ask a video question: graph calls `video_qa`, answer includes frame references
- Start a new session, ask question X, close browser, reopen with same session_id, prior context is restored
- After several sessions, ask "what kinds of videos have I analyzed before?" — memory provides the answer

---

## Phase 2, Task 14 — SSE streaming

**Goal:** Replace polling with Server-Sent Events for both preprocessing progress and chat answers.

### Deliverables

1. **`/upload` response changes:** returns `{video_id, status, cached, stream_url: "/api/preprocess_stream/{video_id}"}`

2. **`GET /api/preprocess_stream/{video_id}`** — SSE endpoint. Streams events:

   ```
   event: stage
   data: {"stage": "probe", "label": "正在打开视频卷轴 📜"}

   event: stage
   data: {"stage": "scenes", "label": "数一数有多少个场景 🎬"}

   event: stage
   data: {"stage": "captions", "label": "瞪大眼睛仔细看每个镜头 👀", "progress": 0.3}

   event: done
   data: {"video_id": "..."}
   ```

3. **Preprocessor refactor** — `preprocess_video()` accepts an optional `progress_callback: Callable[[str, str, float | None], None]`. Each stage calls it with `(stage, label, progress)`. The SSE endpoint subscribes to this via an `asyncio.Queue`.

4. **Trivia messages** — Module `app/progress.py` with:
   ```python
   STAGE_LABELS = {
       "probe":    ["正在打开视频卷轴 📜", "Unrolling the scroll..."],
       "scenes":   ["数一数有多少个场景 🎬", "Counting scenes..."],
       "captions": ["瞪大眼睛仔细看每个镜头 👀", "Studying each scene..."],
       "indexing": ["把看到的写进小本本 📝", "Taking notes..."],
       "embed":    ["给每一帧拍个写真 📸", "Photographing every frame..."],
   }
   ```
   Language toggle via setting `LANG=zh|en`, default zh.

5. **`/chat` streaming:** New endpoint `/api/chat_stream` returns SSE with:
   - `event: frames` first (so UI can render thumbnails immediately)
   - `event: token` for each LLM token
   - `event: done`

6. **Frontend updates:** Replace polling and `fetch` with `EventSource`. Render tokens incrementally.

### Acceptance

- Upload video, watch progress bar advance with trivia messages
- Send question, see thumbnails appear *first*, then text streams in token by token
- No more 2-second polling overhead

---

## Phase 2, Task 15 — Inline keyframe rendering

**Goal:** When the model writes `[FRAME:t=29.7]`, render the corresponding thumbnail inline at that position.

### Deliverables

1. **Update QA system prompt** in `app/vqa.py`:

   ```
   ...When you cite a moment, insert a marker like [FRAME:t=29.7]
   on its own. The renderer will replace it with the corresponding
   thumbnail. Use this sparingly (1-3 times per answer).
   ```

2. **Token stream handler in frontend** — buffer streamed text. When `[FRAME:t=NUMBER]` appears, find the closest timestamp in the already-received frames list, replace with `<img>` element.

3. **Fallback** — Frames not referenced inline still appear in the bottom gallery (Phase 1 behavior preserved).

### Acceptance

- Ask a localized question ("what happens at minute 2?") and see a thumbnail appear *within* the sentence, e.g., "At 2:03 we see [⛏ thumbnail of t=123.0s] the player jumping..."

---

## Phase 2, Task 16 — Session list UI

**Goal:** Sidebar showing user's past sessions. Click to restore.

### Deliverables

1. **Left sidebar** with sessions grouped by date. Each item: title (auto-generated from first user message) + video filename + last-active time.

2. **"New session" button** at top.

3. **On click:** loads session into main view by calling graph with that `thread_id` (which restores from checkpoint).

4. **Backend:** `GET /api/sessions` returns user's sessions. `GET /api/sessions/{id}/messages` returns the message history for restoration.

### Acceptance

- Run multiple sessions across different days
- Sidebar shows them grouped, click restores prior conversation visible in main panel

---

## Phase 2, Task 17 — Polish and ship

### Deliverables

1. **Error pages** — friendly UI for upload too large, video too long, preprocessing failed
2. **Loading skeletons** — instead of empty space
3. **Mobile responsive** — at least usable on tablet width
4. **Update README** — new screenshots, expanded architecture section, perf numbers
5. **New demo GIF** showing Phase 2 features (inline frames, session restore, trivia messages)
6. **Resume bullet points** — add to README under "Project Highlights":
   - "Two-stage retrieval (bge-m3 caption filter + SigLIP2 visual rerank) reduces VLM context from 600 frames to 12 while preserving 90%+ accuracy on LongVideoBench-subset"
   - "LangGraph + LangMem powers per-user persistent conversation memory with 20-name human-readable user IDs"
   - "SSE streams both preprocessing progress (with trivia messages) and token-by-token answers for sub-second perceived latency"
   - "FastAPI + SGLang + Chroma + SigLIP2 + bge-m3, all running on 4×RTX 3080 20GB"

### Acceptance

- README has live demo GIF
- Submit to a friend, get unsolicited "this looks cool" reaction
- Ready to link in resume

---

# Notes for the coding agent

- **Don't combine tasks.** Finish each task's acceptance criteria before starting the next. This catches integration issues early.
- **Test as you go.** Each task should produce at least one runnable thing you can demo to the human user.
- **When stuck, ask.** If a library API has changed, or a Chroma filter syntax doesn't work as written, or VRAM is exceeded, stop and ask the user before improvising.
- **Don't refactor without permission.** If you find an earlier file unclear, point it out instead of rewriting it.
- **Logging over print.** Every stage logs to a configurable logger. Easy to silence for demos, easy to verbose for debugging.
- **Phase boundary is sacred.** Don't sneak Phase 2 features into Phase 1. Phase 1 must be shippable on its own.

---

*End of plan. Start with Phase 1, Task 0.*
