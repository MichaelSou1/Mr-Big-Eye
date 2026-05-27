# Docker ingest 远端跑法

WSL 本地 GPU 不稳，把 ingest 迁到 4×RTX3080 服务器。整个流程不在服务器装 conda/Python，全靠这个镜像。

## 0. 服务器前置条件

- NVIDIA 驱动 ≥525（CUDA 12.1 base 兼容）。检查：`nvidia-smi`
- Docker ≥20.10
- NVIDIA Container Toolkit（即 `nvidia-docker2`）。检查：
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
  ```
  能列出 4 张 3080 = 通过。
- 磁盘 ≥30GB 空闲（镜像 5GB + 数据 10GB + cache 输出 ~10GB）

如果服务器没装 nvidia-container-toolkit，一次性安装：
```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 1. 本地：pre-stage uploads + 构建镜像（可选）

```bash
# 在本机 mbe-phase2 env
python -m scripts.stage_uploads      # 把 60 个原视频 copy 进 data/uploads/
# 预期：staged=28 already_present=32 missing=0 total_uploads=60
```

**镜像构建有两种走法**——按你的网络条件选：

### 1a. 直接在远端 build（推荐：远端网络快）
跳过本地构建，看 §2。

### 1b. 本地 build + scp（本地有 docker desktop 且 WSL 集成开了的话）
```bash
bash docker/build.sh                                  # → mbe-ingest:latest
docker save mbe-ingest:latest | gzip > mbe-ingest.tar.gz   # ~3-5GB
```

## 2. 同步代码 + 数据 + 模型到远端

```bash
REMOTE=user@server                  # 改成你的 ssh 别名
RPATH=/home/user/Mr-Big-Eye         # 远端目标根目录

# 2.1 同步代码（很小，~10MB）
rsync -avz --exclude data/ --exclude models/ --exclude .git/ \
      --exclude '__pycache__/' --exclude '*.pyc' \
      ~/Mr-Big-Eye/ $REMOTE:$RPATH/

# 2.2 同步 .env（含 API keys，单独传）
scp ~/Mr-Big-Eye/.env $REMOTE:$RPATH/.env

# 2.3 同步模型（~6.5GB，最慢的一步）
rsync -avz --progress ~/Mr-Big-Eye/models/ $REMOTE:$RPATH/models/

# 2.4 同步预 stage 好的 uploads（~3GB）
rsync -avz --progress ~/Mr-Big-Eye/data/uploads/ $REMOTE:$RPATH/data/uploads/

# 2.5（可选）如果走 1b 路线，传镜像
scp mbe-ingest.tar.gz $REMOTE:$RPATH/
```

## 3. 远端：build（或 load）+ run

ssh 进服务器后：

```bash
cd /home/user/Mr-Big-Eye

# 3.1 build 或 load 镜像
bash docker/build.sh                                       # 在线 build，~10min
# 或者：gunzip -c mbe-ingest.tar.gz | docker load           # 离线 load

# 3.2 单卡顺序跑（约 5h）
bash docker/run-ingest.sh

# 3.3 四卡并行（约 1.5h）
for g in 0 1 2 3; do
    MBE_INGEST_SHARD="${g}/4" GPU=${g} \
        CONTAINER_NAME=mbe-ingest-gpu${g} \
        bash docker/run-ingest.sh > ingest_gpu${g}.log 2>&1 &
done
wait
echo "all 4 shards done"
```

并行模式下每个 container 占 1 张卡，按 video_id 字典序 mod 4 分片。每张卡处理 15 个视频。

## 4. 跑完拉回 cache

```bash
# 远端 → 本地
rsync -avz --progress $REMOTE:$RPATH/data/cache/ ~/Mr-Big-Eye/data/cache/
rsync -avz $REMOTE:$RPATH/data/transcripts.sqlite3 ~/Mr-Big-Eye/data/transcripts.sqlite3
rsync -avz $REMOTE:$RPATH/data/mr_big_eye.sqlite3 ~/Mr-Big-Eye/data/mr_big_eye.sqlite3
```

回本地后直接 `python -m app.eval_harness --dataset audiovisual --n 20` 验通。

## 排障

- **`nvidia-smi: command not found in container`**：`--gpus` 参数没生效，检查 nvidia-container-toolkit
- **`CUDA error: no kernel image available`**：torch wheel 跟 GPU 架构不匹配。3080/3090/4060/4090 都是 Ampere/Ada，CUDA 12.1 wheel 必然兼容；如果失败说明驱动太老（<525），升驱动
- **`OSError: ./models/bge-m3 not found`**：models/ 没挂进去，检查 `-v` 路径
- **`FAILED: source missing`**：uploads 没全部 stage，本地重跑 `python -m scripts.stage_uploads`
- **第一次跑慢**：ModelScope 要下 SenseVoice + FSMN-VAD（~900MB），存在命名 volume `mbe-modelscope-cache` 里，第二个 container 起来就走缓存了。如果远端不能上 modelscope.cn，提前在本地起个 throwaway container 把这俩下下来然后 `docker volume cp`，或者直接 rsync `~/.cache/modelscope/` 到远端 `/var/lib/docker/volumes/mbe-modelscope-cache/_data/`
