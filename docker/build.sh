#!/usr/bin/env bash
# Build the Mr-Big-Eye ingest image.
# Run from the repo root:  bash docker/build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-mbe-ingest:latest}"

cd "${REPO_ROOT}"
echo "Building ${IMAGE} from ${REPO_ROOT}…"
docker build \
    -f docker/Dockerfile \
    -t "${IMAGE}" \
    .

echo
echo "Done. Inspect: docker image ls ${IMAGE}"
echo "Run:     bash docker/run-ingest.sh"
echo "Export:  docker save ${IMAGE} | gzip > mbe-ingest.tar.gz   # ~3-5GB, rsync to remote"
echo "Import:  gunzip -c mbe-ingest.tar.gz | docker load          # on remote"
