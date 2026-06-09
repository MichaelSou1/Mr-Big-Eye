from __future__ import annotations

import argparse
import json
import subprocess
from typing import Literal


def fetch(artifact_id: str, kind: Literal["model", "dataset"], *, output_dir: str | None = None) -> dict:
    """Resolve an artifact with ModelScope-first policy.

    This function intentionally does not hardcode HuggingFace URLs. If
    ModelScope cannot serve the artifact, it returns a fallback decision for the
    caller/user to handle with their configured mirrors.
    """
    cmd = ["modelscope", "download"]
    if kind == "dataset":
        cmd.append("--dataset")
    cmd.append(artifact_id)
    if output_dir:
        cmd.extend(["--local_dir", output_dir])
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode == 0:
        payload = {
            "artifact_id": artifact_id,
            "kind": kind,
            "source": "modelscope",
            "ok": True,
            "stdout": result.stdout[-1000:],
        }
        if "next-gqa" in artifact_id.lower() or "nextgqa" in artifact_id.lower():
            payload["requires_grounding_gt_verification"] = True
        return payload
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "source": "fallback_required",
        "ok": False,
        "stderr": result.stderr[-1000:],
        "message": (
            "ModelScope did not provide this artifact. Use the original release or "
            "HF mirror via shell-level configuration; do not hardcode fallback URLs in pipeline code."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ModelScope-first artifact resolver.")
    parser.add_argument("artifact_id")
    parser.add_argument("--kind", choices=["model", "dataset"], required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    payload = fetch(args.artifact_id, args.kind, output_dir=args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
