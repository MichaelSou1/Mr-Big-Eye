#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass


REQUIRED_MODULES = ("torch", "transformers", "peft")
OPTIONAL_MODULES = ("trl", "accelerate", "bitsandbytes")


@dataclass
class EnvAudit:
    name: str
    exists: bool
    ok: bool
    python: str = ""
    cuda_available: bool | None = None
    modules: dict[str, dict[str, str | bool]] | None = None
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "exists": self.exists,
            "ok": self.ok,
            "python": self.python,
            "cuda_available": self.cuda_available,
            "modules": self.modules or {},
            "error": self.error,
        }


def _conda_env_exists(name: str) -> bool:
    conda = shutil.which("conda")
    if not conda:
        return False
    result = subprocess.run(
        [conda, "env", "list", "--json"],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    suffix = f"/envs/{name}"
    return any(str(path).endswith(suffix) for path in payload.get("envs", []))


def _audit_env(name: str) -> EnvAudit:
    if not _conda_env_exists(name):
        return EnvAudit(name=name, exists=False, ok=False, error="conda env not found")

    code = r"""
import importlib
import json
import sys

modules = {}
for name in __REQUIRED__ + __OPTIONAL__:
    try:
        module = importlib.import_module(name)
        modules[name] = {
            "available": True,
            "version": str(getattr(module, "__version__", "")),
        }
    except Exception as exc:
        modules[name] = {"available": False, "error": str(exc)}

cuda_available = None
try:
    import torch
    cuda_available = bool(torch.cuda.is_available())
except Exception:
    pass

print(json.dumps({
    "python": sys.executable,
    "cuda_available": cuda_available,
    "modules": modules,
}, ensure_ascii=False))
""".replace("__REQUIRED__", repr(REQUIRED_MODULES)).replace("__OPTIONAL__", repr(OPTIONAL_MODULES))
    result = subprocess.run(
        ["conda", "run", "-n", name, "python", "-c", code],
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return EnvAudit(
            name=name,
            exists=True,
            ok=False,
            error=(result.stderr or result.stdout).strip(),
        )
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        return EnvAudit(name=name, exists=True, ok=False, error=f"invalid audit output: {exc}")

    modules = payload.get("modules") or {}
    ok = all(bool((modules.get(module) or {}).get("available")) for module in REQUIRED_MODULES)
    return EnvAudit(
        name=name,
        exists=True,
        ok=ok,
        python=str(payload.get("python") or ""),
        cuda_available=payload.get("cuda_available"),
        modules=modules,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit existing conda envs before creating a training env."
    )
    parser.add_argument("--candidates", nargs="+", default=["mbe-distill", "vlm_dapo"])
    parser.add_argument("--fallback-name", default="mbe-cot-train")
    args = parser.parse_args()

    audits = [_audit_env(name) for name in args.candidates]
    reusable = next((audit for audit in audits if audit.ok), None)
    decision = (
        {"action": "reuse_env", "env": reusable.name}
        if reusable is not None
        else {"action": "create_env", "env": args.fallback_name}
    )
    payload = {
        "decision": decision,
        "required_modules": list(REQUIRED_MODULES),
        "optional_modules": list(OPTIONAL_MODULES),
        "audits": [audit.as_dict() for audit in audits],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if reusable is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
