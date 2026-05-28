#!/usr/bin/env python
"""Export the 14 TOOLS as OpenAI function schemas (spec §5.6).

All training samples share one schema file, versioned with AGENT_CODE_VERSION so
a tools.py change is detectable. Source is app.tools.TOOLS via bind_tools-equivalent
conversion — never hand-edit the schema.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.distill_format import export_tool_schemas
from app.eval_fingerprint import AGENT_CODE_VERSION


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/distillation/tool_schemas.json")
    args = parser.parse_args()

    schemas = export_tool_schemas()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent_code_version": AGENT_CODE_VERSION,
        "n_tools": len(schemas),
        "tools": schemas,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(schemas)} tools, version={AGENT_CODE_VERSION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
