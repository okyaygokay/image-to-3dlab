#!/usr/bin/env python3
"""Draw one picture with Qwen-Image on the VM, through the same sd-cli the local route runs.

    python scripts/qwen_image_colab.py request.json output.png

This is the remote wrapper `scripts/colab_generate.py` drives for the `qwen_image`
backend. The supervisor's protocol is positional (input, output), and the "image" slot
carries the request itself -- a small JSON of the prompt and the picture settings --
because a text-to-picture run has no input picture.

The sd-cli invocation is built by the *local* route's own `viewer/image_api.build_command`,
imported here rather than copied, so the two routes cannot drift: same flags, same weights
resolution, one definition. Its chatter goes to stdout untouched, which is exactly what the
supervisor's log -- and the studio's progress parser -- already read for a local run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "viewer"))

import image_api


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: qwen_image_colab.py request.json output.png", file=sys.stderr)
        return 2
    request_path, output = Path(argv[0]), Path(argv[1])
    try:
        payload = json.loads(request_path.read_text())
        prompt = str(payload["prompt"])
        settings = image_api.clean_settings(payload.get("settings") or {})
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"[qwen-colab] unreadable request: {exc}", file=sys.stderr)
        return 2

    try:
        weights = image_api.resolve_weights()
    except image_api.MissingWeights as exc:
        print(f"[qwen-colab] weights missing on the VM: {exc}", file=sys.stderr)
        return 3

    command = image_api.build_command(prompt, settings, output, weights)
    print(f"[qwen-colab] drawing {len(prompt)} chars at {settings.get('width')}x"
          f"{settings.get('height')}, seed {settings.get('seed')}", flush=True)
    done = subprocess.run(command, cwd=str(REPO), check=False)
    if done.returncode == 0 and not output.is_file():
        print(f"[qwen-colab] sd-cli exited 0 but wrote no {output}", file=sys.stderr)
        return 4
    return done.returncode


if __name__ == "__main__":
    raise SystemExit(main())
