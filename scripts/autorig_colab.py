#!/usr/bin/env python3
"""Rig a humanoid mesh, or play a preset on one, with SkinTokens on the VM.

    python scripts/autorig_colab.py model.glb rigged.glb
    python scripts/autorig_colab.py rigged.glb animated.glb --preset dance --arm-spread 12

This is the remote wrapper `scripts/colab_generate.py` drives for the `autorig` backend.
The commands it runs are the local route's own -- `viewer/autorig_api.rig_commands` and
`animate_command` -- imported rather than copied, so the two routes cannot drift. A preset
name switches it to animate mode; the motion clip itself is `image_to_3dlab/motions/`
on the VM (the supervisor's sync carries it).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "viewer"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import autorig_api


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    forwarded = [value for value in argv if value != "--"]

    def flag(name: str, default: str | None = None) -> str | None:
        try:
            return forwarded[forwarded.index(name) + 1]
        except (ValueError, IndexError):
            return default

    if len(forwarded) < 2:
        print("usage: autorig_colab.py source.glb result.glb [--preset id "
              "--arm-spread 12 --speed 1]", file=sys.stderr)
        return 2
    source, result = Path(forwarded[0]), Path(forwarded[1])
    preset = flag("--preset")
    if preset is None:
        commands = autorig_api.rig_commands(source, result)
        print(f"[autorig-colab] rigging {source.name} -> {result.name}", flush=True)
    else:
        motions = REPO / "image_to_3dlab" / "motions" / f"{preset}.npz"
        if not motions.is_file():
            print(f"[autorig-colab] no such preset on this VM: {preset}", file=sys.stderr)
            return 2
        spread = flag("--arm-spread", "0")
        pace = flag("--speed", "1.0")
        commands = [(autorig_api.animate_command(source, motions, result,
                                                 float(spread), float(pace)), REPO)]
        print(f"[autorig-colab] playing {preset} on {source.name} -> {result.name}",
              flush=True)

    for command, cwd in commands:
        print(f"[autorig-colab] $ {' '.join(str(part) for part in command)}", flush=True)
        done = subprocess.run(command, cwd=str(cwd), check=False)
        if done.returncode != 0:
            return done.returncode
    if not result.is_file():
        print(f"[autorig-colab] finished but wrote no {result}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
