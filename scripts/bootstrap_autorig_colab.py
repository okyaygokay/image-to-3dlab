#!/usr/bin/env python3
"""Install the auto-rigger on a rented Colab VM: SkinTokens, its venv and its checkpoints.

    python scripts/bootstrap_autorig_colab.py --yes

`bootstrap_autorig.py` already knows this machine shape -- it installs the CUDA torch the
driver can run, applies the portable-attention patch, and fetches the pinned checkpoints.
What it assumes is a `uv` on PATH and a `huggingface_hub` in the interpreter running it;
a fresh Colab VM has neither. So this makes a small driver venv with exactly that one
package (SkinTokens' own venv, with its 2 GB of torch, is built by the bootstrap itself),
then hands the whole install over.

Nothing is duplicated: the rigger's venv, weights and code live at the paths
`viewer/autorig_api` already computes, so the wrapper `scripts/autorig_colab.py` drives the
very commands the local route would.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import host  # noqa: E402, RUF100

VENDOR = REPO / "vendor" / "SkinTokens"
DRIVER = REPO / "vendor" / "autorig-colab"
DRIVER_VENV = DRIVER / ".venv"
BUILT_MARKER = DRIVER / ".i2l-build-complete"
BOOTSTRAP = REPO / "scripts" / "bootstrap_autorig.py"
PYTHON_VERSION = "3.11"
PACKAGE_GB = 2.0
WEIGHTS_GB = 1.6


def driver_python() -> Path:
    return host.venv_python(DRIVER)


def must(command: list[str] | str, *, what: str, timeout: float) -> None:
    """Run one install step, or say which one failed. A string runs through the shell."""
    printable = command if isinstance(command, str) else " ".join(shlex.quote(str(p)) for p in command)
    print(f"  $ {printable}", flush=True)
    done = subprocess.run(command, shell=isinstance(command, str), timeout=timeout, check=False)
    if done.returncode != 0:
        raise SystemExit(f"{what} failed (exit {done.returncode})")


def ensure_uv() -> None:
    if subprocess.run(["sh", "-c", "uv --version"], capture_output=True,
                      timeout=120, check=False).returncode == 0:
        return
    must("curl -LsSf https://astral.sh/uv/install.sh | sh", what="uv install", timeout=600)


def announcement() -> str:
    return "\n".join([
        "", "About to install:", "",
        "  backend: Auto-rig (SkinTokens: skeleton + skin weights)",
        "  route:   this rented Colab VM, into vendor/SkinTokens/",
        f"  size:    about {PACKAGE_GB:.0f} GB of packages (torch, CUDA) and {WEIGHTS_GB:.1f} GB of checkpoints",
        "  source:  VAST-AI-Research/SkinTokens at a pinned commit, patched and built by",
        "           scripts/bootstrap_autorig.py; checkpoints from the pinned Hugging",
        "           Face revision the local route also uses",
        "  licence: MIT (SkinTokens); the checkpoints are pickles pinned to the tested",
        "           revision, which is why the revision is named everywhere",
        "",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the studio and for agents.")
    args = parser.parse_args(argv)

    print(announcement(), flush=True)
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1

    DRIVER.mkdir(parents=True, exist_ok=True)
    ensure_uv()
    if not driver_python().is_file():
        must(["uv", "venv", "--python", PYTHON_VERSION, "--seed", "--clear", str(DRIVER_VENV)],
             what="creating the driver venv", timeout=1200)
    must(["uv", "pip", "install", "--python", str(driver_python()), "huggingface-hub"],
         what="installing the driver's one package", timeout=1200)

    must([str(driver_python()), str(BOOTSTRAP), "--yes"],
         what="installing SkinTokens and its checkpoints", timeout=7200)
    BUILT_MARKER.write_text("ok")
    print("\nDone. Rig with:\n"
          "    python scripts/colab_generate.py model.glb rigged.glb --backend autorig --yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
