#!/usr/bin/env python3
"""Install Pixal3D on a rented Colab VM: the wrapper's venv, then trellis-cli and its weights.

Pixal3D is not a Python model. It is `trellis-cli` (C++/GGML) plus 8.4 GB of Q8_0 weights,
driven by `scripts/pixal3d_generate.py`. On a Mac the wrapper borrows the viewer's own
interpreter and the bootstrap compiles Metal kernels; on a rented Colab VM neither of those
is true, so this does the two halves that machine actually needs:

1. **A venv of its own**, because the wrapper imports Pillow, numpy, rembg, onnxruntime and
   trimesh, and the VM's system Python has none of them. The repo's lock is installed with
   PyTorch and the NVIDIA wheels stripped: Pixal3D never touches torch (its runtime is
   GGML), and pulling 6 GB of CUDA wheels for a backend that cannot use them would double
   the install for nothing.
2. **The prebuilt `trellis-cli` and the weights**, delegated to `bootstrap_pixal3d.py`.
   That script already knows the machine-dependent part -- our CUDA 12 prebuilt when the
   driver is new enough, a compile when it is not -- and keeping the choice in one place is
   what stops the two routes drifting.

The weights are fetched from their pinned Hugging Face revisions by `bootstrap_pixal3d.py`,
so a VM gets exactly the files that were tested rather than whatever was pushed since.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given
for non-interactive use. Defaulting to yes is not allowed, so it does not.

    python scripts/bootstrap_pixal3d_colab.py            # says what it wants, then asks
    python scripts/bootstrap_pixal3d_colab.py --yes      # for the studio and for agents
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

VENDOR = REPO / "vendor" / "pixal3d-cpp"
VENV = VENDOR / ".venv"
# Written last, so an install that died halfway never looks finished.
BUILT_MARKER = VENDOR / ".i2l-build-complete"
BOOTSTRAP = REPO / "scripts" / "bootstrap_pixal3d.py"
LOCK = REPO / "requirements.lock"
PYTHON_VERSION = "3.11"
WEIGHTS_GB = 8.4


def venv_python() -> Path:
    """The venv's interpreter. `host.venv_python` takes the *project* directory."""
    return host.venv_python(VENV.parent)


def ensure_uv() -> None:
    """Install uv if the VM has none, the same guard the Hi3D setup uses.

    A fresh Colab image does not always carry it, and `uv venv` is how the venv is made."""
    if subprocess.run(["sh", "-c", "uv --version"], capture_output=True,
                      timeout=120, check=False).returncode == 0:
        return
    must("curl -LsSf https://astral.sh/uv/install.sh | sh", what="uv install", timeout=600)


def announcement() -> str:
    return "\n".join([
        "", "About to install:", "",
        "  backend: Pixal3D (C++/GGML), single-view",
        "  route:   this rented Colab VM, into vendor/pixal3d-cpp/",
        f"  size:    about {WEIGHTS_GB:.1f} GB of weights, plus the Python packages",
        "  source:  raven38/pixal3d-sv-q8_0-v1 on Hugging Face, plus the pinned",
        "           trellis-cli CUDA 12 prebuilt from this repo's own release",
        "  licence: MIT for the code and flow weights; the bundled image encoder is",
        "           under the DINOv3 License",
        "",
    ])


def must(command: list[str] | str, *, what: str, timeout: float) -> None:
    """Run one install step, or say which one failed. A string runs through the shell."""
    printable = command if isinstance(command, str) else " ".join(shlex.quote(str(p)) for p in command)
    print(f"  $ {printable}", flush=True)
    done = subprocess.run(command, shell=isinstance(command, str),
                          timeout=timeout, check=False)
    if done.returncode != 0:
        raise SystemExit(f"{what} failed (exit {done.returncode})")


def install_python_packages() -> None:
    """The venv the wrapper runs in: the lock, minus PyTorch and the CUDA wheels."""
    ensure_uv()
    if not venv_python().is_file():
        # --clear: a previous attempt can leave the directory without a usable interpreter
        # (uv creates the directory first, then populates it). Without it, the retry dies
        # with "a virtual environment already exists" and the reason is invisible.
        must(["uv", "venv", "--python", PYTHON_VERSION, "--seed", "--clear", str(VENV)],
             what="creating the venv", timeout=1200)
    python = venv_python()
    if not python.is_file():
        raise SystemExit(f"the venv at {VENV} has no interpreter at {python}")

    # `without-cuda` is the same transform the AMD install uses: keep every pin, drop the
    # ones that only exist to carry PyTorch's CUDA runtime.
    lock = LOCK.read_text() if LOCK.is_file() else (REPO / "requirements.txt").read_text()
    stripped = VENV / "requirements.colab.lock"
    stripped.write_text(host.without_cuda(lock))
    must(["uv", "pip", "install", "--python", str(python), "-r", str(stripped)],
         what="installing the wrapper's Python packages", timeout=3600)


def install_trellis_and_weights() -> None:
    """The binary and the weights, through the bootstrap that owns those decisions."""
    must([str(venv_python()), str(BOOTSTRAP), "--yes"],
         what="installing trellis-cli and the Pixal3D weights", timeout=7200)


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

    VENDOR.mkdir(parents=True, exist_ok=True)
    install_python_packages()
    install_trellis_and_weights()
    BUILT_MARKER.write_text("ok")

    print("\nDone. Generate with:\n"
          f"    {venv_python()} scripts/pixal3d_generate.py input.png output.glb --res 1024")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
