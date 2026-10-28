#!/usr/bin/env python3
"""Install Qwen-Image on a rented Colab VM: a CUDA sd-cli build, then its weights.

    python scripts/bootstrap_qwen_image_colab.py --yes

Upstream publishes no Linux CUDA build of stable-diffusion.cpp (CUDA is Windows-only in
its releases; Linux gets Vulkan and ROCm), and a headless Colab VM has no Vulkan ICD to
run the Vulkan build against. So the binary is **compiled here**, from the same commit the
prebuilt routes pin (`master-975-95b3090`), with GGML's CUDA backend: about ten minutes
once per VM, and it lands at `vendor/sdcpp/sd-cli`, where `viewer/image_api` expects it.

The **weights** -- 13.4 GB: the Q8_0 image model, the Q4_K_M text encoder and the VAE --
are fetched by `bootstrap_qwen_image.py --weights-only` into the Hugging Face cache, from
the same pinned files the local route downloads.

The venv is the repo's whole lock (the same set the Pixal3D Colab venv uses): the weights
fetch needs `huggingface_hub`, and keeping one tested set of pins per VM is simpler than
curating a smaller one that can drift.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops, unless `--yes` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import shlex
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import host  # noqa: E402, RUF100

VENDOR = REPO / "vendor" / "sdcpp"
VENV = VENDOR / ".venv"
SRC = VENDOR / "src"
BUILD = VENDOR / "build"
BINARY = host.executable(VENDOR, "sd-cli")
BUILT_MARKER = VENDOR / ".i2l-build-complete"
BOOTSTRAP = REPO / "scripts" / "bootstrap_qwen_image.py"
LOCK = REPO / "requirements.lock"
PYTHON_VERSION = "3.11"
UPSTREAM = "https://github.com/leejet/stable-diffusion.cpp.git"
#: The commit behind the `master-975-95b3090` release every local route pins; building the
#: source at the release's own tag is what keeps the two routes the same program.
PINNED_REF = "master-975-95b3090"
WEIGHTS_GB = 13.4


def venv_python() -> Path:
    """The venv's interpreter. `host.venv_python` takes the *project* directory."""
    return host.venv_python(VENV.parent)


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


def install_python_packages() -> None:
    """The venv the weights fetch runs in: the repo's lock, whole."""
    ensure_uv()
    if not venv_python().is_file():
        must(["uv", "venv", "--python", PYTHON_VERSION, "--seed", "--clear", str(VENV)],
             what="creating the venv", timeout=1200)
    python = venv_python()
    if not python.is_file():
        raise SystemExit(f"the venv at {VENV} has no interpreter at {python}")
    must(["uv", "pip", "install", "--python", str(python), "-r", str(LOCK)],
         what="installing the Python packages", timeout=5400)


def build_commands(nvcc: str) -> list[list[str]]:
    """The clone and the two cmake steps, as data, so a test can read them.

    `--recursive`: ggml is a submodule, and a clone without it configures fine and then
    fails at generate time with a missing-target error that names nothing useful.
    """
    return [
        ["git", "clone", "--recursive", "--depth", "1", "--branch", PINNED_REF,
         UPSTREAM, str(SRC)],
        ["cmake", "-S", str(SRC), "-B", str(BUILD), "-DGGML_CUDA=ON",
         "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(BUILD), "--target", "sd-cli", "-j", "8"],
    ]


def install_binary() -> None:
    if BINARY.is_file():
        print("  sd-cli already built; leaving it alone", flush=True)
        return
    if subprocess.run(["sh", "-c", "command -v nvcc"], capture_output=True,
                      timeout=120, check=False).returncode != 0:
        raise SystemExit(
            "no nvcc on this VM: sd-cli's CUDA build needs the CUDA toolkit. Colab's GPU "
            "images carry it; if the image changed, this is the line to reconsider."
        )
    for command in build_commands("nvcc"):
        must(command, what=f"building sd-cli ({command[0]})", timeout=2400)
    built = BUILD / "bin" / "sd-cli"
    if not built.is_file():
        built = BUILD / "sd-cli"
    if not built.is_file():
        raise SystemExit(f"the build finished but no sd-cli appeared at {built}")
    BINARY.parent.mkdir(parents=True, exist_ok=True)
    built.replace(BINARY)
    subprocess.run(["chmod", "+x", str(BINARY)], check=False)
    print(f"  installed {BINARY}", flush=True)


def install_weights() -> None:
    """The three weight files, through the bootstrap that owns their pins."""
    must([str(venv_python()), str(BOOTSTRAP), "--weights-only", "--yes"],
         what="fetching the Qwen-Image weights", timeout=7200)


def install_edit_mode() -> None:
    """What a back view drawn on the VM needs: the vision projector and the matte.

    Edit mode reads the reference picture through the vision projector (0.75 GB, the
    local route's own pins via --vision), and the reference is the front's cut-out —
    BiRefNet-lite (224 MB), the matte the whole repo trusts with thin parts, from
    matte.py's pinned URL and md5. Without this step the draw falls back to u2net,
    which can eat thin parts out of the reference Qwen then faithfully redraws missing.
    """
    from image_to_3dlab import matte

    must([str(venv_python()), str(BOOTSTRAP), "--vision", "--yes"],
         what="fetching the Qwen vision projector (edit mode)", timeout=3600)
    target = matte.model_file(matte.LITE_MODEL)
    if target.is_file():
        print(f"  {matte.LITE_MODEL} already present; leaving it alone", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {matte.LITE_MODEL} ({matte.LITE_BYTES / 1e9:.2f} GB), md5-checked",
          flush=True)
    with urllib.request.urlopen(matte.LITE_URL, timeout=600) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)
    digest = hashlib.md5(target.read_bytes()).hexdigest()
    if digest != matte.LITE_MD5:
        target.unlink(missing_ok=True)
        raise SystemExit(f"{matte.LITE_MODEL} failed its md5 check ({digest}); retry the job")
    print(f"  installed {target}", flush=True)


def announcement() -> str:
    return "\n".join([
        "", "About to install:", "",
        "  backend: Qwen-Image (text to picture), stable-diffusion.cpp",
        "  route:   this rented Colab VM, into vendor/sdcpp/",
        f"  size:    about {WEIGHTS_GB:.1f} GB of weights, plus a CUDA compile (~10 min),",
        "           plus ~1.0 GB for edit mode (the vision projector and BiRefNet-lite,",
        "           which back views drawn on this VM need)",
        "  source:  leejet/stable-diffusion.cpp at " + PINNED_REF + " (compiled here:",
        "           upstream publishes no Linux CUDA build), and the pinned Qwen-Image",
        "           files from Hugging Face via bootstrap_qwen_image.py",
        "  licence: MIT (stable-diffusion.cpp); Qwen-Image's own licence for the weights",
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

    VENDOR.mkdir(parents=True, exist_ok=True)
    install_python_packages()
    install_binary()
    install_weights()
    install_edit_mode()
    BUILT_MARKER.write_text("ok")

    print("\nDone. Draw with:\n"
          f"    {venv_python()} scripts/qwen_image_colab.py request.json output.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
