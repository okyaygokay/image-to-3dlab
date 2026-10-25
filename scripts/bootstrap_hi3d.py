#!/usr/bin/env python3
"""Install Hi3D v1.0 for Linux with an NVIDIA card: high-resolution mesh refinement.

Hi3D is not an image-to-mesh model. It refines a *coarse mesh* against a reference picture
and samples a much finer one, up to an effective 1536^3 sparse grid. The coarse mesh comes
from another route here -- Hunyuan3D-2 mini's turbo variant -- which this installs too, so
one command leaves you able to run the whole pair.

    python scripts/bootstrap_hi3d.py            # says what it will fetch, then asks
    python scripts/bootstrap_hi3d.py --yes      # for the viewer and for agents
    python scripts/bootstrap_hi3d.py --check    # exit 0 if installed, 1 if not

**The stack is pinned, and it is not Hi3D's published one.** Hi3D documents Python 3.10 /
torch 2.5.0+cu124 / spconv-cu120. That combination has no kernels for compute capability
12.0 (RTX 50-series and RTX PRO 6000), so on a Blackwell card it cannot run at all. This
installer pins torch 2.9.1+cu128 and spconv-cu126 2.3.8, which were verified on a live
RTX PRO 6000; every other version is pinned to the exact number Hi3D was measured with.
See `docs/hi3d.md` for the full table and for what "verified" covered.

`flash-attn` is installed from a prebuilt wheel built for that exact torch/cp310/sm_120
triple. Building it from source takes about an hour and needs the CUDA toolkit; the wheel
is addressed by a pinned dataset commit and checked against its sha256, not by ``main``.

**Licence:** Hi3D is under the Hi3D Open License Agreement 1.0. Commercial use is
permitted, but a commercial product must display "Powered by Hi3D 1.0", and passing
200,000 monthly users or USD 1,000,000 monthly revenue requires a separate written licence
within 60 days. The coarse stage's weights (Tencent Hunyuan 3D 2.0) are **not** licensed
for use in the EU, the UK or South Korea -- so this whole pairing inherits that
restriction, which is why the announcement names it.

`AGENTS.md`: a download path must name the backend, name the route, state the size, and
require an affirmative answer. This prints all of that and stops without `--yes`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import host, matte  # noqa: E402, RUF100

VENDOR = REPO / "vendor" / "hi3d"
CHECKOUT = VENDOR / "Hi3Dv1.0"
VENV = VENDOR / ".venv"
BUILT_MARKER = VENDOR / ".i2l-build-complete"

#: Pinned source commits. Never a branch: a refinement run is only reproducible if the
#: code that produced it is identified exactly.
HI3D_COMMIT = "13657ad09afaa92e350b461a1cd66399e44a59c4"
HI3D_REPO = "https://github.com/hi3d-official/Hi3Dv1.0"
HUNYUAN_COMMIT = "f8db63096c8282cb27354314d896feba5ba6ff8a"
HUNYUAN_REPO = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2"
COARSE_CHECKOUT = VENDOR / "Hunyuan3D-2"

PYTHON_VERSION = "3.10.22"
UV_VERSION = "0.13.4"

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
PYPI_INDEX = "https://pypi.org/simple"

#: flash-attn for exactly torch 2.9.1+cu128 / cp310 / sm_120. Addressed by dataset commit
#: so it cannot be swapped under us, and verified by sha256 below.
FLASH_ATTN_URL = (
    "https://huggingface.co/datasets/multimodalart/zerogpu-blackwell-wheels/resolve/"
    "2c8c2580a83f59b2332c2ed6668f5f0b2f30cdf5/wheels/pt291-cu128-cp310/"
    "flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl"
)
FLASH_ATTN_SHA256 = "19e19483b38310cf458c05bd64c6602a47ca4f2e97ca5208ec883d410618ee42"
SPCONV_VERSION = "spconv-cu126==2.3.8"

#: Every other Python dependency, at the exact version Hi3D was measured against.
PINNED = [
    "transformers==4.46.3",
    "huggingface-hub==0.28.1",
    "safetensors==0.4.5",
    "numpy==1.26.4",          # Hi3D requires numpy < 2
    "pillow==11.0.0",
    "trimesh==4.4.9",
    "pyfqmr==0.4.0",
    "accelerate==1.1.1",
    "einops==0.8.0",
    "diffusers==0.31.0",
    "omegaconf==2.3.0",
    "pymeshlab==2023.12.post1",
    "pygltflib==1.16.3",
    "xatlas==0.0.9",
    "opencv-python-headless==4.10.0.84",
    "scikit-image==0.24.0",
    "tqdm==4.66.5",
    "pyyaml==6.0.2",
    "scipy==1.13.1",
]

#: The coarse stage. Tencent's own code, pinned, installed with --no-deps because the
#: versions above already decide every package it needs.
COARSE_PINNED = ["rembg==2.0.61", "onnxruntime==1.20.1", "pymatting==1.1.16",
                 "numba==0.60.0", "llvmlite==0.43.0", "pooch==1.8.2"]

#: Weights, as (label, source, approximate bytes, what it is for). Sizes are measured
#: from ModelScope and Hugging Face, and exist to set expectations, not to verify.
WEIGHTS = (
    ("Hi3D v1.0 refinement weights", "hi3d-official/Hi3Dv1.0", 7.73 * 1024 ** 3,
     "two 1.8 B-parameter transformers, the sparse VAE, and the scheduler config"),
    ("DINOv2-large image encoder", "facebook/dinov2-large", 1.22 * 1024 ** 3,
     "Hi3D reads the reference picture with it"),
    ("CLIP-large image encoder", "openai/clip-vit-large-patch14", 1.71 * 1024 ** 3,
     "the second half of Hi3D's image conditioning"),
    ("Hunyuan3D-2 mini turbo (coarse stage)", "tencent/Hunyuan3D-2mini", 3.82 * 1024 ** 3,
     "makes the coarse mesh Hi3D refines"),
)

LICENCE = (
    "Hi3D Open License 1.0: commercial use allowed, but a commercial product must display\n"
    '          "Powered by Hi3D 1.0", and 200k monthly users or USD 1M monthly revenue needs\n'
    "          a separate licence. The coarse stage's Hunyuan weights are NOT licensed in\n"
    "          the EU, the UK or South Korea."
)


def venv_python() -> Path:
    """The venv's interpreter.

    `host.venv_python` takes the *project* directory and appends `.venv/bin/python`; this
    module's VENV already ends in `.venv`, so it must be given the parent or the path
    doubles to `.venv/.venv/bin/python` (which is how the remote install failed).
    """
    return host.venv_python(VENV.parent)


def total_gb() -> float:
    return sum(size for _, _, size, _ in WEIGHTS) / 1024 ** 3


def announcement(code: bool = True, weights: bool = True) -> str:
    lines = ["", "About to install:", "", "  backend: Hi3D v1.0 (refinement)",
             "  route:   NVIDIA / CUDA on Linux, its own venv"]
    if code:
        lines.append(f"  code:    {HI3D_REPO.split('/')[-2]}/{HI3D_REPO.split('/')[-1]} "
                     f"@ {HI3D_COMMIT[:7]} -> vendor/hi3d/Hi3Dv1.0/")
        lines.append(f"           plus the coarse stage, Tencent-Hunyuan/Hunyuan3D-2 "
                     f"@ {HUNYUAN_COMMIT[:7]}")
        lines.append(f"  stack:   Python {PYTHON_VERSION}, torch 2.9.1+cu128, "
                     f"{SPCONV_VERSION}, flash-attn 2.7.4.post1 (prebuilt)")
        lines.append("           pinned because Hi3D's published cu124 stack has no "
                     "Blackwell kernels")
    if weights:
        lines.append(f"  weights: {total_gb():.1f} GB total")
        for label, source, size, _ in WEIGHTS:
            lines.append(f"             {size / 1024 ** 3:>6.2f} GB  {label} ({source})")
        lines.append(f"  and:     BiRefNet-lite background remover, "
                     f"{matte.LITE_BYTES / 1e6:.0f} MB -> {matte.model_file(matte.LITE_MODEL)}")
    lines += ["", "  licence: " + LICENCE, ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# Pure helpers, tested without a GPU
# ----------------------------------------------------------------------------------------


def parse_compute_capability(value: str | tuple[int, int] | None) -> tuple[int, int] | None:
    """`host.compute_capability()` spells it CMake-style ("120"); accept either form.

    Written as a parser rather than a cast because the two spellings are easy to confuse:
    "120" is major 12 minor 0, not major 1 minor 20.
    """
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    digits = str(value).strip()
    if not digits.isdigit() or len(digits) < 2:
        return None
    return int(digits[:-1]), int(digits[-1])


def choose_route(compute_capability: str | tuple[int, int] | None,
                 driver: tuple[int, ...] | None) -> tuple[str | None, str]:
    """Which torch build this card needs, and why.

    Same shape as the Hunyuan installer's: a card too old is refused with a reason rather
    than left to fail inside a kernel, and a card newer than Hi3D's documented stack gets
    the cu128 build that has kernels for it.
    """
    from image_to_3dlab.hi3d_backend import MIN_COMPUTE_CAPABILITY

    capability_tuple = parse_compute_capability(compute_capability)
    if capability_tuple is None:
        return None, ("could not read this card's compute capability; install the NVIDIA "
                      "driver so nvidia-smi reports it")
    major, minor = capability_tuple
    capability = major + minor / 10
    if capability < MIN_COMPUTE_CAPABILITY:
        return None, (f"compute capability {capability:.1f} is older than "
                      f"{MIN_COMPUTE_CAPABILITY:.1f}; Hi3D has no path for it")
    if driver is not None and tuple(driver)[:2] < (12, 4):
        return None, (f"driver supports CUDA {'.'.join(map(str, driver))}, and the cu128 "
                      "build needs 12.4 or newer")
    # Every card from Ampere onwards uses the cu128 build: it is the one that has
    # Blackwell kernels and it still runs Ampere and Hopper.
    return "cu128", (f"compute capability {capability:.1f}, driver CUDA "
                     f"{'.'.join(map(str, driver)) if driver else 'unknown'}")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_installed() -> bool:
    """Whether the checkout, the venv, the build marker and every weight are present."""
    if not (BUILT_MARKER.is_file() and venv_python().is_file()):
        return False
    return all(checkout_installed() and weights_installed())


def checkout_installed() -> bool:
    from image_to_3dlab.hi3d_backend import BUILD_PROBES

    return all((CHECKOUT / name).is_file() for name in BUILD_PROBES)


def weights_installed() -> bool:
    """True when each repo is in the Hugging Face cache *and* holds real weights.

    A cache with only symlinks and config files is a broken download, not an install, so
    this checks for a file big enough to be a model rather than for mere existence.
    """
    return not missing_weight_files()


def purge_broken_weight_cache() -> list[str]:
    """Delete cache entries that look present but hold no weights.

    `snapshot_download` short-circuits on such an entry, so a run would otherwise tidy
    itself around the corruption and "succeed" with nothing installed.
    """
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    purged = []
    for source in missing_weight_files():
        folder = cache / ("models--" + source.replace("/", "--"))
        if folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
            purged.append(source)
    return purged


def _has_bytes(path: Path) -> bool:
    """Any file at all, following links: the cache links snapshots at blobs."""
    if not path.is_dir():
        return False
    for item in path.rglob("*"):
        try:
            if item.is_file() and item.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def missing_weight_files(min_bytes: int = 1 << 20) -> list[str]:
    """Weight repos in the cache that do not actually hold their model files.

    A Hugging Face cache can be left *looking* complete: `snapshot_download` writes the
    snapshot symlinks first, and an interrupted download leaves them pointing at blobs that
    were never written. `_has_bytes` follows links, so those dangling entries raise
    OSError and are skipped -- but `model_index.json` and a LICENSE genuinely are files, so
    a 160 KB "model" passes a naive check and the run then stalls in the first transformer
    with no error at all (found on a live VM).

    The test here is the one that matters: is there a file big enough to be weights?
    """
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    missing: list[str] = []
    for _, source, _, _ in WEIGHTS:
        folder = cache / ("models--" + source.replace("/", "--"))
        biggest = 0
        if folder.is_dir():
            for item in folder.rglob("*"):
                try:
                    if item.is_file():
                        biggest = max(biggest, item.stat().st_size)
                except OSError:
                    continue
        if biggest < min_bytes:
            missing.append(source)
    return missing


# ----------------------------------------------------------------------------------------
# Installation
# ----------------------------------------------------------------------------------------


def run(cmd: list[str] | str, *, timeout: int = 3600, env: dict | None = None) -> int:
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"  $ {printable}", flush=True)
    merged = {**os.environ, **(env or {})}
    proc = subprocess.run(cmd, shell=isinstance(cmd, str), timeout=timeout, env=merged,
                          check=False)
    return proc.returncode


def must(cmd: list[str] | str, *, what: str, timeout: int = 3600, env: dict | None = None) -> None:
    if run(cmd, timeout=timeout, env=env) != 0:
        raise SystemExit(f"{what} failed (exit code above). Nothing was left half-installed "
                         f"on purpose; re-run when the cause is fixed.")


def install_code(route: str) -> None:
    VENDOR.mkdir(parents=True, exist_ok=True)
    if BUILT_MARKER.is_file() and checkout_installed():
        print("  code already installed; leaving it alone", flush=True)
        return

    if run(["sh", "-c", "uv --version"], timeout=120) != 0:
        must("curl -LsSf https://astral.sh/uv/install.sh | sh", what="uv install",
             timeout=600)

    if not venv_python().is_file():
        # --clear: a previous attempt can leave the directory without a usable interpreter
        # (uv creates the directory first, then populates it). Without it, the retry dies
        # with "a virtual environment already exists" and the reason is invisible.
        must(["uv", "venv", "--python", PYTHON_VERSION, "--seed", "--clear", str(VENV)],
             what="creating the venv", timeout=1200)
    if not venv_python().is_file():
        raise SystemExit(f"the venv at {VENV} has no interpreter at {venv_python()}")

    pip = ["uv", "pip", "install", "--python", str(venv_python())]
    must([*pip, "torch==2.9.1+cu128", "torchvision==0.24.1+cu128",
          "--index-url", TORCH_INDEX, "--extra-index-url", PYPI_INDEX,
          "--index-strategy", "unsafe-best-match"], what="installing torch", timeout=3600)
    must([*pip, SPCONV_VERSION], what="installing spconv", timeout=1800)

    wheel = VENDOR / "flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl"
    if not wheel.is_file() or sha256_of(wheel) != FLASH_ATTN_SHA256:
        print("  fetching flash-attn (prebuilt, pinned by commit and sha256)", flush=True)
        _download(FLASH_ATTN_URL, wheel)
        actual = sha256_of(wheel)
        if actual != FLASH_ATTN_SHA256:
            wheel.unlink(missing_ok=True)
            raise SystemExit(f"flash-attn wheel hash mismatch: {actual} != "
                             f"{FLASH_ATTN_SHA256}. Nothing installed.")
    must([*pip, str(wheel)], what="installing flash-attn", timeout=1800)
    must([*pip, *PINNED], what="installing Hi3D's dependencies", timeout=3600)
    must([*pip, *COARSE_PINNED], what="installing the coarse stage's dependencies",
         timeout=2400)

    clone_at(HI3D_REPO, HI3D_COMMIT, CHECKOUT)
    must([*pip, "--no-deps", "-e", str(CHECKOUT)], what="installing Hi3D", timeout=1800)
    clone_at(HUNYUAN_REPO, HUNYUAN_COMMIT, COARSE_CHECKOUT)
    must([*pip, "--no-deps", "-e", str(COARSE_CHECKOUT)], what="installing the coarse stage",
         timeout=1800)

    verify = subprocess.run(
        [str(venv_python()), "-c",
         ("import torch, spconv, flash_attn, sparc3d, hy3dgen;"
          "print(torch.__version__, torch.version.cuda);"
          "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')")],
        capture_output=True, text=True, timeout=600, check=False,
    )
    if verify.returncode != 0:
        raise SystemExit(
            "the pinned stack does not import on this machine:\n"
            f"{verify.stdout}{verify.stderr}\n"
            "This usually means the card is not Blackwell/Ampere or torch drifted."
        )
    print(verify.stdout.strip(), flush=True)
    BUILT_MARKER.write_text(json.dumps({
        "hi3d_commit": HI3D_COMMIT,
        "hunyuan_commit": HUNYUAN_COMMIT,
        "route": route,
        "python": PYTHON_VERSION,
        "torch": "2.9.1+cu128",
        "spconv": SPCONV_VERSION,
        "flash_attn_sha256": FLASH_ATTN_SHA256,
    }, indent=2, sort_keys=True) + "\n")


def clone_at(repo: str, commit: str, target: Path) -> None:
    if target.is_dir():
        head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False)
        if head.stdout.strip() == commit:
            print(f"  {target.name} already at {commit[:7]}", flush=True)
            return
        shutil.rmtree(target)
    must(["git", "clone", "--quiet", repo, str(target)], what=f"cloning {repo}", timeout=1800)
    must(["git", "-C", str(target), "checkout", "--quiet", commit],
         what=f"checking out {commit[:7]}")


def install_weights() -> None:
    from huggingface_hub import snapshot_download

    for label, source, _, purpose in WEIGHTS:
        print(f"\n  {label} ({source})\n    {purpose}", flush=True)
        snapshot_download(repo_id=source)
    install_background_remover()


def install_background_remover(target: Path | None = None, download=None) -> None:
    """The cut-out every upload without alpha goes through."""
    from bootstrap_matte import install_if_missing

    install_if_missing(target, download)


def _download(url: str, target: Path, opener=urllib.request.urlopen) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with opener(url, timeout=300) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=8 << 20)
    partial.replace(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for agents.")
    parser.add_argument("--check", action="store_true",
                        help="Exit 0 when installed, 1 when not. Installs nothing.")
    parser.add_argument("--code-only", action="store_true",
                        help="Install the code and compile nothing, leaving the weights.")
    parser.add_argument("--weights-only", action="store_true", help="Fetch the weights only.")
    args = parser.parse_args(argv)

    if args.check:
        installed = is_installed()
        print("Hi3D is installed." if installed else "Hi3D is not installed.")
        return 0 if installed else 1

    if host.build_target() != "linux-nvidia":
        print("This installs Hi3D on Linux with an NVIDIA card. Hi3D has no CPU or Apple "
              "Silicon path, so on other machines use Pixal3D, TRELLIS.2 or Hunyuan instead. "
              "Nothing downloaded.")
        return 1

    route, why = choose_route(host.compute_capability(), host.driver_cuda_version())
    if route is None:
        print(f"Cannot install Hi3D here: {why}. Nothing downloaded.")
        return 1

    code = not args.weights_only
    weights = not args.code_only
    print(announcement(code=code, weights=weights))

    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1

    if code:
        install_code(route)
    if weights:
        install_weights()
    print("\nDone. Pick Hi3D in the studio (Create, Advanced: 3D engine), or run:\n"
          "    vendor/hi3d/.venv/bin/python scripts/hi3d_generate.py input.png output.glb\n"
          "\nMake sure the studio shows \"Powered by Hi3D 1.0\" wherever its output is "
          "shipped commercially.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
