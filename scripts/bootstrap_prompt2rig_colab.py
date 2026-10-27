#!/usr/bin/env python3
"""Install the prompt-to-motion backend on a rented Colab VM: H3, GVHMR, WiLoR.

    python scripts/bootstrap_prompt2rig_colab.py --yes

Three models, one venv at `vendor/prompt2rig/.venv`:

* **MiniMax H3** (video, the clip a motion is read from) — the official bf16 checkpoint
  is ~124 GB even before `transformer_ref/` and the original SGLang copies this route
  never reads, which is more than a Colab disk holds comfortably. So each big component
  is downloaded alone, quantized to int8 with the torchao recipe from the diffusers
  model card, saved into `vendor/prompt2rig/h3/`, and its raw copy deleted — peak disk
  stays near 100 GB and settles near 70 GB, and a warm session loads the int8 weights
  straight back without paying either cost again.
* **GVHMR** (video → SMPL-X motion) — the `gvhmr` PyPI package with its preprocessing
  extra; its own checkpoints (~7 GB) are fetched on first use, into the directory named
  below. The registration-gated SMPL-X body model comes from a public Hugging Face
  mirror, pinned by revision and verified by checksum.
* **WiLoR** (fingers) — the WiLoR-mini package at a pinned commit, with the authors'
  weights from their Hugging Face repository (which also carries the MANO hand model
  WiLoR needs, so nothing is registration-gated here). MediaPipe's hand landmarker is
  installed beside it as the fallback when WiLoR's weights are absent.

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
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import host  # noqa: E402, RUF100

VENDOR = REPO / "vendor" / "prompt2rig"
DRIVER = REPO / "vendor" / "prompt2rig-driver"
DRIVER_VENV = DRIVER / ".venv"
VENV = VENDOR / ".venv"
BODY_MODELS = VENDOR / "body_models"
WEIGHTS = VENDOR / "weights"
H3_DIR = VENDOR / "h3"
H3_RAW = DRIVER / "h3-raw"
PYTHON_VERSION = "3.11"
# The G4's Blackwell card needs the cu128 wheels (sm_120), never PyPI's cu126 default
# build. Of those, 2.10.0 is the oldest diffusers 0.42 can live with: its loader chain
# imports ScalingType from torch.nn.functional, a symbol 2.9 and older do not carry
# (probed on the real wheels; found on the first live quantize). gvhmr's own torch bound
# is a bare >=2.4 — the <2.8 lives only in its cu12x extras, which are not installed.
TORCH_VERSION = "2.10.0"
TORCHVISION_VERSION = "0.25.0"

# Everything pinned: the mirrors move, and a moved mirror is a broken run months later.
GVHMR_VERSION = "1.6.1"
DIFFUSERS_VERSION = "0.42.0"
WILOR_MINI_COMMIT = "ebec42f94c389070cdd7dda6fd1bf0b4a659c960"
H3_REPO, H3_REVISION = "MiniMaxAI/MiniMax-H3", "42ed227ee7df40d41602854ae760620d6eb651fe"
SMPLX_REPO, SMPLX_REVISION = "artabatest/smplx_model", "92f4c34af9104fe983c61c138b59223d5f10b16f"
SMPLX_FILE = "human_models/human_model_files/models_smplx_v1_1/smplx/SMPLX_NEUTRAL.npz"

#: The alternate video engine, installed on demand (`--ltx2-only`): LTX-2.5 distilled,
#: the anti-flicker pick — its pitch is temporal stability in high-motion shots, and
#: flicker is what motion recovery turns into shimmer. It needs diffusers *main*
#: (LTX2Pipeline is not in a release), so it gets its own venv rather than gambling
#: the H3 one on an upgrade. The weights come from ModelScope's mirror of the
#: Diffusers pack: the Hugging Face original is click-through gated and wants an
#: account's token, the mirror is ungated — no account, no token, no click.
LTX2_MS_MODEL, LTX2_MS_REVISION = "Lightricks/LTX-2.5-Diffusers", \
    "72d538df36d92956d682557f39409d3d072626d8"
LTX2_VENV = REPO / "vendor" / "ltx2" / ".venv"
LTX2_MODEL = REPO / "vendor" / "ltx2" / "model"
#: What `model_index.json` actually loads and nothing else. The mirror carries the
#: whole 174 GB repo, and four loads of freight the pipeline never reads:
#: `transformer_full/` (the un-distilled base DiT, 38 GB), a second copy of the
#: distilled transformer re-sharded eight ways (38 GB — the index names the four-shard
#: set), `connectors/`' unsharded single-file duplicate (6.3 GB — its index names the
#: shards), and `prompt_enhancer/` + `processor/` (10.2 GB — optional components whose
#: job, rewriting short prompts into long captions, this chain does itself in
#: `ltx_motion_prompt`; the generator passes them as None so nothing looks for them).
#: The two latent upsamplers and the root LoRA are not in `model_index.json` at all.
LTX2_INCLUDE = [
    "model_index.json",
    "transformer/*.json", "transformer/*-of-00004.safetensors",
    "text_encoder/*", "tokenizer/*", "scheduler/*",
    "connectors/*.json", "connectors/*-of-00002.safetensors",
    "vae/*", "audio_vae/*", "diffusion_decoder/*", "duration_head/*", "vocoder/*",
]
LTX2_GB = 71.0
DIFFUSERS_MAIN_COMMIT = "06a9cfb86ae8aa69c6a220ba0da0f37b819dde6a"
LTX2_TRANSFORMERS, LTX2_ACCELERATE = "5.21.0", "1.15.0"
MODELSCOPE_VERSION = "1.40.2"
#: PyAV: the I2V pipeline re-compresses the conditioning frame as H.264 before feeding
#: it (training-matched CRF conditioning, the model's own default) and refuses to run
#: without it — found on the first live run.
AV_VERSION = "19.0.2"
#: The ltx2 venv's build id, bumped whenever its package set changes: a VM warm on an
#: older build re-runs the venv step (minutes) without re-paying the weights, whose own
#: marker carries the revision. Build 2 added PyAV; build 1 was the unversioned "ok".
LTX2_BUILD = "2"
LTX2_LICENCE = (
    "LTX-2.5: LTX-2.x Community License — free including commercial use under\n"
    "  $10M annual revenue, no mandatory branding; above that, a paid agreement with\n"
    "  Lightricks. https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x"
)
# What the pinned revisions actually serve, each set after downloading it and checking
# the file's shape: a SMPL-X v1.1 layout (10475 verts, 20908 faces, a 55-joint
# J_regressor, the hand PCA blocks). The first recording of these checksums was wrong
# and cost a live VM run at its first download.
SMPLX_SHA256 = "376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992"
WILOR_REPO, WILOR_REVISION = "warmshao/WiLoR-mini", "b00adea9a6843bbb4c9042109c5eb29ab2a59dea"
WILOR_SHA256 = {
    "MANO_RIGHT.pkl": "45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767",
    "detector.pt": "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033",
    "wilor_final.ckpt": "3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2",
    "mano_mean_params.npz": "efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988",
}
MEDIAPIPE_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                      "hand_landmarker/float16/latest/hand_landmarker.task")
MEDIAPIPE_TASK_SHA256 = None  # 'latest' moves; MediaPipe is the fallback, WiLoR is pinned

#: The diffusers-format pieces of the H3 repo, by how they are installed. The two big
#: ones are quantized on the way in; everything else is small and copied as-is. The
#: original-format `FL2VA/`, `Ref2VA/` and the unused `transformer_ref/` partition are
#: deliberately never downloaded (~124 GB this route would throw away).
H3_SMALL = ("*.json", "tokenizer/*", "processor/*", "image_processor/*", "video_processor/*",
            "scheduler/*", "audio_scheduler/*", "vae/*", "audio_vae/*")
H3_BIG = ("transformer", "text_encoder")

PACKAGE_GB = 20.0
DOWNLOAD_GB = 70.0  # what stays on disk; the bf16 originals pass through in stages


def driver_python() -> Path:
    return host.venv_python(DRIVER)


def hf() -> str:
    """The driver venv's `hf` command — not on PATH, so it is addressed directly."""
    return str(driver_python().parent / "hf")


def venv_python() -> Path:
    return host.venv_python(VENDOR)


#: The driver venv's one package: pinned (the [cli] extra vanished in hub 2.x — the
#: `hf` command itself stayed in core; verified locally that this version carries it).
HF_HUB_VERSION = "2.3.0"


def must(command: list[str] | str, *, what: str, timeout: float,
         cwd: Path | None = None) -> None:
    """Run one install step, or say which one failed. A string runs through the shell."""
    printable = command if isinstance(command, str) else " ".join(shlex.quote(str(p)) for p in command)
    print(f"  $ {printable}", flush=True)
    done = subprocess.run(command, shell=isinstance(command, str), timeout=timeout,
                          check=False, cwd=cwd)
    if done.returncode != 0:
        raise SystemExit(f"{what} failed (exit {done.returncode})")


#: Pinned so the uv.toml below keeps meaning what it meant when this was written (a local
#: uv of the same version reproduced the chumpy failure and verified the fix).
UV_VERSION = "0.12.10"


def ensure_uv() -> None:
    if subprocess.run(["sh", "-c", "uv --version"], capture_output=True,
                      timeout=120, check=False).returncode == 0:
        return
    must(f"curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh | sh",
         what="uv install", timeout=600)


def marker(name: str) -> Path:
    return DRIVER / f".{name}-done"


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        path.unlink(missing_ok=True)
        raise SystemExit(f"{path.name}: sha256 {digest.hexdigest()} != pinned {expected}")


def hf_download(repo: str, filename: str, destination: Path, revision: str,
                sha256: str | None = None) -> Path:
    """One pinned file, verified, flattened to `destination`.

    `hf download` keeps the file's repository-relative path inside `--local-dir`, so the
    download lands in a scratch directory and is moved to where the wrapper expects it —
    a warm session short-circuits on the destination's checksum, which passed before."""
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = DRIVER / "hf-scratch"
    must([hf(), "download", repo, filename, "--revision", revision,
          "--local-dir", str(scratch)], what=f"downloading {filename}", timeout=7200)
    import shutil

    shutil.move(str(scratch / filename), str(destination))
    shutil.rmtree(scratch, ignore_errors=True)
    if sha256:
        verify_sha256(destination, sha256)
    return destination


def quantize_script(part: str) -> str:
    """Load one bf16 H3 component, quantize it to int8 (torchao, the model card's
    recipe) and save it beside the small parts. The raw copy is dropped by the caller."""
    keep = repr(["proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
                 "token_refiner", "norm_out", "proj_out", "audio_proj_out"])
    keep_enc = repr(["model.visual", "model.language_model.embed_tokens",
                     "model.language_model.norm", "lm_head"])
    loader = {
        "transformer": (
            "MiniMaxH3Transformer3DModel.from_pretrained(src, subfolder='transformer',\n"
            "        dtype=torch.bfloat16, quantization_config=TorchAoConfig(\n"
            f"            Int8WeightOnlyConfig(version=2), modules_to_not_convert={keep}))"),
        "text_encoder": (
            "Qwen3VLForConditionalGeneration.from_pretrained(src, subfolder='text_encoder',\n"
            "        dtype=torch.bfloat16, quantization_config=TransformersTorchAoConfig(\n"
            f"            Int8WeightOnlyConfig(version=2), modules_to_not_convert={keep_enc}))"),
    }[part]
    return f"""import torch
from diffusers import MiniMaxH3Transformer3DModel, TorchAoConfig
from transformers import Qwen3VLForConditionalGeneration
from transformers import TorchAoConfig as TransformersTorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig

src, dst = {str(H3_RAW)!r}, {str(H3_DIR)!r}
model = {loader}
model.requires_grad_(False)
model.save_pretrained(dst + '/{part}')
print('quantized {part} to int8')
"""


def quantize_part(part: str) -> None:
    must([str(venv_python()), "-c", quantize_script(part)],
         what=f"quantizing {part} to int8", timeout=7200)
    must(["rm", "-rf", str(H3_RAW / part)], what=f"dropping the raw {part}", timeout=600)
    marker(f"h3-{part}").write_text("ok")


def ensure_h3() -> None:
    """The staged install described in the module docstring."""
    if all(marker(f"h3-{part}").is_file() for part in H3_BIG) and \
            (H3_DIR / "modular_model_index.json").is_file():
        return
    H3_DIR.mkdir(parents=True, exist_ok=True)
    if not (H3_DIR / "modular_model_index.json").is_file():
        includes = [arg for pattern in H3_SMALL for arg in ("--include", pattern)]
        must([hf(), "download", H3_REPO, *includes, "--revision", H3_REVISION,
              "--local-dir", str(H3_DIR)],
             what="downloading H3's small components", timeout=3600)
    for part in H3_BIG:
        if marker(f"h3-{part}").is_file():
            continue
        must([hf(), "download", H3_REPO, "--include", f"{part}/*",
              "--revision", H3_REVISION, "--local-dir", str(H3_RAW)],
             what=f"downloading H3's {part} (bf16, ~62 GB)", timeout=14400)
        quantize_part(part)


def ensure_body_model() -> None:
    target = BODY_MODELS / "smplx" / "SMPLX_NEUTRAL.npz"
    if not target.is_file():
        hf_download(SMPLX_REPO, SMPLX_FILE, target, SMPLX_REVISION, SMPLX_SHA256)
        print(f"  SMPL-X body model: {target}", flush=True)


def ensure_wilor() -> None:
    root = WEIGHTS / "wilor" / "pretrained_models"
    for name, sha256 in WILOR_SHA256.items():
        hf_download(WILOR_REPO, f"pretrained_models/{name}", root / name, WILOR_REVISION, sha256)
    task = WEIGHTS / "hand_landmarker.task"
    if not task.is_file():
        must(["curl", "-LsSf", "-o", str(task), MEDIAPIPE_TASK_URL],
             what="downloading MediaPipe's hand landmarker", timeout=600)
    print(f"  WiLoR weights: {root}", flush=True)


def ensure_venv() -> None:
    if venv_python().is_file() and marker("venv").is_file():
        return
    # torch first, alone, and always from the cu128 index: PyPI's default Linux torch is
    # a cu126 build a Blackwell card cannot run, and the pin above was never published to
    # cu130 — the newer index a CUDA 13 driver would suggest — so the driver's newest
    # build is the wrong thing to ask for here. A newer driver runs older CUDA wheels
    # fine; what must be checked is the floor.
    require_cu128_driver()
    index = host.TORCH_INDEX + "cu128"
    # chumpy — the SMPL stack's ancient dependency, pulled from git — imports pip inside
    # its setup.py, and uv's isolated build environment carries no pip (found on the
    # first live install). This file tells uv to add one, and the installs below run
    # from VENDOR because that is the directory uv reads its config from. The directory
    # itself is born here: nothing before this line has made it yet (uv venv would, but
    # the config must exist first).
    VENDOR.mkdir(parents=True, exist_ok=True)
    (VENDOR / "uv.toml").write_text(
        "# Written by bootstrap_prompt2rig_colab.py: chumpy's setup.py imports pip,\n"
        "# which uv's isolated build environment does not carry.\n"
        "[extra-build-dependencies]\nchumpy = [\"pip\"]\n")
    must(["uv", "venv", "--python", PYTHON_VERSION, "--seed", "--clear", str(VENV)],
         what="creating the prompt2rig venv", timeout=1200)
    must(["uv", "pip", "install", "--python", str(venv_python()), "--index-url", index,
          f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}"],
         what=f"installing torch {TORCH_VERSION} for this driver", timeout=7200,
         cwd=VENDOR)
    # transformers 5, not 4: diffusers 0.42 needs huggingface-hub>=1.32 and every 4.57.x
    # needs hub<1.0 — the two eras cannot share a venv (found on the first live install;
    # uv resolves this pairing to transformers 5.21 / hub 2.3, verified locally).
    must(["uv", "pip", "install", "--python", str(venv_python()),
          f"gvhmr[preproc]=={GVHMR_VERSION}",
          f"diffusers=={DIFFUSERS_VERSION}", "transformers>=5,<6",
          "accelerate", "torchao", "mediapipe"],
         what="installing gvhmr, diffusers and mediapipe", timeout=7200,
         cwd=VENDOR)
    # WiLoR-mini pins ultralytics==8.1.34 and torch<=2.5 — the first impossible next to
    # gvhmr's preproc extra (ultralytics>=8.2), the second next to the Blackwell torch.
    # Everything it actually imports is already here through gvhmr's own
    # dependencies, so --no-deps installs the code and lets its stale pins lose.
    must(["uv", "pip", "install", "--python", str(venv_python()), "--no-deps",
          f"wilor-mini @ git+https://github.com/warmshao/WiLoR-mini@{WILOR_MINI_COMMIT}"],
         what="installing WiLoR-mini without its stale pins", timeout=1200,
         cwd=VENDOR)
    must(["uv", "pip", "install", "--python", str(venv_python()), "roma"],
         what="installing roma, WiLoR's one import gvhmr does not bring", timeout=600,
         cwd=VENDOR)
    marker("venv").write_text("ok")


def ensure_autorig() -> None:
    """The retarget stage runs `kimodo_retarget.py` in SkinTokens' venv (it carries
    Blender's bpy), so that venv belongs to this backend's VM too. The sibling bootstrap
    is idempotent and marker-gated; a warm session skips straight past it."""
    if marker("autorig").is_file():
        return
    done = subprocess.run([sys.executable, str(REPO / "scripts" / "bootstrap_autorig_colab.py"),
                           "--yes"], timeout=7200, check=False)
    if done.returncode != 0:
        raise SystemExit("the SkinTokens install (for the retarget stage) failed")
    marker("autorig").write_text("ok")


def ltx2_venv_python() -> Path:
    return LTX2_VENV / "bin" / "python"


def require_cu128_driver() -> None:
    """Both torch venvs pay the same wheel floor, so the check lives in one place."""
    driver = host.driver_cuda_version()
    if driver is None or driver < (12, 8):
        why = "no NVIDIA driver answered (nvidia-smi)" if driver is None else \
            f"the driver supports CUDA {driver[0]}.{driver[1]}; the cu128 wheels need 12.8 or newer"
        raise SystemExit(f"cannot install a torch venv here: {why}")


def modelscope() -> str:
    """The driver venv's `modelscope` command — not on PATH, so it is addressed
    directly. Installed on demand: only the LTX-2.5 download needs it."""
    return str(driver_python().parent / "modelscope")


def ensure_ltx2() -> None:
    """The alternate video engine (LTX-2.5 distilled): its own venv, because LTX2Pipeline
    lives in diffusers main and gambling the H3 venv on an upgrade would put two moving
    targets in one environment. Two markers split the cost: the build one says the venv
    matches this script's package set, the weights one (carrying the revision) says the
    ~71 GB are on disk — a warm VM re-pays neither, a build bump re-pays only the venv,
    a revision bump re-verifies only the weights. Nothing runs at all unless a run
    actually asks for `--video-engine ltx2`."""
    if marker("ltx2").is_file() and marker("ltx2").read_text() == LTX2_BUILD:
        return
    free_gb = shutil.disk_usage(str(REPO)).free / 1e9
    if free_gb < LTX2_GB + 10:
        raise SystemExit(
            f"LTX-2.5 needs about {LTX2_GB:.0f} GB and this VM has {free_gb:.0f} GB free; "
            "skipping it (the default H3 engine needs no extra room)")
    require_cu128_driver()
    index = host.TORCH_INDEX + "cu128"
    LTX2_VENV.parent.mkdir(parents=True, exist_ok=True)
    must(["uv", "pip", "install", "--python", str(driver_python()),
          f"modelscope=={MODELSCOPE_VERSION}"],
         what="installing the modelscope CLI for the LTX-2.5 download", timeout=1200)
    # Weights before venv: a venv build that fails must not re-trigger the weights pass.
    weights = marker("ltx2-weights")
    if not (weights.is_file() and weights.read_text() == LTX2_MS_REVISION):
        print(f"downloading LTX-2.5 ({LTX2_GB:.0f} GB, one time on this VM) from ModelScope "
              f"(ungated mirror; pinned revision {LTX2_MS_REVISION[:8]})", flush=True)
        must([modelscope(), "download", LTX2_MS_MODEL, "--revision", LTX2_MS_REVISION,
              "--include", *LTX2_INCLUDE, "--local_dir", str(LTX2_MODEL)],
             what="downloading LTX-2.5", timeout=14400)
        weights.write_text(LTX2_MS_REVISION)
    must(["uv", "venv", "--python", PYTHON_VERSION, "--seed", "--clear", str(LTX2_VENV)],
         what="creating the ltx2 venv", timeout=1200)
    must(["uv", "pip", "install", "--python", str(ltx2_venv_python()), "--index-url", index,
          f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}"],
         what=f"installing torch {TORCH_VERSION} for ltx2", timeout=7200)
    must(["uv", "pip", "install", "--python", str(ltx2_venv_python()),
          f"diffusers @ git+https://github.com/huggingface/diffusers@{DIFFUSERS_MAIN_COMMIT}",
          f"transformers=={LTX2_TRANSFORMERS}", f"accelerate=={LTX2_ACCELERATE}",
          f"av=={AV_VERSION}"],
         what="installing diffusers main for ltx2", timeout=3600)
    marker("ltx2").write_text(LTX2_BUILD)


def announcement(*, ltx2: bool = False) -> str:
    if ltx2:
        return "\n".join([
            "", "About to install:", "",
            "  backend: the LTX-2.5 video engine (the --video-engine ltx2 stage of",
            "           prompt-to-motion)",
            "  route:   this rented Colab VM, into vendor/ltx2/",
            (f"  size:    about 8 GB of packages (CUDA torch, cu128 on a G4) and"
            f" {LTX2_GB:.0f} GB of weights that stay on disk"),
            "  source:  ModelScope's ungated mirror of Lightricks/LTX-2.5-Diffusers,",
            f"           pinned at revision {LTX2_MS_REVISION[:8]} — the Hugging Face",
            "           original is click-through gated, the mirror needs no account;",
            "           only what model_index.json loads, ~71 of the mirror's 174 GB;",
            f"           diffusers main at commit {DIFFUSERS_MAIN_COMMIT[:8]}, because",
            "           LTX-2.5 is not in a release",
            f"  licence: {LTX2_LICENCE}",
            "",
        ])
    return "\n".join([
        "", "About to install:", "",
        "  backend: Prompt-to-motion (MiniMax H3 video · GVHMR motion · WiLoR fingers)",
        "  route:   this rented Colab VM, into vendor/prompt2rig/",
        (f"  size:    about {PACKAGE_GB:.0f} GB of packages (CUDA torch, cu128 on a G4)"
         f"           and {DOWNLOAD_GB:.0f} GB of weights that stay on disk; the bf16 H3"
         "           originals (~62 GB at a time) pass through and are deleted once"
         "           their int8 copies exist"),
        "  source:  MiniMaxAI/MiniMax-H3, quantized to int8 at a pinned revision; the",
        "           gvhmr package at a pinned version with its own checkpoints; WiLoR-mini",
        "           at a pinned commit with the WiLoR authors' weights; the SMPL-X body",
        "           model from a public Hugging Face mirror, checksum-verified",
        "  licence: MiniMax H3 Community License (US/EU/UK/KR excluded without an",
        "           application); GVHMR, WiLoR, SMPL-X and MANO are research-only — the",
        "           viewer writes all of this into every clip's provenance record",
        "  optional: the LTX-2.5 video engine installs on demand (--video-engine ltx2,",
        f"            ~{LTX2_GB:.0f} GB more, from an ungated ModelScope mirror)",
        "",
    ])


def ensure_driver() -> None:
    """The `hf` CLI every download drives through, in its own tiny venv."""
    DRIVER.mkdir(parents=True, exist_ok=True)
    ensure_uv()
    if not driver_python().is_file():
        must(["uv", "venv", "--python", PYTHON_VERSION, "--seed", "--clear", str(DRIVER_VENV)],
             what="creating the driver venv", timeout=1200)
    must(["uv", "pip", "install", "--python", str(driver_python()),
          f"huggingface_hub=={HF_HUB_VERSION}"],
         what="installing the driver's one package", timeout=1200)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the studio and for agents.")
    parser.add_argument("--ltx2-only", action="store_true",
                        help="Install only the LTX-2.5 video engine (its own venv plus "
                             f"~{LTX2_GB:.0f} GB of weights): what a run asking for "
                             "`--video-engine ltx2` builds on demand. Nothing here is "
                             "needed for the default H3 engine.")
    args = parser.parse_args(argv)

    print(announcement(ltx2=args.ltx2_only), flush=True)
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1

    ensure_driver()
    if args.ltx2_only:
        ensure_ltx2()
        print("\nDone. The prompt-to-motion chain can now use --video-engine ltx2.")
        return 0
    ensure_venv()
    ensure_body_model()
    ensure_wilor()
    ensure_h3()
    ensure_autorig()
    print("\nDone. A prompt becomes a motion with:\n"
          "    python scripts/colab_generate.py request.json motion.npz "
          "--backend prompt2rig --yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
