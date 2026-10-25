"""Hi3D v1.0: high-resolution refinement of a coarse mesh, on any CUDA card.

Hi3D is unusual among the routes here: it is not an image-to-mesh model. It takes a
*picture* and a *coarse mesh of the same object* and samples a far finer mesh from the
pair, up to an effective 1536^3 sparse grid. So it needs a coarse stage in front of it,
and the coarse stage is another backend's job:

    picture -> Hunyuan3D-2 mini turbo (coarse) -> Hi3D (refine) -> GLB

That is why this module exists as a pair of functions rather than a loop of its own: the
coarse half is interchangeable, and any generator that returns a triangle mesh in a
roughly unit-scale frame can stand in (see ``refine_scene``).

**This module imports no torch.** Everything that can be decided from strings and
paths -- settings validation, the resolution ladder, the licence record, argument
building -- is here and unit-tested without a GPU. The torch work lives in
``scripts/hi3d_generate.py``, which is what actually runs.

Licensing is the reason this file is fussy. Hi3D is *commercial-conditional*, not
commercial-clear: commercial use is permitted, but a commercial product must display
"Powered by Hi3D 1.0", and passing 200,000 monthly users or USD 1,000,000 monthly
revenue triggers a separate written licence. Its own weights are also not on
ModelScope, so they are fetched from Hugging Face at a pinned commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Resolutions Hi3D actually supports. The number is the *sparse extraction grid*, not a
#: dense voxel count: 512 runs one refinement pass, 1024 two, 1536 all three.
RESOLUTIONS = (512, 1024, 1536)

#: Passes each rung runs, from the project's own ladder. Used for progress weighting and
#: for the time estimate, because a 1536 run is not three times a 512 run -- it also
#: materialises the second 1.8 B-parameter transformer.
PASSES = {512: 1, 1024: 2, 1536: 3}

#: The smallest card Hi3D runs on: compute capability 8.0 (Ampere) or newer. Turing and
#: older are refused rather than left to fail in a flash-attn kernel.
MIN_COMPUTE_CAPABILITY = 8.0

#: Rough VRAM guidance, NOT a validated limit. The project measured 1536 on a 96 GB H20
#: and explicitly declined to establish a minimum; these numbers are only here so the UI
#: can warn before a run rather than after one.
VRAM_GUIDE_GB = {512: 16, 1024: 24, 1536: 48}

DEFAULTS: dict[str, Any] = {
    "resolution": 1024,
    "steps": 25,
    "seed": 0,
    "coarse_steps": 30,
    "coarse_octree": 256,
}

#: The coarse generator's own settings, kept separate because they belong to Hunyuan's
#: sampler, not Hi3D's. 30 steps and octree 256 are the turbo variant's distilled range.
COARSE_OCTREE_CHOICES = (128, 256, 384, 512)


@dataclass(frozen=True)
class RefineResult:
    """What a refinement produced, and what it achieved rather than asked for."""

    output: str
    resolution: int
    faces: int
    vertices: int
    coarse_faces: int


def validate_settings(raw: Any) -> dict[str, Any]:
    """Defaults filled in and every value checked, before a model loads.

    Pure and importable: the Generate tab calls this on every submitted job, and the test
    suite calls it without a GPU. Errors are ``ValueError`` with the remedy in the message.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("settings must be a JSON object")
    settings = {**DEFAULTS, **raw}
    try:
        for key in ("resolution", "steps", "seed", "coarse_steps", "coarse_octree"):
            settings[key] = int(settings[key])
    except (TypeError, ValueError) as exc:
        raise ValueError("all Hi3D settings must be integers") from exc

    if settings["resolution"] not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}")
    if not 1 <= settings["steps"] <= 100:
        raise ValueError("steps must be 1 to 100 (25 per pass is the default)")
    if not 1 <= settings["coarse_steps"] <= 100:
        raise ValueError("coarse_steps must be 1 to 100 (30 suits the turbo variant)")
    if settings["coarse_octree"] not in COARSE_OCTREE_CHOICES:
        raise ValueError(f"coarse_octree must be one of {COARSE_OCTREE_CHOICES}")
    if settings["seed"] < 0:
        raise ValueError("seed must not be negative")
    return settings


def passes_for(resolution: int) -> int:
    """How many refinement passes a resolution needs."""
    try:
        return PASSES[int(resolution)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}") from exc


def gpu_can_run(compute_capability: tuple[int, int] | None) -> tuple[bool, str | None]:
    """Whether this card can run Hi3D at all, and why not when it cannot.

    ``None`` means the capability could not be read, which is not a refusal: an unreadable
    card is worth trying, and the failure if any arrives from torch with a real message.
    """
    if compute_capability is None:
        return True, None
    major, minor = compute_capability
    capability = major + minor / 10
    if capability < MIN_COMPUTE_CAPABILITY:
        return False, (
            f"Hi3D needs an NVIDIA card with compute capability "
            f"{MIN_COMPUTE_CAPABILITY:.1f} or newer (Ampere onwards); this one reports "
            f"{capability:.1f}. There is no CPU or older-GPU path."
        )
    return True, None


def vram_warning(resolution: int, free_bytes: int | None) -> str | None:
    """A heads-up when the chosen rung probably will not fit, or None.

    A warning, never a refusal: these figures are guidance, the project never published a
    validated minimum, and a run that might work is not ours to block.
    """
    wanted = VRAM_GUIDE_GB.get(int(resolution))
    if wanted is None or free_bytes is None:
        return None
    free_gb = free_bytes / 1024 ** 3
    if free_gb >= wanted:
        return None
    return (
        f"Refining at {resolution}^3 wants roughly {wanted} GB of GPU memory and this card "
        f"has {free_gb:.1f} GB free. The project measured 1536^3 on a 96 GB card and never "
        f"established a lower limit, so this may work or may run out part-way through."
    )


def build_manifest(
    *,
    image: str,
    output: str,
    coarse_mesh: str,
    settings: dict[str, Any],
    artifacts: dict[str, Any],
    timings: dict[str, float],
    coarse_model: str,
    matte_model: str | None,
) -> dict[str, Any]:
    """The run manifest written beside the GLB, in this repo's schema-1 shape."""
    return {
        "schema_version": 1,
        "generator": "hi3d_generate.py",
        "backend": "hi3d",
        "port": "hi3d-official/Hi3Dv1.0 (official, CUDA)",
        "device": "cuda",
        "image": image,
        "output": output,
        "coarse_mesh": coarse_mesh,
        "coarse_model": coarse_model,
        "params": settings,
        "matte_model": matte_model,
        "artifacts": artifacts,
        "timings": timings,
    }


def build_provenance(
    *,
    image_path: Any,
    parameters: dict[str, Any],
    matte_model: str | None,
    backend_revision: str | None,
    sha256_file,
    now: str,
) -> dict[str, Any]:
    """The `.provenance.json` licence record for a Hi3D run.

    Three models are involved and all three are named, because a reviewer reading this
    file needs to know that a coarser, separately-licensed generator made the input mesh,
    and that a third-party encoder read the picture. The Hi3D profile's own conditions are
    copied in rather than summarised, so the record stands alone.
    """
    from image_to_3dlab.provenance import HI3D_BRANDING, LICENSES, MATTE_LICENSES

    profile = LICENSES["hi3d"]
    components: list[dict[str, Any]] = [
        {
            "component": "tencent/Hunyuan3D-2mini",
            "purpose": "coarse mesh generation (the input Hi3D refines)",
            "license": "Tencent Hunyuan 3D 2.0 Community License",
            "commercial_status": "territory-restricted",
            "note": "Not licensed for use in the EU, the UK or South Korea.",
        },
        {
            "component": "facebook/dinov2-large",
            "purpose": "image conditioning",
            "license": "Apache-2.0",
            "commercial_status": "commercial-clear",
        },
        {
            "component": "openai/clip-vit-large-patch14",
            "purpose": "image conditioning",
            "license": "MIT",
            "commercial_status": "commercial-clear",
        },
    ]
    if matte_model:
        components.append({
            "component": f"rembg/{matte_model}",
            "purpose": "background removal",
            "license": MATTE_LICENSES.get(matte_model, "see rembg"),
            "commercial_status": "commercial-clear",
        })
    return {
        "schema_version": 1,
        "created_at": now,
        "input": {
            "path": str(image_path),
            "sha256": sha256_file(image_path),
            "source": "user-provided",
        },
        "model": {
            "backend": "hi3d",
            "parameters": parameters,
            "coarse_model": "tencent/Hunyuan3D-2mini",
        },
        "license": {
            "name": profile.license_name,
            "url": profile.license_url,
            "conditions": list(profile.conditions),
        },
        "branding": HI3D_BRANDING,
        "components": components,
        "software": {"backend_revision": backend_revision},
    }


# ----------------------------------------------------------------------------------------
# The run itself. Everything below imports torch lazily, so this module stays importable
# in the viewer process and in the test suite.
# ----------------------------------------------------------------------------------------

#: Files that prove the Hi3D checkout is installed with its compiled kernels.
BUILD_PROBES = (
    "sparc3d/__init__.py",
    "sparc3d/pipelines/sparc3d_pipeline.py",
)

#: The weights repository, separate from the *code* checkout. Hi3D's loader accepts either
#: a Hub id or a directory of the eight runtime files; pointing it at the git checkout finds
#: the repo's own files and no weights, which is a 160 KB "model" and a hang in the first
#: transformer. So the two are named apart, here and in the wrapper.
MODEL_REPO = "hi3d-official/Hi3Dv1.0"

#: The files Hi3D's loader insists on, relative to a model directory.
MODEL_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "transformer_512/config.json",
    "transformer_512/model.safetensors",
    "transformer_1024/config.json",
    "transformer_1024/model.safetensors",
    "vae/config.json",
    "vae/model.safetensors",
)


def materialise_weights(source: Any, target: Any) -> Any:
    """Copy a Hugging Face snapshot to ``target`` as real files, and return the path.

    Hi3D's loader refuses any symlink in the model directory -- deliberate supply-chain
    hardening, since a symlink could point anywhere -- and a Hugging Face cache is *made*
    of them: every file in a snapshot links into `blobs/`. Pointing `from_pretrained` at
    the cache therefore fails with "unexpected symlink" on the first transformer, which is
    how the live run died.

    Hard links are used when the filesystem allows them (the cache and the target are
    normally on one volume, and then this costs no bytes and no time); a copy is the
    fallback. Either way the result is a directory of ordinary files.
    """
    import os
    import shutil
    from pathlib import Path

    origin = Path(source)
    destination = Path(target)
    destination.mkdir(parents=True, exist_ok=True)
    for relative in MODEL_FILES:
        src = origin / relative
        if not src.is_file():
            raise FileNotFoundError(f"Hi3D weights are missing {relative} in {origin}")
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not dst.is_symlink() and dst.stat().st_size == src.stat().st_size:
            continue
        dst.unlink(missing_ok=True)
        # `src.resolve()` matters: the cache entry is itself a symlink, and hard-linking a
        # symlink makes another symlink (or fails), which is the thing being avoided.
        try:
            os.link(src.resolve(), dst)
        except OSError:
            shutil.copy2(src.resolve(), dst)
    return destination


#: Where a materialised copy goes when a caller does not name one. Under the Hugging Face
#: cache so it is discarded with it, and separate from the snapshot it was built from.
MATERIALISED_ENV = "I2L_HI3D_WEIGHTS"


def weights_directory() -> Any:
    """A directory of real files Hi3D's loader will accept, built from the Hub cache.

    `snapshot_download` first, so a cold machine fetches; then a materialised copy, because
    the cache it writes is all symlinks and Hi3D refuses those.
    """
    import os
    from pathlib import Path

    from huggingface_hub import snapshot_download

    override = os.environ.get(MATERIALISED_ENV)
    if override:
        return Path(override)
    snapshot = snapshot_download(repo_id=MODEL_REPO)
    target = Path(snapshot).parent.parent / "materialised"
    return materialise_weights(snapshot, target)

#: The coarse generator, pinned. Its weights are territory-restricted; see the catalogue.
COARSE_MODEL = "tencent/Hunyuan3D-2mini"
COARSE_SUBFOLDER = "hunyuan3d-dit-v2-mini-turbo"

#: Longest edge Hi3D's VAE octree takes for the coarse stage.
COARSE_OCTREE = 256


def checkout(repo: Any) -> Any:
    """The Hi3D root, checked before anything expensive happens."""
    from pathlib import Path

    root = Path(repo).expanduser().resolve()
    missing = [name for name in BUILD_PROBES if not (root / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Hi3D checkout not found at {root} (missing {', '.join(missing)}). "
            "Run scripts/bootstrap_hi3d.py first, or pass --hi3d-repo."
        )
    return root


def coarse_mesh_for(image: Any, work_dir: Any, *, steps: int, seed: int) -> Any:
    """A coarse triangle mesh for ``image``, from Hunyuan3D-2 mini's turbo variant.

    Any generator that returns a `trimesh.Trimesh` in a roughly unit-scale frame would do;
    this one is used because it is small, fast and already in this repo's orbit. It is a
    separate licence from Hi3D's and the catalogue says so.
    """
    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    generator = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        COARSE_MODEL,
        subfolder=COARSE_SUBFOLDER,
        use_safetensors=True,
        device="cuda",
    )
    try:
        mesh = generator(
            image=image,
            num_inference_steps=steps,
            octree_resolution=COARSE_OCTREE,
            guidance_scale=5.0,
            generator=torch.manual_seed(seed),
            output_type="trimesh",
            enable_pbar=False,
        )[0]
    finally:
        # Hi3D wants the whole card; the coarse generator is finished with by now.
        del generator
        import gc

        gc.collect()
        torch.cuda.empty_cache()
    return mesh


def refine(
    image: Any,
    coarse_mesh: Any,
    repo: Any,
    *,
    resolution: int,
    steps: int,
    seed: int,
    progress: Any = None,
) -> Any:
    """Refine ``coarse_mesh`` against ``image`` with Hi3D, and return the result mesh.

    ``repo`` is the *code* checkout; the weights come from ``MODEL_REPO``. Passing the
    checkout here instead loaded the repository's own text files as a "model" -- the
    loader happily accepted the directory and then stalled at the first transformer.
    """
    from sparc3d import Sparc3DPipeline

    checkout(repo)  # fail early and clearly if the code is not installed
    pipeline = Sparc3DPipeline.from_pretrained(weights_directory())
    pipeline.to("cuda")
    if progress is not None:
        pipeline.progress_callback = progress
    return pipeline(
        image,
        coarse_mesh,
        resolution=resolution,
        num_inference_steps=steps,
        seed=seed,
    )


def generate_hi3d(
    image_path: Any,
    output_dir: Any,
    *,
    repo: Any,
    coarse_mesh: Any = None,
    resolution: int = 1024,
    steps: int = 25,
    coarse_steps: int = 30,
    seed: int = 0,
    progress: Any = None,
) -> Any:
    """Image -> refined GLB, end to end, and return the written path.

    Two stages, one call: cut the picture out, get a coarse mesh (supplied or generated),
    refine it, write the GLB. The intermediate coarse mesh is kept beside the output so a
    run can be re-refined without paying for the coarse stage again.
    """
    import trimesh
    from PIL import Image

    from image_to_3dlab import matte

    settings = validate_settings({
        "resolution": resolution, "steps": steps, "coarse_steps": coarse_steps, "seed": seed,
    })
    root = checkout(repo)
    out_dir = __import__("pathlib").Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{image_path.stem}_hi3d.glb"
    coarse_path = out_dir / f"{image_path.stem}_hi3d_coarse.glb"

    image = Image.open(image_path).convert("RGBA")
    if not matte.is_matted(image):
        image, _ = matte.cut_out(image)

    if coarse_mesh is not None:
        mesh = trimesh.load(str(coarse_mesh), force="mesh", process=False)
    else:
        mesh = coarse_mesh_for(image, out_dir,
                               steps=settings["coarse_steps"], seed=settings["seed"])
        mesh.export(str(coarse_path))

    result = refine(image, mesh, root,
                    resolution=settings["resolution"], steps=settings["steps"],
                    seed=settings["seed"], progress=progress)
    result.mesh.export(str(output))
    return output
