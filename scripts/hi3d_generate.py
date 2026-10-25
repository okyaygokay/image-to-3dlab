#!/usr/bin/env python3
"""Image -> refined GLB through Hi3D v1.0, on an NVIDIA card.

Two stages, one command:

1. A coarse mesh, from Hunyuan3D-2 mini's turbo variant -- unless ``--coarse-mesh``
   supplies one that was made earlier.
2. Hi3D refines it against the same picture, up to an effective 1536^3 sparse grid.

Run it with the checkout's own interpreter::

    vendor/hi3d/.venv/bin/python scripts/hi3d_generate.py in.png out.glb
    vendor/hi3d/.venv/bin/python scripts/hi3d_generate.py in.png out.glb --resolution 1536
    vendor/hi3d/.venv/bin/python scripts/hi3d_generate.py in.png out.glb --check

Writes ``<out>.glb``, ``<out>.json`` (the run manifest) and ``<out>.provenance.json``
(the licence record). ``--shape-only`` stops after the coarse stage.

Images without a real cut-out are matted by our own remover (``image_to_3dlab/matte.py``)
first, for the same reason the other routes do it: a backdrop left in the picture comes
back as geometry.

Progress is printed as machine-readable lines so the studio's job runner can follow it::

    STAGE coarse|refine|export
    STEP  <n>/<total> <label>       # refinement schedule updates, per pass
    PASS  <n>/<total>               # which rung of the resolution ladder is running

``--check`` verifies the stack imports, the card is new enough and the weights are present,
without loading a model or writing anything. Also ``--coarse-only``, which needs no Hi3D
weights at all.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from image_to_3dlab import hi3d_backend as hi3d  # noqa: E402, RUF100
from image_to_3dlab.provenance import sha256_file  # noqa: E402, RUF100

DEFAULT_VENDOR = REPO / "vendor" / "hi3d"
CHECKOUT_NAME = "Hi3Dv1.0"
BUILT_MARKER_NAME = ".i2l-build-complete"

#: The coarse stage's weights come from Tencent and are territory-restricted. Named here
#: rather than imported from the bootstrap so this file can run without it present.
COARSE_MODEL = hi3d.COARSE_MODEL
COARSE_SUBFOLDER = hi3d.COARSE_SUBFOLDER


def checkout(vendor_root: Path) -> Path:
    return vendor_root / CHECKOUT_NAME


def stage(name: str) -> None:
    print(f"STAGE {name}", flush=True)


def progress_line(*parts: Any) -> None:
    print(" ".join(str(part) for part in parts), flush=True)


def validate(settings: dict[str, Any]) -> dict[str, Any]:
    """Settings checked before a model loads, via the same pure function the studio uses."""
    return hi3d.validate_settings(settings)


def build_manifest(*, image: str, output: str, coarse_mesh: str, settings: dict[str, Any],
                   artifacts: dict[str, Any], timings: dict[str, float],
                   coarse_model: str, matte_model: str | None) -> dict[str, Any]:
    return hi3d.build_manifest(
        image=image, output=output, coarse_mesh=coarse_mesh, settings=settings,
        artifacts=artifacts, timings=timings, coarse_model=coarse_model,
        matte_model=matte_model,
    )


def build_provenance(*, image: Path, output: Path, parameters: dict[str, Any],
                     matte_model: str | None, backend_revision: str | None) -> dict[str, Any]:
    """The licence record, via the shared builder so the CLI and studio cannot diverge."""
    return hi3d.build_provenance(
        image_path=image,
        parameters=parameters,
        matte_model=matte_model,
        backend_revision=backend_revision,
        sha256_file=sha256_file,
        now=datetime.now(timezone.utc).isoformat(),
    )


def git_revision(root: Path) -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare_image(image_path: Path, work_dir: Path) -> tuple[Any, str | None, Path]:
    """Cut the subject out, and say which remover did it. Returns (image, model, saved)."""
    from PIL import Image

    from image_to_3dlab import matte

    image = Image.open(image_path).convert("RGBA")
    model: str | None = None
    if not matte.is_matted(image):
        image, model = matte.cut_out(image)
        note = matte.fallback_note(model)
        if note:
            print(f"note: {note}", flush=True)
    prepared = work_dir / f"{image_path.stem}__matted.png"
    image.save(prepared)
    print(f"matted with {model or 'none (already cut out)'} -> {prepared}", flush=True)
    return image, model, prepared


def load_coarse(image: Any, settings: dict[str, Any]) -> Any:
    """Hunyuan3D-2 mini turbo -> one coarse triangle mesh."""
    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    print(f"loading {COARSE_MODEL} / {COARSE_SUBFOLDER}", flush=True)
    started = time.time()
    generator = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        COARSE_MODEL, subfolder=COARSE_SUBFOLDER, use_safetensors=True, device="cuda",
    )
    print(f"loaded coarse pipeline in {time.time() - started:.1f}s", flush=True)
    try:
        mesh = generator(
            image=image,
            num_inference_steps=settings["coarse_steps"],
            octree_resolution=settings["coarse_octree"],
            guidance_scale=5.0,
            generator=torch.manual_seed(settings["seed"]),
            output_type="trimesh",
            enable_pbar=False,
        )[0]
    finally:
        del generator
        gc.collect()
        torch.cuda.empty_cache()
    print(f"coarse mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces", flush=True)
    return mesh


def export_mesh(mesh: Any, output: Path) -> None:
    """Write the refined mesh as a GLB that actually renders.

    Two things trimesh does by default are wrong for a bare refined mesh:

    1. It drops NORMAL unless asked (`include_normals=True`). Without normals every viewer
       shades from a flat default and the model renders as a black silhouette -- the mesh is
       fine and looks broken.
    2. A mesh with no visual gets no material at all. Hi3D produces geometry and no colour
       (painting is a separate stage), so it needs a plain default material, or the GLB
       imports as an untextured black object.

    Neither is visible from the vertex counts, which is why this is checked by reading the
    written file in the tests rather than trusting `export()`.
    """
    import trimesh

    if getattr(mesh.visual, "kind", None) is None:
        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[204, 204, 204, 255],
                metallicFactor=0.0,
                roughnessFactor=0.75,
            )
        )
    mesh.export(str(output), include_normals=True)


def refine(image: Any, coarse: Any, root: Path, settings: dict[str, Any]) -> Any:
    """Hi3D refinement. Progress events become the STEP/PASS lines the studio parses.

    ``root`` is the code checkout; the weights are a separate Hub repository
    (``hi3d.MODEL_REPO``). Pointing the loader at the checkout instead made it read the
    repository's own files as a model and stall in the first transformer with no error.
    """
    from sparc3d import Sparc3DPipeline

    total_passes = hi3d.passes_for(settings["resolution"])
    print(f"loading Hi3D weights {hi3d.MODEL_REPO} (code in {root})", flush=True)
    started = time.time()
    # Not the Hub cache itself: it is symlinks, and Hi3D's loader refuses those on purpose.
    pipeline = Sparc3DPipeline.from_pretrained(hi3d.weights_directory())
    pipeline.to("cuda")
    print(f"loaded Hi3D in {time.time() - started:.1f}s", flush=True)

    def report(event: Any) -> None:
        # The event carries the pass it belongs to; the studio groups on PASS.
        progress_line("PASS", f"{event.pass_index + 1}/{total_passes}")
        progress_line("STEP", f"{event.step}/{event.num_inference_steps}",
                      f"pass {event.pass_index + 1}")

    pipeline.progress_callback = report
    started = time.time()
    result = pipeline(
        image, coarse,
        resolution=settings["resolution"],
        num_inference_steps=settings["steps"],
        seed=settings["seed"],
    )
    print(f"refined in {time.time() - started:.1f}s: {len(result.mesh.vertices)} verts, "
          f"{len(result.mesh.faces)} faces at {result.resolution}^3", flush=True)
    return result


def check(root: Path) -> int:
    """Verify everything a real run needs, without loading a model. Exit 0 when ready."""
    import importlib

    problems: list[str] = []
    for module in ("torch", "spconv", "flash_attn", "sparc3d", "hy3dgen"):
        try:
            importlib.import_module(module)
            print(f"  PASS: {module} imports", flush=True)
        except Exception as exc:  # any failure is the answer
            problems.append(f"{module} does not import: {exc}")

    try:
        import torch

        if not torch.cuda.is_available():
            problems.append("torch cannot see a CUDA device")
        else:
            can, why = hi3d.gpu_can_run(torch.cuda.get_device_capability(0))
            print(f"  PASS: {torch.cuda.get_device_name(0)}, "
                  f"capability {torch.cuda.get_device_capability(0)}", flush=True)
            if not can:
                problems.append(why or "this card cannot run Hi3D")
    except Exception as exc:
        problems.append(f"could not inspect the GPU: {exc}")

    if not hi3d.checkout(root):
        problems.append(f"Hi3D checkout incomplete at {root}")

    for problem in problems:
        print(f"  FAIL: {problem}", flush=True)
    if problems:
        return 1
    print("  PASS: ready to refine", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR,
                        help="Where the checkout and its venv live (vendor/hi3d).")
    parser.add_argument("--coarse-mesh", type=Path,
                        help="Reuse an existing coarse mesh instead of generating one.")
    parser.add_argument("--resolution", type=int, choices=hi3d.RESOLUTIONS,
                        default=hi3d.DEFAULTS["resolution"])
    parser.add_argument("--steps", type=int, default=hi3d.DEFAULTS["steps"])
    parser.add_argument("--coarse-steps", type=int, default=hi3d.DEFAULTS["coarse_steps"])
    parser.add_argument("--coarse-octree", type=int, default=hi3d.DEFAULTS["coarse_octree"])
    parser.add_argument("--seed", type=int, default=hi3d.DEFAULTS["seed"])
    parser.add_argument("--check", action="store_true",
                        help="Verify the stack and the card, then exit. Writes nothing.")
    parser.add_argument("--coarse-only", action="store_true",
                        help="Stop after the coarse mesh; needs no Hi3D weights.")
    parser.add_argument("--keep-coarse", action="store_true",
                        help="Keep the intermediate coarse mesh beside the output.")
    args = parser.parse_args(argv)

    root = checkout(args.vendor)
    if args.check:
        return check(root)

    if args.image is None or args.output is None:
        parser.error("image and output are required unless --check is given")
    if not args.image.is_file():
        print(f"error: input image does not exist: {args.image}", file=sys.stderr)
        return 2

    settings = validate({
        "resolution": args.resolution, "steps": args.steps,
        "coarse_steps": args.coarse_steps, "coarse_octree": args.coarse_octree,
        "seed": args.seed,
    })
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    coarse_path = output.with_name(f"{output.stem}__coarse.glb")
    timings: dict[str, float] = {}
    artifacts: dict[str, Any] = {}
    started = time.time()

    stage("prepare")
    image, matte_model, _matted = prepare_image(args.image, output.parent)

    stage("coarse")
    coarse_started = time.time()
    if args.coarse_mesh is not None:
        import trimesh

        coarse = trimesh.load(str(args.coarse_mesh), force="mesh", process=False)
        print(f"reusing coarse mesh {args.coarse_mesh}", flush=True)
        coarse_source = str(args.coarse_mesh)
    else:
        coarse = load_coarse(image, settings)
        export_mesh(coarse, coarse_path)
        coarse_source = str(coarse_path)
        print(f"coarse mesh -> {coarse_path}", flush=True)
    timings["coarse"] = round(time.time() - coarse_started, 1)
    artifacts["coarse_faces"] = int(len(coarse.faces))

    if args.coarse_only:
        if coarse_source != str(coarse_path):
            export_mesh(coarse, coarse_path)
        timings["total"] = round(time.time() - started, 1)
        print(f"DONE in {timings['total']:.1f}s -> {coarse_path}", flush=True)
        return 0

    stage("refine")
    refine_started = time.time()
    result = refine(image, coarse, root, settings)
    timings["refine"] = round(time.time() - refine_started, 1)

    stage("export")
    export_mesh(result.mesh, output)
    artifacts["faces"] = int(len(result.mesh.faces))
    artifacts["vertices"] = int(len(result.mesh.vertices))
    artifacts["achieved_resolution"] = int(result.resolution)

    if not args.keep_coarse and coarse_source == str(coarse_path):
        coarse_path.unlink(missing_ok=True)

    timings["total"] = round(time.time() - started, 1)
    parameters = {**settings, "coarse_mesh_reused": args.coarse_mesh is not None}

    output.with_suffix(".json").write_text(json.dumps(build_manifest(
        image=str(args.image), output=str(output), coarse_mesh=coarse_source,
        settings=settings, artifacts=artifacts, timings=timings,
        coarse_model=COARSE_MODEL, matte_model=matte_model,
    ), indent=2, sort_keys=True) + "\n")
    output.with_suffix(".provenance.json").write_text(json.dumps(build_provenance(
        image=args.image, output=output, parameters=parameters,
        matte_model=matte_model, backend_revision=git_revision(root),
    ), indent=2, sort_keys=True) + "\n")

    print(f"Manifest: {output.with_suffix('.json')}", flush=True)
    print(f"Provenance: {output.with_suffix('.provenance.json')}", flush=True)
    print(f"DONE in {timings['total']:.1f}s -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    # A finished run has segfaulted before during interpreter shutdown, after every file
    # was written; flush and leave hard instead of reporting a failure that did not happen.
    code = 1
    try:
        code = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
