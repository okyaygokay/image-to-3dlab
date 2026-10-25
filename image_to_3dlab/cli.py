from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .provenance import (
    finalize_output,
    load_run_manifest,
    validate_run_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn one image into a 3D asset with SF3D or ComfyUI/Hunyuan3D."
    )
    parser.add_argument("image", nargs="?", type=Path, help="PNG/JPEG/WebP input image")
    parser.add_argument(
        "--run-manifest", type=Path, help="JSON run manifest (schema v1)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", help="Run SF3D in this process")
    mode.add_argument(
        "--quality",
        action="store_true",
        help="Run an API-format Hunyuan3D ComfyUI workflow",
    )
    mode.add_argument(
        "--trellis", action="store_true", help="Run TRELLIS.2 through the Mac port"
    )
    mode.add_argument(
        "--hi3d",
        action="store_true",
        help=(
            "Refine a coarse mesh with Hi3D v1.0 (needs NVIDIA, compute capability 8.0+). "
            "Pass --coarse-mesh, or let Hunyuan3D-2 mini make one."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--view",
        dest="views",
        type=Path,
        action="append",
        help=(
            "extra view of the same subject in the same pose (repeatable, trellis2 "
            "only). Views must orbit the camera, not change the pose."
        ),
    )

    fast = parser.add_argument_group("SF3D options")
    fast.add_argument("--sf3d-repo", type=Path, default=Path("vendor/stable-fast-3d"))
    fast.add_argument("--model", default="stabilityai/stable-fast-3d")
    fast.add_argument("--texture-resolution", type=int, default=1024)
    fast.add_argument("--foreground-ratio", type=float, default=0.85)
    fast.add_argument("--remesh", choices=("none", "triangle", "quad"), default="none")
    fast.add_argument("--target-vertices", type=int, default=-1)
    fast.add_argument("--cpu", action="store_true", help="Force SF3D to CPU")
    fast.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="Do not retry on CPU after an MPS OOM",
    )

    trellis = parser.add_argument_group("TRELLIS.2 options")
    trellis.add_argument(
        "--trellis-repo", type=Path, default=Path("vendor/trellis-mac")
    )
    trellis.add_argument("--seed", type=int, default=42)
    trellis.add_argument(
        "--pipeline-type", choices=("512", "1024", "1024_cascade"), default="512"
    )
    trellis.add_argument(
        "--trellis-texture-size", choices=(512, 1024, 2048), type=int, default=1024
    )
    trellis.add_argument(
        "--trellis-bake-target-faces",
        type=int,
        default=50_000,
        help="Triangle budget used by the CPU UV/texture fallback",
    )
    trellis.add_argument(
        "--trellis-pre-simplify-cap",
        type=int,
        default=None,
        help="Face count above which generate.py's crude pre-simplify safety net runs "
             "(maps to the I2L_PRE_CAP env var). Leave unset to use the port's default "
             "of 4,000,000; a manifest may override it via 'pre_simplify_cap'.",
    )
    trellis.add_argument("--steps", type=int)
    trellis.add_argument(
        "--trellis-raw-material",
        dest="trellis_normalize_material",
        action="store_false",
        help="Keep TRELLIS's raw material instead of normalizing it (leaves it transparent)",
    )
    trellis.add_argument(
        "--trellis-remesh",
        action="store_true",
        help="Narrow-band DC remeshing before UV unwrap. generate.py calls this the stage "
             "that targets boundary edges at source; every asset we ship is open",
    )
    trellis.add_argument(
        "--trellis-remesh-project",
        type=float,
        default=0.0,
        help="How far remeshed vertices snap back to the original surface. The vendored "
             "port defaults to 0.9, which measurably worsened boundary edges "
             "(6,008 -> 16,954 on the Forest Variant), so we default to 0.0",
    )
    trellis.add_argument(
        "--trellis-material-mode",
        choices=("matte", "pbr"),
        default="pbr",
        help="pbr keeps TRELLIS's metallic-roughness map, matching the reference "
             "implementation; matte discards it and renders mathematically flat",
    )

    quality = parser.add_argument_group("ComfyUI/Hunyuan3D options")
    quality.add_argument(
        "--workflow", type=Path, help="Workflow exported with ComfyUI Export (API)"
    )
    quality.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    quality.add_argument(
        "--image-node", help="LoadImage node ID; auto-detected when unambiguous"
    )
    quality.add_argument(
        "--output-node", help="Output node ID; otherwise all outputs are searched"
    )
    quality.add_argument(
        "--timeout", type=float, default=3600, help="Job timeout in seconds"
    )
    quality.add_argument("--poll-interval", type=float, default=2)

    hi3d = parser.add_argument_group("Hi3D options")
    hi3d.add_argument(
        "--hi3d-repo", type=Path, default=Path("vendor/Hi3Dv1.0"),
        help="Hi3D checkout, its own venv and CUDA kernels",
    )
    hi3d.add_argument(
        "--coarse-mesh", type=Path,
        help=(
            "An existing coarse mesh of the same object. Without it Hunyuan3D-2 mini "
            "generates one (its weights are territory-restricted; see the licence note)."
        ),
    )
    hi3d.add_argument(
        "--resolution", type=int, choices=(512, 1024, 1536), default=1024,
        help="Sparse extraction grid. 512 is one pass, 1536 is all three.",
    )
    hi3d.add_argument(
        "--hi3d-steps", type=int, default=25,
        help="Scheduler updates per pass (25 is the project's default)",
    )
    hi3d.add_argument(
        "--coarse-steps", type=int, default=30,
        help="Hunyuan3D-2 mini sampler steps (30 suits the turbo variant)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_manifest = None
    manifest_data = None
    if args.run_manifest:
        source_manifest = args.run_manifest.expanduser().resolve()
        try:
            manifest_data = load_run_manifest(source_manifest)
            model = manifest_data["model"]
            backend = model["backend"]
            # A manifest may name one image, or several views of the same subject in
            # the same pose. Extra views replace guessing about unseen sides.
            manifest_input = manifest_data["input"]
            if "views" in manifest_input:
                args.views = [
                    source_manifest.parent / view for view in manifest_input["views"]
                ]
                args.image = args.views[0]
            else:
                args.image = source_manifest.parent / manifest_input["path"]
            args.fast = backend == "sf3d"
            args.quality = backend == "hunyuan-comfyui"
            args.trellis = backend == "trellis2"
            args.hi3d = backend == "hi3d"
            if not (args.fast or args.quality or args.trellis or args.hi3d):
                raise ValueError(f"backend {backend!r} is not implemented yet")
            parameters = model.get("parameters", {})
            args.model = model.get("id", args.model)
            for key in (
                "texture_resolution",
                "foreground_ratio",
                "remesh",
                "target_vertices",
            ):
                if key in parameters:
                    setattr(args, key, parameters[key])
            for key in ("seed", "pipeline_type", "steps"):
                if key in parameters:
                    setattr(args, key, parameters[key])
            if "texture_size" in parameters:
                args.trellis_texture_size = parameters["texture_size"]
            if "bake_target_faces" in parameters:
                args.trellis_bake_target_faces = parameters["bake_target_faces"]
            if "pre_simplify_cap" in parameters:
                args.trellis_pre_simplify_cap = parameters["pre_simplify_cap"]
            if "normalize_material" in parameters:
                args.trellis_normalize_material = bool(parameters["normalize_material"])
            if "material_mode" in parameters:
                args.trellis_material_mode = parameters["material_mode"]
            if "trellis_remesh" in parameters:
                args.trellis_remesh = bool(parameters["trellis_remesh"])
            if "trellis_remesh_project" in parameters:
                args.trellis_remesh_project = float(parameters["trellis_remesh_project"])
            for key in ("resolution", "steps", "coarse_steps"):
                if key in parameters:
                    # `steps` means Hi3D's when refining, TRELLIS's otherwise; the
                    # manifest names one backend, so there is no ambiguity to resolve.
                    setattr(args, "hi3d_steps" if key == "steps" else key, parameters[key])
            args.output_dir = Path(
                manifest_data.get("output", {}).get("directory", args.output_dir)
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"error: invalid run manifest: {exc}", file=sys.stderr)
            return 2
    elif args.image is None or sum((args.fast, args.quality, args.trellis, args.hi3d)) != 1:
        print(
            "error: provide --run-manifest, or IMAGE with exactly one mode",
            file=sys.stderr,
        )
        return 2

    views = [v.expanduser().resolve() for v in (args.views or [args.image])]
    missing = [str(v) for v in views if not v.is_file()]
    if missing:
        print(f"error: input image does not exist: {', '.join(missing)}", file=sys.stderr)
        return 2
    # Provenance and output naming follow the first view.
    image = views[0]
    if len(views) > 1 and not args.trellis:
        print("error: multiple views are only supported by the trellis2 backend", file=sys.stderr)
        return 2
    intent = {
        "use_case": manifest_data.get("use_case", "showcase")
        if manifest_data
        else "showcase",
        "distribution": manifest_data.get("distribution", "private")
        if manifest_data
        else "private",
        "commercial_intent": manifest_data.get("commercial_intent", False)
        if manifest_data
        else False,
    }
    backend = ("sf3d" if args.fast else "trellis2" if args.trellis
               else "hi3d" if args.hi3d else "hunyuan-comfyui")
    policy = manifest_data.get("license_policy", {}) if manifest_data else {}
    try:
        profile = validate_run_policy(
            backend,
            intent["use_case"],
            intent["distribution"],
            policy.get("allow_conditional", True),
        )
    except ValueError as exc:
        print(f"error: license policy: {exc}", file=sys.stderr)
        return 2
    working_output_dir = args.output_dir / ".working"
    working_output_dir.mkdir(parents=True, exist_ok=True)
    trellis_texture_backend = None
    trellis_material_normalized = None
    trellis_material_mode = None

    if not 0 < args.foreground_ratio <= 1:
        print("error: --foreground-ratio must be in (0, 1]", file=sys.stderr)
        return 2
    if args.texture_resolution <= 0:
        print("error: --texture-resolution must be positive", file=sys.stderr)
        return 2
    if args.trellis_bake_target_faces <= 0:
        print("error: --trellis-bake-target-faces must be positive", file=sys.stderr)
        return 2

    try:
        if args.fast:
            if args.cpu:
                os.environ["SF3D_USE_CPU"] = "1"
            from .sf3d_backend import SF3DOptions, generate_sf3d

            result = generate_sf3d(
                image,
                working_output_dir,
                SF3DOptions(
                    repo=args.sf3d_repo,
                    model=args.model,
                    texture_resolution=args.texture_resolution,
                    foreground_ratio=args.foreground_ratio,
                    remesh=args.remesh,
                    target_vertices=args.target_vertices,
                    cpu_fallback=not args.no_cpu_fallback,
                ),
            )
        elif args.trellis:
            from .trellis_backend import TrellisOptions, generate_trellis

            trellis_result = generate_trellis(
                views,
                working_output_dir,
                TrellisOptions(
                    repo=args.trellis_repo,
                    seed=args.seed,
                    pipeline_type=args.pipeline_type,
                    texture_size=args.trellis_texture_size,
                    bake_target_faces=args.trellis_bake_target_faces,
                    steps=args.steps,
                    normalize_material=args.trellis_normalize_material,
                    material_mode=args.trellis_material_mode,
                    pre_simplify_cap=args.trellis_pre_simplify_cap,
                    remesh=args.trellis_remesh,
                    remesh_project=args.trellis_remesh_project,
                ),
            )
            result = trellis_result.asset
            trellis_texture_backend = trellis_result.texture_backend
            trellis_material_normalized = trellis_result.material_normalized
            trellis_material_mode = trellis_result.material_mode
        elif args.hi3d:
            # Hi3D is a pair of stages: a coarse mesh, then refinement. Running it from
            # this CLI means the coarse half runs here too, in the same interpreter, which
            # is why this is one backtick and not two commands for the user to sequence.
            from .hi3d_backend import generate_hi3d

            result = generate_hi3d(
                image,
                working_output_dir,
                repo=args.hi3d_repo,
                coarse_mesh=args.coarse_mesh,
                resolution=args.resolution,
                steps=args.hi3d_steps,
                coarse_steps=args.coarse_steps,
                seed=args.seed,
            )
        else:
            if args.workflow is None:
                raise ValueError(
                    "--quality requires --workflow (ComfyUI API-format JSON)"
                )
            from .comfyui_backend import ComfyUIClient

            result = ComfyUIClient(args.comfy_url).generate(
                image=image,
                workflow_path=args.workflow,
                output_dir=args.output_dir,
                image_node=args.image_node,
                output_node=args.output_node,
                timeout=args.timeout,
                poll_interval=args.poll_interval,
            )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parameters = {
        "model": args.model if args.fast or args.trellis else None,
        "texture_resolution": args.texture_resolution if args.fast else None,
        "foreground_ratio": args.foreground_ratio if args.fast else None,
        "remesh": args.remesh if args.fast else None,
        "target_vertices": args.target_vertices if args.fast else None,
        "seed": args.seed if args.trellis else None,
        "pipeline_type": args.pipeline_type if args.trellis else None,
        "texture_size": args.trellis_texture_size if args.trellis else None,
        "bake_target_faces": args.trellis_bake_target_faces if args.trellis else None,
        "texture_backend": trellis_texture_backend if args.trellis else None,
        "material_normalized": trellis_material_normalized if args.trellis else None,
        "material_mode": trellis_material_mode if args.trellis else None,
        "trellis_remesh": args.trellis_remesh if args.trellis else None,
        "trellis_remesh_project": (
            args.trellis_remesh_project if args.trellis and args.trellis_remesh else None
        ),
        "steps": args.steps if args.trellis else None,
        "pre_simplify_cap": args.trellis_pre_simplify_cap if args.trellis else None,
        "resolution": args.resolution if args.hi3d else None,
        "hi3d_steps": args.hi3d_steps if args.hi3d else None,
        "coarse_steps": args.coarse_steps if args.hi3d else None,
        "coarse_mesh": str(args.coarse_mesh) if args.hi3d and args.coarse_mesh else None,
    }
    result, sidecar = finalize_output(
        generated=result,
        image=image,
        output_root=args.output_dir,
        backend=backend,
        profile=profile,
        intent=intent,
        parameters=parameters,
        source_manifest=source_manifest,
        backend_repo=(
            args.sf3d_repo if args.fast
            else args.trellis_repo if args.trellis
            else args.hi3d_repo if args.hi3d
            else None
        ),
    )
    print(result)
    print(sidecar)
    return 0
