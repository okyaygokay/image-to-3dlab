#!/usr/bin/env python3
"""Turn a motion description into an animated model, on a rented Colab VM.

    python scripts/prompt2rig_colab.py rigged.glb animated.glb \
        --prompt "a knight swinging a sword" [--seconds 6] [--seed 3] \
        [--arm-spread 12] [--speed 1.1]

The remote wrapper `scripts/colab_generate.py` drives for the `prompt2rig` backend,
installed by `scripts/bootstrap_prompt2rig_colab.py`. Five stages, all on the VM:

1. **Video** — LTX-2.5 distilled acts the prompt out: one person, full body, planted on
   the spot, static camera, ~6 s at 960×544. It won the A/B that decided this default
   (same rig, same prompt, against H3): recovery shimmer 5.6 vs 25.1, finger jitter
   5.8 vs 26.7, hands tracked 97%/100% vs 82%/91% — the flicker H3's clips carried is
   what motion recovery turned into shimmer and broken-looking hands. Its ~71 GB of
   weights install on the VM on first use, from an ungated ModelScope mirror.
   `--video-engine h3` keeps MiniMax H3 (`fl2va`, int8) available: motion recovery does
   not need more than the smaller canvas, which is ~2.3× faster per step than its
   trained 768p one. Two takes by default (`--tries`): the video model rolls a
   different actor each seed, and a take whose hands track badly or whose feet wander
   is thrown away.
2. **Body motion** — GVHMR lifts SMPL-X motion from the clip (a generated clip has a
   static camera, which is its `-s` fast path).
3. **Fingers** — WiLoR (or MediaPipe) tracks the hands on the full-rate clip, in parallel
   thinking, and `prompt2rig_hands` hands them over as wrist-relative world keypoints.
4. **Clean** — `motion_clean` pins planted feet (the skate), takes the slow wander out
   of the floor (the float) and the frame-to-frame shimmer out of the rotations, and
   scores each take on what is left; the best take is kept.
5. **Retarget** — `pack_smplx_clip` packs the motion into the clip format the Animate tab
   plays, and `kimodo_retarget` plays it on the uploaded rig — the same command the
   preset path runs, imported from `viewer/autorig_api`, so the routes cannot drift.

Artifacts land beside the GLB (`<stem>.clip.mp4`, `<stem>.motion.npz`) and ride the
supervisor home, so a run keeps its evidence: the clip the motion came from and the
packed motion itself, replayable on any other rig.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "viewer", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

VENDOR = REPO / "vendor" / "prompt2rig"
H3_DIR = VENDOR / "h3"
BODY_MODELS = VENDOR / "body_models"
WEIGHTS = VENDOR / "weights"
# GVHMR reads these at import; a lazy import inside the stage keeps them honest.
os.environ.setdefault("GVHMR_CHECKPOINTS", str(VENDOR / "gvhmr-checkpoints"))
os.environ.setdefault("GVHMR_BODY_MODELS", str(BODY_MODELS))
# Colab's kernel pre-sets the notebook-inline matplotlib backend, which this venv's
# matplotlib refuses at import (found on the first live run) and GVHMR pulls it in
# through torchmetrics. Overridden, not defaulted: the bad value is already set.
os.environ["MPLBACKEND"] = "Agg"

FPS_H3 = 24
CANVAS = (960, 544)  # height, width — multiples of 32 (a live run refused 944: 944/32 is
                     # 29.5); 960×544 keeps the same fast near-16:9 canvas


def log(message: str) -> None:
    print(f"[prompt2rig {time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(command: list[str], *, what: str) -> None:
    """A subprocess on the VM: its output is captured and reprinted, because the exec
    channel only relays kernel prints — silent output is how a hang looks from here."""
    log(f"{what}: $ {' '.join(str(c) for c in command)}")
    done = subprocess.run([str(c) for c in command], capture_output=True, text=True,
                          check=False)
    for stream in (done.stdout, done.stderr):
        for line in (stream or "").splitlines():
            print(f"  {line}", flush=True)
    if done.returncode != 0:
        raise SystemExit(f"{what} failed (exit {done.returncode})")


def snapped_frames(seconds: float) -> int:
    """H3 decodes 17·n+5 frames; the request is rounded up into that grid."""
    target = max(5, min(15, seconds)) * FPS_H3
    n = -(-max(target - 5, 0) // 17)  # ceil
    return int(17 * n + 5)


def motion_prompt(user_prompt: str) -> str:
    """The description H3 acts out: the user's motion, framed so exactly one full body
    performs it planted on the spot — the shape of clip the motion fixer and the
    retargeter play best. "Stays in frame" alone let H3's actors wander mid-action
    (found live: a hammering clip whose root crept 30 cm and bobbed 23 cm)."""
    return (f"One person {user_prompt.rstrip('.')}, standing in one spot with both feet "
            "planted on the floor, not walking or stepping around. Full body visible and "
            "centered, the whole person stays in frame for the entire clip, static camera, "
            "medium-wide shot, flat lighting, plain studio background, realistic human "
            "proportions.")


def ltx_motion_prompt(user_prompt: str) -> str:
    """The description LTX-2.5 acts out: the same planted-in-place framing the H3 prompt
    carries, but as one long paragraph — its card says it was trained on long
    single-paragraph audio-visual captions and degrades on short ones, so the light and
    the (quiet) room are named too."""
    return (f"A single continuous shot of one person {user_prompt.rstrip('.')}, standing "
            "in one spot with both feet planted on the floor, not walking or stepping "
            "around, the whole person in frame and centered for the entire clip. The "
            "camera is locked on a tripod, medium-wide, eye level; a plain studio "
            "backdrop in flat, even lighting; no cuts, no camera movement. Realistic "
            "human proportions, natural weight shifts as the person moves; the only "
            "sound is the quiet room.")


def render_keyframe(rigged: Path, out: Path) -> bool:
    """The clip's first frame: our own rig, full body, front — the keyframe that keeps
    H3's actor our character and its framing still for the motion recovery.

    Rendered through the SkinTokens venv (the one bpy on this VM), best-effort: without
    it the clip still generates, just as a text-only request whose actor is whoever H3
    casts — the motion is the same and only the motion survives the chain.
    """
    if out.is_file():
        return True
    blender = REPO / "vendor" / "SkinTokens" / ".venv" / "bin" / "python"
    if not blender.is_file():
        log("keyframe: no SkinTokens venv here — a text-only clip it is")
        return False
    try:
        run([blender, REPO / "scripts" / "hero_blender.py", "--", "still", rigged, out],
            what="keyframe")
        return out.is_file()
    except SystemExit as exc:
        log(f"keyframe: render failed ({exc}) — a text-only clip it is")
        return False


def generate_clip(prompt: str, seconds: float, seed: int, out: Path,
                  keyframe: Path | None = None, engine: str = "ltx2") -> None:
    """Stage 1: the acting clip. MiniMax H3 (int8) by default; LTX-2.5 distilled with
    `engine="ltx2"` — the anti-flicker pick, and flicker is exactly what motion
    recovery turns into shimmer. Both land at 24 fps, so nothing downstream knows
    which engine acted the clip out."""
    if out.is_file():
        log(f"video: have {out.name}")
        return
    if engine == "ltx2":
        generate_ltx2(prompt, seconds, seed, out, keyframe)
    else:
        generate_h3(prompt, seconds, seed, out, keyframe)


def ensure_ltx2() -> None:
    """The LTX-2.5 engine installs on demand: its own venv and ~71 GB of weights that a
    VM only pays for when a run actually asks for the engine. The weights come from
    ModelScope's ungated mirror, so no token or account is needed. The marker carries
    the bootstrap's build id, so a VM warm on an older build re-runs the install and
    picks up what changed (a venv package, say) without re-downloading the weights."""
    import bootstrap_prompt2rig_colab as bootstrap

    marker = REPO / "vendor" / "prompt2rig-driver" / ".ltx2-done"
    if marker.is_file() and marker.read_text() == bootstrap.LTX2_BUILD:
        return
    log("video: installing LTX-2.5 on this VM first (one time, about 71 GB, from "
        "ModelScope's ungated mirror of the weights)")
    run([sys.executable, REPO / "scripts" / "bootstrap_prompt2rig_colab.py",
         "--yes", "--ltx2-only"], what="ltx2 install")


def generate_ltx2(prompt: str, seconds: float, seed: int, out: Path,
                  keyframe: Path | None) -> None:
    """Stage 1, the LTX-2.5 way: the distilled checkpoint in its own venv, driven by
    `scripts/ltx2_video.py` (which lives over the details — the sigma schedule, the
    8k+1 frame grid, the silent soundtrack we drop)."""
    import ltx2_video

    ensure_ltx2()
    frames = ltx2_video.snapped_frames(seconds)
    argv = [str(REPO / "vendor" / "ltx2" / ".venv" / "bin" / "python"),
            str(REPO / "scripts" / "ltx2_video.py"), str(out),
            "--prompt", ltx_motion_prompt(prompt),
            "--seconds", f"{seconds:g}", "--seed", str(seed)]
    if keyframe is not None and keyframe.is_file():
        argv += ["--keyframe", str(keyframe)]
    log(f"video: LTX-2.5 acting out {frames} frames "
        f"({frames / ltx2_video.FPS:.1f}s, distilled"
        f"{', from the rig keyframe' if keyframe else ''})")
    run(argv, what="video")


def generate_h3(prompt: str, seconds: float, seed: int, out: Path,
                keyframe: Path | None) -> None:
    """Stage 1, the H3 way: MiniMax H3, int8, from the local install the bootstrap
    quantized."""
    # fl2va without a keyframe is a chain that noises an empty list of conditions
    # (found on the first live run): the keyframe is what selects the workflow, and the
    # pipeline wants it opened — a path string dies in its resize step (`keyframe.size`).
    workflow = "fl2va" if keyframe is not None and keyframe.is_file() else "t2va"
    # Opened, and forced to RGB: the encoder normalizes three channels and a keyframe
    # with an alpha channel dies in its pixel-mean step (found on a live run).
    condition = (f", image=Image.open({str(keyframe)!r}).convert('RGB')"
                 if workflow == "fl2va" else "")
    script = f"""
import torch
from diffusers import ModularPipeline
from diffusers.utils.export_utils import encode_video
from PIL import Image

pipe = ModularPipeline.from_pretrained({str(H3_DIR)!r}, workflow={workflow!r})
pipe.load_components(dtype=torch.bfloat16)
for name in ("transformer", "text_encoder", "vae", "audio_vae"):
    getattr(pipe, name).to("cuda").requires_grad_(False)
results = pipe(prompt={motion_prompt(prompt)!r}, num_frames={snapped_frames(seconds)},
               height={CANVAS[0]}, width={CANVAS[1]}{condition},
               generator=torch.Generator().manual_seed({seed}), output=["videos"])
encode_video(results["videos"][0], fps={FPS_H3}, output_path={str(out)!r})
print("clip written")
"""
    log(f"video: H3 acting out {snapped_frames(seconds)} frames "
        f"({snapped_frames(seconds) / FPS_H3:.1f}s, {workflow})")
    run([sys.executable, "-c", script], what="video")


def reencode_for_gvhmr(source: Path, work: Path) -> Path:
    """GVHMR wants 30 fps and no more than 854 px on the long side; a fast re-encode
    halves the decode cost at no accuracy cost (it only sees 256 px person crops)."""
    target = work / "gvhmr_input.mp4"
    if target.is_file():
        return target
    fit = ("scale='if(gt(iw,ih),min(854,iw),-2)':'if(gt(iw,ih),-2,min(854,ih))'")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-vf", f"fps=30,{fit}",
         "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", str(target)],
        what="re-encode")
    return target


def recover_motion(clip: Path, work: Path) -> dict:
    """Stage 2: GVHMR. Returns the arrays `pack_smplx_clip` wants, plus the camera-frame
    pelvis rotations and the 2D pose cache the hands stage needs. The pipeline's own
    preprocess caches make a rerun cheap, so the result object is used directly rather
    than its saved `.pt`, whose layout is not a promise."""
    import numpy as np

    out_dir = work / "gvhmr"
    out_dir.mkdir(parents=True, exist_ok=True)
    import gvhmr

    log("motion: GVHMR recovering SMPL-X from the clip")
    pipe = gvhmr.pipeline("human-motion-recovery", device="cuda")
    result = pipe(str(clip), static_camera=True, output_dir=str(out_dir),
                  output_root=str(work), progress=False)

    def array(params, key):
        value = getattr(params, key) if hasattr(params, key) else params[key]
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    world, camera = result.smpl_params_world, result.smpl_params_camera
    motion = {key: array(world, key) for key in ("global_orient", "body_pose", "transl")}
    vitpose = out_dir / "preprocess" / "vitpose.pt"
    pose_px = None
    if vitpose.is_file():
        import torch

        pose_px = np.asarray(torch.load(vitpose, weights_only=False))
    return {"motion": motion, "pose_px": pose_px,
            "orient_incam": array(camera, "global_orient")}


def track_hands(source: Path, work: Path, motion: dict, frames: int,
                fps: float) -> dict | None:
    """Stage 3: fingers. WiLoR's weights when installed, MediaPipe otherwise, none at
    all when neither can run — the body motion never depends on the hands."""
    if motion.get("pose_px") is None:
        log("hands: no 2D pose cache to assign wrists by — riding the wrists")
        return None
    try:
        import prompt2rig_hands as hands
        backend = hands.backend(WEIGHTS / "wilor", WEIGHTS / "hand_landmarker.task")
        log(f"hands: {backend.name} over the clip")
        frames_list, src_fps = hands.read_frames(source)
        hw = frames_list[0].shape[:2]
        out = hands.finish(backend(frames_list, src_fps), src_fps, hw,
                           motion["pose_px"], max(hw), motion["motion"]["global_orient"],
                           motion["orient_incam"], frames, fps)
        log("hands: " + "  ".join(f"{s} {out[s]['valid'].mean():.0%}" for s in ("R", "L")))
        return {s: {"kp": out[s]["kp"], "valid": out[s]["valid"]} for s in ("R", "L")}
    except Exception as exc:
        log(f"hands: failed ({type(exc).__name__}: {exc}) — riding the wrists")
        return None


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rigged", type=Path, help="a SkinTokens-rigged GLB")
    parser.add_argument("result", type=Path, help="the animated GLB to write")
    parser.add_argument("--prompt", required=True, help="the motion to act out")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tries", type=int, default=2,
                        help="how many takes to act out; the best-scoring one is kept")
    parser.add_argument("--video-engine", choices=("h3", "ltx2"), default="ltx2",
                        help="Who acts the prompt out: LTX-2.5 distilled (default — "
                             "won the A/B: a quarter of H3's shimmer, hands tracked "
                             "97%/100% vs 82%/91%) or MiniMax H3 (int8)")
    parser.add_argument("--arm-spread", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args(argv)


def take(args: argparse.Namespace, work: Path, seed: int, keyframe: Path | None) -> dict:
    """One take: video → body motion → hands → pack → clean, all under `work/`.

    Each stage skips what already exists, so an interrupted take resumes instead of
    paying for its video again. Returns the artifacts plus the cleaner's report.
    """
    clip = work / "clip.mp4"
    generate_clip(args.prompt, args.seconds, seed, clip, keyframe, args.video_engine)
    gvhmr_clip = reencode_for_gvhmr(clip, work)
    motion = recover_motion(gvhmr_clip, work)
    frames = len(motion["motion"]["transl"])
    hands = track_hands(clip, work, motion, frames, 30.0)

    import numpy as np
    import pack_smplx_clip

    log(f"pack: {frames} frames into the retargeter's clip format")
    arrays = pack_smplx_clip.pack(motion["motion"]["global_orient"],
                                  motion["motion"]["body_pose"],
                                  motion["motion"]["transl"],
                                  BODY_MODELS / "smplx" / "SMPLX_NEUTRAL.npz",
                                  hands=hands, fps=30.0)

    import motion_clean

    arrays, report = motion_clean.clean(arrays["quats"], arrays["joints0"],
                                        arrays["hips"], 30.0)
    log("clean: " + "  ".join(f"{key}={value}" for key, value in report.items()))
    packed = work / "motion.npz"
    np.savez_compressed(packed, **arrays)
    return {"clip": clip, "packed": packed, "report": report, "hands": hands}


def score_of(one: dict) -> float:
    """Which take plays best: hands tracked, planted in place, feet still, no shimmer.

    Numbers the cleaner's report already carries, weighted to match what reads as bad on
    the rig: hands lead (nothing else in the chain fixes them), planted-in-place next
    (a wandering take cannot be pinned), then slide, then shimmer.
    """
    import numpy as np

    hands = one["hands"] or {}
    valid = (float(np.mean([np.asarray(side["valid"]).mean() for side in hands.values()]))
             if hands else 0.0)
    report = one["report"]
    return (100.0 * valid + (100.0 if report["pinned"] else 0.0)
            - report["slide_after"] - 0.1 * report["shimmer_after"])


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)

    work = args.result.parent / "prompt2rig"
    work.mkdir(parents=True, exist_ok=True)
    clip, packed = args.result.with_suffix(".clip.mp4"), args.result.with_suffix(".motion.npz")

    keyframe = work / "keyframe.png"
    if not render_keyframe(args.rigged, keyframe):
        keyframe = None
    takes = []
    for i in range(max(1, args.tries)):
        take_work = work / f"try{i}"
        take_work.mkdir(parents=True, exist_ok=True)
        log(f"take {i + 1} of {max(1, args.tries)} (seed {args.seed + i})")
        takes.append(take(args, take_work, args.seed + i, keyframe))
        log(f"take {i + 1} scores {score_of(takes[-1]):.1f}")
    best = max(takes, key=score_of)
    shutil.copy(best["clip"], clip)
    shutil.copy(best["packed"], packed)
    log(f"keeping take {takes.index(best) + 1} (score {score_of(best):.1f})")

    # Stage 4: the preset path's own command, so both routes stay one command apart.
    import autorig_api

    run(autorig_api.animate_command(args.rigged, packed, args.result,
                                    args.arm_spread, args.speed), what="retarget")
    if not args.result.is_file():
        print(f"retarget wrote no {args.result}", file=sys.stderr)
        return 4
    log(f"done: {args.result.name} (+ {clip.name}, {packed.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
