#!/usr/bin/env python3
"""Draw one acting clip with LTX-2.5 (distilled), as the video stage of the
prompt-to-motion chain.

    vendor/ltx2/.venv/bin/python scripts/ltx2_video.py clip.mp4 \
        --prompt "..." --seconds 6.5 --seed 1 [--keyframe pose.png]

Runs on the Colab VM, inside the venv `bootstrap_prompt2rig_colab.py --ltx2-only`
builds (diffusers main, pinned by commit, because LTX-2.5 is not in a release).
The distilled checkpoint is driven the way its model card says — an explicit sigma
schedule, not a step count, all guidance off — and the clip lands at 24 fps in the
same 960x544 canvas the H3 stage uses, so every later stage (GVHMR, WiLoR, the
packer) is engine-agnostic.

LTX-2.5 is the anti-flicker pick: its whole pitch is temporal stability in
high-motion shots, and flicker is what motion recovery turns into shimmer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: LTX's video VAE compresses time 8x, so a clip must be 8k+1 frames long.
FRAME_GRID = 8
FPS = 24

#: The distilled model card's recipe: one unguided forward per sigma, all guidance
#: scales at 1 or 0. Anything else quietly costs quality.
SIGMAS = "DISTILLED_SIGMA_VALUES"


def snapped_frames(seconds: float) -> int:
    """The nearest grid frame count at least `seconds` long."""
    want = max(2.0, min(15.0, seconds)) * FPS
    return int((int((want - 1) // FRAME_GRID) + 1) * FRAME_GRID + 1)


def generate(prompt: str, out: Path, seconds: float, seed: int,
             keyframe: Path | None, width: int = 960, height: int = 544) -> None:
    """One clip, video only — audio is generated jointly but motion recovery has no
    use for it, so it is dropped rather than encoded."""
    import torch
    from diffusers import LTX2Pipeline
    from diffusers.pipelines.ltx2.utils import (
        DEFAULT_NEGATIVE_PROMPT,
        DISTILLED_SIGMA_VALUES,
    )
    from diffusers.utils import encode_video
    from PIL import Image

    model_dir = Path(__file__).resolve().parents[1] / "vendor" / "ltx2" / "model"
    frames = snapped_frames(seconds)
    print(f"ltx2: {frames} frames ({frames / FPS:.1f}s) at {width}x{height}, seed {seed}",
          flush=True)
    pipe = LTX2Pipeline.from_pretrained(str(model_dir), dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    call = {
        "prompt": prompt, "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "width": width, "height": height, "num_frames": frames, "frame_rate": float(FPS),
        "sigmas": DISTILLED_SIGMA_VALUES,
        "guidance_scale": 1.0, "audio_guidance_scale": 1.0, "stg_scale": 0.0,
        "audio_stg_scale": 0.0, "modality_scale": 1.0, "audio_modality_scale": 1.0,
        "generator": torch.Generator("cuda").manual_seed(seed),
        "output_type": "np", "return_dict": False,
    }
    if keyframe is not None:
        # RGB, for the same reason the H3 stage forces it: a keyframe with an alpha
        # channel dies in the encoder's normalisation.
        call["image"] = Image.open(str(keyframe)).convert("RGB")
    video, _audio = pipe(**call)
    encode_video(video[0], fps=FPS, output_path=str(out))
    print("clip written", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="the clip to write (mp4)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seconds", type=float, default=6.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keyframe", type=Path,
                        help="a first-frame image; without one the clip is text-to-video")
    args = parser.parse_args(argv)
    generate(args.prompt, args.out, args.seconds, args.seed,
             args.keyframe if args.keyframe and args.keyframe.is_file() else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
