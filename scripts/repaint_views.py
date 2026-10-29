#!/usr/bin/env python3
"""Recolour one fitted render with Qwen in edit mode, from its reference card.

    python scripts/repaint_views.py card.png view.png --render 768x1024 [--seed 3]

The card (`image_to_3dlab.repaint_view.reference_card`) holds the render framed exactly
as its fitted camera sees it, with the front's cut-out as a small colour reference
above; its alpha channel carries the render's own matte in the lower region. The edit
keeps composition — the same property the back-view draw is held to — so the lower
region comes back with pose and framing untouched: cropped out and matted, it is a
picture of this mesh from this camera, in the character's colours.

The render's size is passed in (`--render WxH`) because the card alone does not say
where the strip ends. Runs wherever Qwen lives: locally with the repo's weights, or as
the `qwen_image` backend's `--exec` worker on a rented VM — the same script, the same
pinned weights. A `<view>.provenance.json` records the run like any picture.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "viewer")]

import image_api

from image_to_3dlab import repaint_view

DEFAULT_SEED = 3
VISION_GB = 0.75


def missing_vision_message() -> str:
    directory, filename = image_api.VISION_FILE
    repo = directory.removeprefix("models--").replace("--", "/")
    return (f"Repainting a view needs Qwen's vision projector, {filename} "
            f"({VISION_GB:.2f} GB),\nfrom {repo}. Fetch it with:\n"
            f"    python scripts/bootstrap_qwen_image.py --vision")


def repaint_one(card_path: Path, view_path: Path, render_size: tuple[int, int],
                seed: int = DEFAULT_SEED) -> int:
    vision = image_api.resolve_vision()
    if vision is None:
        print(missing_vision_message())
        return 1
    weights = image_api.resolve_weights()

    angle = view_path.stem
    card = Image.open(card_path).convert("RGBA")
    width, height = card.size
    flat = Image.new("RGB", (width, height), repaint_view.CARD_GREY)
    flat.paste(card, (0, 0), card)                        # what Qwen sees: no alpha holes
    flat_path = card_path.with_name(f"{card_path.stem}__flat.png")
    flat.save(flat_path)

    prompt = repaint_view.REPAINT_PROMPT.format(view=repaint_view.VIEW_WORDS[angle])
    settings = image_api.clean_settings({"seed": seed, "width": width, "height": height})
    command = image_api.build_command(prompt, settings, view_path, weights,
                                      reference=flat_path, vision=vision)
    started = time.time()
    result = subprocess.run(command, capture_output=True, text=True, cwd=str(REPO),
                            check=False)
    if result.returncode or not view_path.exists():
        print((result.stdout + result.stderr)[-1500:])
        return 1
    seconds = time.time() - started

    # The edit wrote the whole card; the view is its lower region, matted with the
    # render's own alpha (kept in the card's alpha, eroded a little for drift).
    painted = Image.open(view_path).convert("RGB")
    if painted.size != card.size:                    # a size the edit rounds a hair
        painted = painted.resize(card.size)
    alpha = card.getchannel("A").crop((0, height - render_size[1], width, height))
    repaint_view.render_from_card(painted, render_size, alpha).save(view_path)
    flat_path.unlink(missing_ok=True)

    record = image_api.provenance(prompt, settings, seconds, view_path)
    record["reference"] = {"card": str(card_path), "render_size": list(render_size),
                           "mode": f"edit (repainted {angle} render)"}
    view_path.with_suffix(".provenance.json").write_text(json.dumps(record, indent=2))
    print(f"[repaint] {seconds:.0f}s -> {view_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("card", type=Path)
    parser.add_argument("view", type=Path)
    parser.add_argument("--render", required=True, metavar="WxH",
                        help="the render's size, which says where the card's strip ends")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if not args.card.is_file():
        raise SystemExit(f"not found: {args.card}")
    width, _, height = args.render.partition("x")
    try:
        size = (int(width), int(height))
    except ValueError:
        raise SystemExit(f"--render wants WxH, got {args.render}")
    return repaint_one(args.card, args.view, size, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
