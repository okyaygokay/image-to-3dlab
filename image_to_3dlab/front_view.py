"""Fitted cameras for a mesh, so Pixel Match can paint it from pictures of it.

A Pixal3D body carries the cameras its maker used (a ``.svviews`` directory); a body
from any other engine (Hi3D, TRELLIS, ...) carries none, so Finish's photo stage had
nothing to paint through and the body kept the engine's own texture. But the picture
that made such a model *is* a front view of it — one camera is all Pixel Match needs
to paint the front with the real photo's pixels. The angles nobody photographed (back,
sides) get cameras too, and their pictures are then made to match the mesh: the mesh is
rendered through the very same cameras and the render is repainted by Qwen (edit mode)
with the front picture as the colour reference — so the picture always agrees with the
geometry underneath (a freely *drawn* back disagrees: Qwen invents an apron of crossed
straps where the mesh has a full one, 2026-10-28).

Two fits make each camera. The mesh gives the scale: its extents along the camera's
axes, against that picture's *figure box* (`figure_box` below — a cut-out's alpha, else
everything far from the picture's own border colour), so the model lands where the
figure actually sits in the picture rather than neatly centred in a frame it may fill
only partly; a synthetic view's box is chosen (the model fills the canvas with a margin)
because its render reproduces the camera exactly. The frame's convention gives the
axes: GLB +X to the front picture's right and +Y up, the other angles turns around the
vertical, all expressed in the view frame ``photo_paint`` reads cameras in
(``PIXAL3D_FRAME``). Each camera sits far enough back that the perspective stays mild,
and every transforms.json says it was fitted, so a record never mistakes one for a
maker's camera.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from image_to_3dlab.photo_paint import to_view_space

#: How far each fitted camera sits from the model, in units of the model's largest span:
#: a maker's real camera distance is not known, and a near fit would bake in a
#: perspective the picture cannot have.
DISTANCE = 4.0

#: A figure box smaller than this share of the frame says the finder failed (no cut-out,
#: no plain background); the whole frame is then the figure, centred as before.
MIN_FIGURE_SHARE = 0.05

#: How far a pixel must sit from the border colour to be figure rather than background.
BACKGROUND_TOLERANCE = 40


@dataclass(frozen=True)
class Angle:
    """One viewing direction: the camera's right/up/back axes in GLB space.

    `back` points from the model at the camera; the camera looks the other way, like
    every camera `photo_paint.project` reads (depth positive in front).
    """
    name: str
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    back: tuple[float, float, float]


#: The front picture's view (GLB +X right, +Y up, seen from +Z) and the drawn back's
#: (half a turn around the vertical, mirrored like any real photo taken from behind).
FRONT = Angle("front", (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
BACK = Angle("back", (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
#: The two sides: the subject's own left (GLB +X, which the front view shows on the
#: picture's right) and right, each camera a quarter turn from the front.
LEFT = Angle("left", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
RIGHT = Angle("right", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
ANGLES = {a.name: a for a in (FRONT, BACK, LEFT, RIGHT)}

#: How much of a synthetic view's canvas the model leaves free, so nothing that matters
#: sits on the frame's edge and the silhouette fade has room to fall.
RENDER_MARGIN_SHARE = 0.05


def figure_box(picture: Path) -> tuple[int, int, int, int]:
    """The figure's pixel box in `picture`, or the whole frame when none is findable.

    A cut-out's alpha is exact (`hero.picture_box`). Otherwise the background is
    whatever the picture's own borders are — the studio's light grey, a generated
    near-black, a photo's white — and the figure is everything meaningfully far from
    it. A guess that covers nearly the whole frame is no guess: back to the full frame.
    """
    from image_to_3dlab import hero

    image = Image.open(picture)
    width, height = image.size
    if np.asarray(image.convert("RGBA").getchannel("A")).min() < 128:
        return hero.picture_box(image)               # a real cut-out: its box is exact
    rgb = np.asarray(image.convert("RGB")).astype(int)
    ring = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = np.median(ring, axis=0)
    far = np.abs(rgb - background).max(-1) > BACKGROUND_TOLERANCE
    ys, xs = np.nonzero(far)
    wide = len(xs) and (xs.max() - xs.min()) > width * MIN_FIGURE_SHARE
    tall = len(ys) and (ys.max() - ys.min()) > height * MIN_FIGURE_SHARE
    if wide and tall and far.mean() < 0.95:
        return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return 0, width - 1, 0, height - 1


def _camera(mid: np.ndarray, half: np.ndarray, width: int, height: int,
            box: tuple[int, int, int, int], angle: Angle, generous: bool = False) -> dict:
    """One fitted camera as a transforms.json frame for `width x height` pixels framing
    `box` (the figure's pixel box), seen from `angle`.

    The mesh's extents along the camera's right/up axes give the scale against the box;
    measured to the mesh's *nearest* face along the view, which is what would clip —
    farther surfaces then land a little inside the box instead of outside it. A picture
    compromises between the axes (geometric mean, so an aspect mismatch splits between
    them: the mesh must land *on* the box). A synthetic view frames generously instead
    (`generous`): the limiting axis fills its box and the other leaves margin — a
    compromise would zoom the limiting axis out of frame when the two disagree wildly
    (a T-pose seen side-on, 2026-10-28).
    """
    right = np.array(angle.right)
    up = np.array(angle.up)
    back = np.array(angle.back)
    x0, x1, y0, y1 = box
    depth = DISTANCE * max(float(half.max()), 1e-6)
    near = max(depth - float(np.abs(back) @ half), 1e-6)
    span_x = max((x1 - x0) / 2 * near / max(float(np.abs(right) @ half), 1e-6), 1e-9)
    span_y = max((y1 - y0) / 2 * near / max(float(np.abs(up) @ half), 1e-6), 1e-9)
    focal = min(span_x, span_y) if generous else math.sqrt(span_x * span_y)
    shift = np.array([-(x0 + x1) / 2 + width / 2, (y0 + y1) / 2 - height / 2]) * near / focal
    eye = mid + back * depth + right * shift[0] + up * shift[1]

    cam = np.eye(4)
    cam[:3, :3] = np.stack([to_view_space(np.array(a)[None, :])[0]
                            for a in (right, up, back)], axis=1)
    cam[:3, 3] = to_view_space(eye[None, :])[0]
    return {"file_path": "",
            "camera_angle_x": 2 * math.atan(width / 2 / focal),
            "transform_matrix": cam.tolist(),
            "name": f"{angle.name} (fitted)",
            "figure_box": [x0, x1, y0, y1]}


def fit_views(model: Path, pictures: list[tuple[Path, str]], out_dir: Path,
              renders: list[tuple[str, int, int]] = ()) -> Path:
    """Write `out_dir` as a transforms.json holding one fitted camera per picture.

    Each entry is `(picture, angle)` with an angle from `ANGLES` ("front", "back").
    `renders` are angles with no picture yet: `(angle, width, height)` — a flat
    placeholder of that size is written (the render step replaces its pixels and the
    repaint recolours them), framed so the model fills the canvas with
    `RENDER_MARGIN_SHARE` to spare: for a view nobody photographed, the framing is ours
    to choose, and the render reproduces this exact camera. Returns `out_dir`, holding
    transforms.json and a copy of every picture — the same shape a ``.svviews``
    directory has, ready for `--views`.
    """
    import trimesh

    lo, hi = np.asarray(trimesh.load(model, force="scene", process=False).bounds,
                        dtype=np.float64)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for picture, angle in pictures:
        shutil.copy(picture, out_dir / picture.name)
        image = Image.open(picture)
        width, height = image.size
        frames.append(_camera(mid, half, width, height, figure_box(picture), ANGLES[angle])
                      | {"file_path": picture.name})
    for angle, width, height in renders:
        name = render_name(angle)
        mx, my = round(width * RENDER_MARGIN_SHARE), round(height * RENDER_MARGIN_SHARE)
        Image.new("RGB", (width, height), (128, 128, 128)).save(out_dir / name)
        frames.append(_camera(mid, half, width, height, (mx, width - 1 - mx, my, height - 1 - my),
                              ANGLES[angle], generous=True)
                      | {"file_path": name, "render": True})
    (out_dir / "transforms.json").write_text(json.dumps({
        "mesh_scale": 1.0,
        "fitted": "cameras fitted to the mesh and each picture's figure box; "
                  "its maker saved none",
        "frames": frames,
    }, indent=2), encoding="utf-8")
    return out_dir


def render_name(angle: str) -> str:
    """The placeholder/render image name for a synthetic angle."""
    return f"{angle}.png"


def fit_front_view(model: Path, picture: Path, out_dir: Path) -> Path:
    """The one-picture case: `picture` seen from the front (see `fit_views`)."""
    return fit_views(model, [(picture, "front")], out_dir)
