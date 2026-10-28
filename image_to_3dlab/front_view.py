"""A front camera fitted to a mesh and its picture, so Pixel Match can paint the body.

A Pixal3D body carries the cameras its maker used (a ``.svviews`` directory); a body
from any other engine (Hi3D, TRELLIS, ...) carries none, so Finish's photo stage had
nothing to paint through and the body kept the engine's own texture. But the picture
that made such a model *is* a front view of it, and one camera is all Pixel Match's
front pass needs: the picture's real pixels land on everything the camera can see,
facing it and away from the silhouette edge — exactly the "on (front)" a one-view
Pixal3D run gets.

Two fits make that camera. The mesh gives the scale: its width and height, against the
picture's *figure box* (``figure_box`` below — a cut-out's alpha, else everything far
from the picture's own border colour), so the model lands where the figure actually
sits in the picture rather than neatly centred in a frame it may fill only partly. The
frame's convention gives the axes: GLB +X to the picture's right and +Y up (the same
front view ``hero.project`` assumes for the head crop), expressed in the view frame
``photo_paint`` reads cameras in (``PIXAL3D_FRAME``). The camera sits far enough back
that the perspective stays mild, and the transforms.json says it was fitted, so a
record never mistakes it for a maker's camera.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from image_to_3dlab.photo_paint import to_view_space

#: How far the fitted camera sits from the model, in units of the model's largest span:
#: the maker's real camera distance is not known, and a near fit would bake in a
#: perspective the picture cannot have.
DISTANCE = 4.0

#: A figure box smaller than this share of the frame says the finder failed (no cut-out,
#: no plain background); the whole frame is then the figure, centred as before.
MIN_FIGURE_SHARE = 0.05

#: How far a pixel must sit from the border colour to be figure rather than background.
BACKGROUND_TOLERANCE = 40

#: Columns of the camera's axes in the view frame (PIXAL3D_FRAME): right = GLB +X,
#: up = GLB +Y, looking along GLB -Z from the GLB +Z side — the front view hero.project
#: assumes, rotated into the frame photo_paint reads.
_FRONT_ROTATION = np.array([[-1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0]])


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


def fit_front_view(model: Path, picture: Path, out_dir: Path) -> Path:
    """Write `out_dir` as a one-view transforms.json: `picture` seen from the front.

    The camera is fitted to `model`'s bounds and to the picture's figure box, so every
    part of the mesh projects onto the pixels it was made from. Returns `out_dir`,
    holding transforms.json and a copy of the picture — the same shape a ``.svviews``
    directory has, ready for `--views`.
    """
    import trimesh

    scene = trimesh.load(model, force="scene", process=False)
    lo, hi = np.asarray(scene.bounds, dtype=np.float64)
    mid = (lo + hi) / 2
    half = (hi - lo) / 2
    x0, x1, y0, y1 = figure_box(picture)
    image = Image.open(picture)
    width, height = image.size

    # Scale: the mesh's width and height against the figure box's, geometric mean, so an
    # aspect mismatch (a pose drawn fuller than the mesh) splits between the axes. The
    # fit is measured to the mesh's *nearest* face: that is what would clip, and farther
    # surfaces then land a little inside the box instead of outside it.
    depth = DISTANCE * max(float(half.max()), 1e-6)
    near = max(depth - float(half[2]), 1e-6)
    focal_x = (x1 - x0) / 2 * near / max(float(half[0]), 1e-6)
    focal_y = (y1 - y0) / 2 * near / max(float(half[1]), 1e-6)
    focal = math.sqrt(focal_x * focal_y)
    fov_x = 2 * math.atan(width / 2 / focal)

    # Centre: the mesh's middle lands on the figure box's middle (project() centres
    # pixels at width/2, height/2; picture y runs down).
    eye = np.array([mid[0] - ((x0 + x1) / 2 - width / 2) * near / focal,
                    mid[1] + ((y0 + y1) / 2 - height / 2) * near / focal,
                    mid[2] + depth])
    position = to_view_space(eye[None, :])[0]
    cam = np.eye(4)
    cam[:3, :3] = _FRONT_ROTATION
    cam[:3, 3] = position

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(picture, out_dir / picture.name)
    (out_dir / "transforms.json").write_text(json.dumps({
        "camera_angle_x": fov_x,
        "mesh_scale": 1.0,
        "fitted": "front camera fitted to the mesh and the picture's figure box; "
                  "its maker saved none",
        "figure_box": [x0, x1, y0, y1],
        "frames": [{"file_path": picture.name,
                    "transform_matrix": cam.tolist(),
                    "name": "front (fitted)"}],
    }, indent=2), encoding="utf-8")
    return out_dir
