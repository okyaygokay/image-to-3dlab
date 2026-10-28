"""A front camera fitted to a mesh and its picture, so Pixel Match can paint a foreign body.

A body from any engine but Pixal3D arrives with no cameras, and Pixel Match without a
camera keeps the engine's own texture. The picture that made the model is a front view
of it, so one camera fitted to the mesh's bounds and the picture's figure box is enough
— the same "on (front)" a one-view Pixal3D run gets. These tests pin the fit: the model
must land where the figure actually sits in the picture (not neatly centred in a frame
it may fill only partly), the right way up, and the file must say its camera was fitted,
never mistaken for a maker's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import front_view
from image_to_3dlab import photo_paint as pp


def _box_model(tmp_path: Path, extents=(2.0, 3.0, 1.0)) -> Path:
    import trimesh

    model = tmp_path / "body.glb"
    trimesh.creation.box(extents=extents).export(model)
    return model


def _figure_picture(tmp_path: Path, size=(400, 600), figure=(100, 300, 200, 500),
                    background=(25, 25, 25)) -> Path:
    """A plain background with the figure as a solid patch at `figure` = (x0, x1, y0, y1)."""
    picture = Image.new("RGB", size, background)
    x0, x1, y0, y1 = figure
    picture.paste(Image.new("RGB", (x1 - x0, y1 - y0), (214, 178, 148)), (x0, y0))
    path = tmp_path / "picture.png"
    picture.save(path)
    return path


def _projected(model: Path, fitted: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import trimesh

    lo, hi = np.asarray(trimesh.load(model, force="scene", process=False).bounds)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])], dtype=np.float64)
    view = pp.load_views(fitted)[0][0]
    return pp.project(pp.to_view_space(corners), view)


def test_the_fitted_camera_lands_the_model_on_the_picture_figure(tmp_path):
    """The point of the fit: the mesh's pixels land where the figure sits in the picture
    (a box off-centre, in a picture the model fills only partly), the right way up."""
    model = _box_model(tmp_path)                       # 2 wide, 3 tall — matches the figure
    figure = (100, 300, 200, 500)
    picture = _figure_picture(tmp_path, figure=figure)
    fitted = front_view.fit_front_view(model, picture, tmp_path / "views")

    assert front_view.figure_box(picture) == (100, 299, 200, 499)  # the figure found, 1px inclusive
    x, y, depth = _projected(model, fitted)
    x0, x1, y0, y1 = figure
    pad_x, pad_y = (x1 - x0) * 0.2, (y1 - y0) * 0.2    # perspective spread and a mild aspect
    assert (depth > 0).all()                           # split; the centre lands on the centre
    assert x.min() >= x0 - pad_x and x.max() <= x1 + pad_x
    assert y.min() >= y0 - pad_y and y.max() <= y1 + pad_y
    assert abs((x.min() + x.max()) / 2 - (x0 + x1) / 2) < pad_x
    assert abs((y.min() + y.max()) / 2 - (y0 + y1) / 2) < pad_y

    import trimesh
    lo, hi = np.asarray(trimesh.load(model, force="scene", process=False).bounds)
    mid = (lo + hi) / 2
    top = pp.project(pp.to_view_space(np.array([[mid[0], hi[1], mid[2]]])), pp.load_views(fitted)[0][0])
    bottom = pp.project(pp.to_view_space(np.array([[mid[0], lo[1], mid[2]]])), pp.load_views(fitted)[0][0])
    assert top[1][0] < bottom[1][0]                    # up in the world is up in the picture


def test_a_picture_with_no_findable_figure_falls_back_to_the_whole_frame(tmp_path):
    """A busy photo (borders as varied as the middle) hides its figure; the fit then
    centres the model in the frame rather than guessing."""
    picture = tmp_path / "busy.png"
    Image.new("RGB", (500, 500), (90, 60, 30)).save(picture)
    model = _box_model(tmp_path, extents=(3.0, 3.0, 1.0))
    fitted = front_view.fit_front_view(model, picture, tmp_path / "views")

    assert front_view.figure_box(picture) == (0, 499, 0, 499)
    x, y, depth = _projected(model, fitted)
    assert (depth > 0).all() and (x >= -1).all() and (x <= 501).all()
    assert (y >= -1).all() and (y <= 501).all()


def test_the_file_says_its_camera_was_fitted_not_saved_by_a_maker(tmp_path):
    """A record must never mistake this for the maker's camera."""
    fitted = front_view.fit_front_view(_box_model(tmp_path), _figure_picture(tmp_path),
                                       tmp_path / "views")
    meta = json.loads((fitted / "transforms.json").read_text())
    assert "fitted" in meta and meta["mesh_scale"] == 1.0
    assert meta["frames"][0]["figure_box"] == [100, 299, 200, 499]   # what it was fitted to
    assert meta["frames"][0]["file_path"] == "picture.png"
    assert (fitted / "picture.png").is_file()           # the view is self-contained


def test_an_unreadable_mesh_raises_rather_than_writing_a_bogus_camera(tmp_path):
    broken = tmp_path / "body.glb"
    broken.write_bytes(b"not a glb")
    with pytest.raises(ValueError):
        front_view.fit_front_view(broken, _figure_picture(tmp_path), tmp_path / "views")
    assert not (tmp_path / "views" / "transforms.json").is_file()


def _bumped_model(tmp_path: Path) -> Path:
    """A box with a bump on GLB +X — asymmetric, so a camera that walks round the model
    can be told from one that mirrors it."""
    import trimesh

    bump = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    bump.apply_translation([1.2, 0.0, 0.0])            # sticks out on GLB +X
    model = tmp_path / "bumped.glb"
    trimesh.util.concatenate([trimesh.creation.box(extents=(2.0, 3.0, 1.0)), bump]).export(model)
    return model


def test_two_views_share_one_file_and_both_frame_the_model(tmp_path):
    """A front and a back picture fit into one transforms.json — the same shape a
    .svviews directory has — with each camera framing the whole model in its own
    picture and nothing clipped."""
    import trimesh

    model = _box_model(tmp_path)
    front, back = _figure_picture(tmp_path), tmp_path / "back.png"
    Image.new("RGB", (400, 600), (25, 25, 25)).save(back)

    fitted = front_view.fit_views(model, [(front, "front"), (back, "back")], tmp_path / "views")

    views, mesh_scale = pp.load_views(fitted)
    assert mesh_scale == 1.0 and len(views) == 2
    assert [v.name for v in views] == ["front (fitted)", "back (fitted)"]
    assert (fitted / "picture.png").is_file() and (fitted / "back.png").is_file()
    lo, hi = np.asarray(trimesh.load(model, force="scene", process=False).bounds)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])], dtype=np.float64)
    for view in views:
        x, y, depth = pp.project(pp.to_view_space(corners), view)
        # a full-frame box meets the frame's edge exactly; a pixel's half-centre can
        # poke past the last pixel's centre — sub-pixel, against the fit's ~25 px live error
        assert (depth > 0).all() and (x >= -1).all() and (x <= 401).all()
        assert (y >= -1).all() and (y <= 601).all()


def test_the_back_camera_walks_round_the_model_rather_than_mirroring_it(tmp_path):
    """A bump on GLB +X sits picture-right from the front and picture-left from the
    back — the way a real person's features swap sides when you walk round them. A
    mirrored back camera would paint the other side's pixels onto the bump."""
    model = _bumped_model(tmp_path)
    front, back = _figure_picture(tmp_path), tmp_path / "back.png"
    Image.new("RGB", (400, 600), (25, 25, 25)).save(back)

    fitted = front_view.fit_views(model, [(front, "front"), (back, "back")], tmp_path / "views")
    front_view_cam, back_cam = pp.load_views(fitted)[0]

    bump = pp.to_view_space(np.array([[1.2, 0.0, 0.0]]))   # the bump's centre, GLB +X
    fx, _, fdepth = pp.project(bump, front_view_cam)
    bx, _, bdepth = pp.project(bump, back_cam)
    assert (fdepth > 0).all() and (bdepth > 0).all()       # both cameras see it
    assert fx[0] > 200                                     # front: picture-right of centre
    assert bx[0] < 200                                     # back: picture-left of centre
