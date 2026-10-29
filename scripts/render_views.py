#!/usr/bin/env python3
"""Render a mesh through the fitted cameras nobody photographed, for repainting.

    <blender> --background --python scripts/render_views.py -- model.glb views_dir

Each transforms.json frame marked `render: true` names an angle with no picture — a
placeholder fitted to the mesh. This renders the mesh through that very camera, flat
clay grey on a transparent background, and writes the pixels over the placeholder: the
render agrees with the geometry by construction (an apron in the mesh is an apron in
the picture), and the repaint that follows only adds colour. A frame whose render
fails is dropped from transforms.json, so a flat placeholder is never painted.

Runs with Blender's own python, never in the repo venv: `bpy` comes with Blender.
Camera math: a fitted camera lives in photo_paint's view frame (PIXAL3D_FRAME); its
axes are mapped back to GLB space and into Blender's (glTF Y-up imported to Z-up),
where a camera looks down its local -Z exactly as photo_paint reads it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

CLAY = (0.62, 0.62, 0.62, 1.0)


def _to_blender(view: Vector) -> Vector:
    """View-frame coords -> GLB (PIXAL3D_FRAME inverse) -> Blender (Z-up)."""
    glb = Vector((-view.x, view.z, view.y))
    return Vector((glb.x, -glb.z, glb.y))


def _camera_matrix(frame: dict) -> Matrix:
    """The fitted view-space camera as a Blender world matrix (same -Z convention)."""
    cam = Matrix(frame["transform_matrix"])
    axes = [_to_blender(cam.col[i]) for i in range(3)]      # right, up, back
    world = Matrix(axes).transposed().to_4x4()              # columns = the camera's axes
    world.translation = _to_blender(cam.translation)
    return world


def _clay_material() -> bpy.types.Material:
    material = bpy.data.materials.new("clay")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = CLAY
    bsdf.inputs["Roughness"].default_value = 0.9
    return material


def render(source: Path, views_dir: Path) -> int:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        print(f"[render-views] no mesh in {source}", file=sys.stderr)
        return 1
    clay = _clay_material()
    for obj in bpy.context.scene.objects:
        obj.select_set(obj in meshes)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(clay)

    scene = bpy.context.scene
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    world = bpy.data.worlds.new("even")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.75, 0.75, 0.75, 1.0)
    background.inputs["Strength"].default_value = 0.9
    scene.world = world

    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = 36.0
    scene.collection.objects.link(camera)
    scene.camera = camera

    meta_path = views_dir / "transforms.json"
    meta = json.loads(meta_path.read_text())
    kept, failed = [], []
    wanted = sum(1 for f in meta["frames"] if f.get("render"))
    for frame in meta["frames"]:
        if not frame.get("render"):
            kept.append(frame)
            continue
        name = frame["file_path"]
        marker = views_dir / f"{Path(name).stem}.rendered"
        if marker.is_file():
            print(f"[render-views] {name}: kept (already rendered)", flush=True)
            kept.append(frame)
            continue
        try:
            placeholder = bpy.data.images.load(str(views_dir / name))
            width, height = placeholder.size[0], placeholder.size[1]
            bpy.data.images.remove(placeholder)
            scene.render.resolution_x = width
            scene.render.resolution_y = height
            scene.render.resolution_percentage = 100
            camera.matrix_world = _camera_matrix(frame)
            camera.data.lens = 18.0 / math.tan(float(frame["camera_angle_x"]) / 2)
            scene.render.filepath = str(views_dir / name)
            bpy.ops.render.render(write_still=True)
            marker.write_text("ok")
            print(f"[render-views] {name} rendered ({width}x{height})", flush=True)
            kept.append(frame)
        except Exception as exc:
            failed.append(name)
            print(f"[render-views] {name} failed ({exc}); dropped from the views",
                  flush=True)
    meta["frames"] = kept
    meta_path.write_text(json.dumps(meta, indent=2))
    survivors = sum(1 for f in kept if f.get("render"))
    if wanted and not survivors:
        print("[render-views] every render failed; nothing synthetic survives", flush=True)
        return 1
    return 0


def main() -> int:
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(args) != 2:
        print("usage: <blender> --background --python scripts/render_views.py -- "
              "model.glb views_dir", file=sys.stderr)
        return 2
    return render(Path(args[0]), Path(args[1]))


if __name__ == "__main__":
    raise SystemExit(main())
