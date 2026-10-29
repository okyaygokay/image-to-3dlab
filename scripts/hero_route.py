#!/usr/bin/env python3
"""Build a hero character: a full-body model with a high-detail head grafted on, as one GLB.

    .venv/bin/python scripts/hero_route.py picture.png [--back back.png] [--out output/hero/NAME]
    ... --body model.glb --body-views model.svviews [--body-dump model.post.bin] --result out.glb

A face that is a few dozen pixels of a full-body picture comes out of Pixal3D soft: goggle
eyes, smeared paint. The hero route builds the head again from an upscaled crop, where it
fills Pixal3D's whole box, and joins it to the body:

  1. body       Pixal3D on the picture (front, or front + back), Finish to --body-faces
                with Pixel Match.
  2. measure    a throwaway rig of the body. Its topmost bone region (+ the bones whose
                middle sits inside it) is the head: no names, no hand-drawn box.
  3. crop       the head's pixels, squared and padded, upscaled 4x with Real-ESRGAN.
  4. head       Pixal3D on the crop, cleaned up to --head-faces, Pixel Match from the crop.
  5. graft      the head placed on the body's head (same centre, same height), lined up,
                swapped in along the body head's drawn outline.
  6. join       one object, two materials (body, head): a skin later swaps only the body's
                paint sheet, so the face survives. Textures compressed.
  7. check      close-ups of the head and the whole figure, to judge before using it.

Rigging and animation are downstream: rig `hero.glb` like any model. Every step skips when
its output exists, so a rerun picks up after a fix. Where the code is unsure (a fallback,
a coin-flip check) the log says UNSURE: that is where a human should look.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "viewer"))
sys.path.insert(0, str(REPO / "scripts"))

# The Python running this script: a hard-coded .venv/bin/python does not exist on Windows.
PYTHON = Path(sys.executable)
GENERATE = REPO / "scripts" / "pixal3d_generate.py"
FINISH = REPO / "scripts" / "retopo_repaint.py"
UPSCALE = REPO / "scripts" / "upscale_esrgan.py"
RENDER_VIEWS = REPO / "scripts" / "render_views.py"
REPAINT_VIEWS = REPO / "scripts" / "repaint_views.py"
COLAB = REPO / "scripts" / "colab_generate.py"
BLENDER = REPO / "scripts" / "hero_blender.py"
COMPRESS = REPO / "scripts" / "compress_glb_textures.py"
BODY_FACES = 32000
HEAD_FACES = 16000
#: How much of a failed step's own output is echoed, so a bug report carries the real error.
FAIL_TAIL_LINES = 15

#: Lines from a leg worth saying out loud while it runs. A first-run install on the VM
#: is minutes of quiet otherwise — the studio's panel sat on "Finishing the body · 0:30"
#: for six straight minutes while the install worked fine, indistinguishable from a
#: wedged run (found live, 2026-10-29, on a warrior hero). Held in step with
#: `viewer/finish_api.py`'s copy by a test, so a line added here cannot silently miss
#: the panel.
LEG_LINES = ("[install", "[archive]", "[repaint]", "restored from Google Drive",
             "Google Drive is not mounted", "still working on the VM", "wrote ",
             "session '", "prefetch ", "SETUP_EXIT", "PREFETCH_EXIT")
SEED = 42
#: Finish settings shared with the studio's defaults, Pixel Match only (no repaint).
FINISH_FLAGS = ["--atlas", "2048", "--angle", "89.0", "--voxel", "0.004", "--metallic", "0.25",
                "--roughness", "0.65", "--ior", "1.45", "--paint-seed", "0", "--paint-res", "512",
                "--paint-steps", "15", "--paint-tex", "4096", "--texture-size", "2048", "--skip-paint"]
#: What else touched the output; Pixal3D's own licence is copied from its run records.
LICENCES = [
    {"component": "Real-ESRGAN x4plus", "license": "BSD-3-Clause", "purpose": "head crop upscale"},
    {"component": "SkinTokens", "license": "MIT", "purpose": "throwaway rig to find the head; not in the output"},
]


class Run:
    def __init__(self, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        # UTF-8 throughout: Windows defaults to cp1252, which can't write every folder name.
        self.log = open(out / "run.log", "a", buffering=1, encoding="utf-8")  # noqa: SIM115 - open for the whole run
        # Kept on disk by step: a rerun skips finished steps and their notes must survive, while
        # a step that runs again replaces its own old notes.
        self.unsure_file = out / "unsure.json"
        self.notes: dict[str, list[str]] = (json.loads(self.unsure_file.read_text(encoding="utf-8"))
                                            if self.unsure_file.is_file() else {})
        if isinstance(self.notes, list):                  # older runs kept a flat list
            self.notes = {"earlier": self.notes}
        self.current = "start"
        self.started = time.time()

    def say(self, text: str) -> None:
        line = f"[hero {time.time() - self.started:6.0f}s] {text}"
        print(line, flush=True)
        self.log.write(line + "\n")

    @property
    def unsure(self) -> list[str]:
        return [note for notes in self.notes.values() for note in notes]

    def flag(self, text: str) -> None:
        notes = self.notes.setdefault(self.current, [])
        if text not in notes:
            notes.append(text)
            self.unsure_file.write_text(json.dumps(self.notes, indent=2), encoding="utf-8")
        self.say(f"UNSURE: {text}")

    def run(self, command: list, cwd: Path | None = None, tries: int = 1, required: bool = True) -> bool:
        for attempt in range(1, tries + 1):
            self.say("$ " + " ".join(str(c) for c in command))
            process = subprocess.Popen([str(c) for c in command], cwd=cwd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
            tail: deque[str] = deque(maxlen=FAIL_TAIL_LINES)
            for line in process.stdout:
                line = line.rstrip("\n")
                if line.startswith(LEG_LINES):
                    self.say(line)          # live to the studio, not only run.log
                else:
                    self.log.write(line + "\n")
                tail.append(line)
            if not process.wait():
                return True
            if attempt < tries:
                self.flag(f"{Path(str(command[1])).name} failed once ({process.returncode}); tried again")
        if required:
            print("\n".join(tail), flush=True)   # run.log has it all; the studio only sees stdout
            last = next((line.strip() for line in reversed(tail) if line.strip()), "")
            raise SystemExit(f"[hero] step failed ({process.returncode})" + (f": {last}" if last else "")
                             + f"; see {self.out / 'run.log'}")
        return False

    def step(self, name: str, output: Path) -> bool:
        """True when the step must run; False (and a note) when its output is already there."""
        self.current = name
        if output.exists():
            self.say(f"{name}: kept {output.name}")
            return False
        self.notes.pop(name, None)
        self.notes.pop("earlier", None)
        self.unsure_file.write_text(json.dumps(self.notes, indent=2), encoding="utf-8")
        self.say(f"{name} ...")
        return True


STAGE_MARKER = "I2L_STAGE"   # the studio's Finish progress reads these (viewer/finish_api.py)
STAGES = ("body", "measure", "crop", "head", "graft", "join", "check")


def stage(run: Run, name: str, message: str) -> None:
    """Announce a stage on stdout for the studio, and in the log."""
    print(f"{STAGE_MARKER}::{name}::{message}", flush=True)
    run.log.write(f"{STAGE_MARKER}::{name}::{message}\n")


def missing_parts(parts: dict[str, Path]) -> list[str]:
    """The named parts whose file is not on disk: checked before 20 minutes of work, not after."""
    return [name for name, path in parts.items() if not Path(path).exists()]


def needed_parts(colab: bool = False) -> dict[str, Path]:
    """Everything the route runs, by what to install to get it.

    In the colab mode the model parts (trellis-cli, the upscaler, post-replay) live on the
    rented VM, so requiring them here would refuse a route that is about to work. The
    throwaway rigger rides the same VM when SkinTokens is not installed here (which is
    also what the Animate tab does); only when neither machine has one is it a local
    install to demand. Blender stays local in every mode: Finish is a local step.
    """
    import autorig_api
    import retopo_repaint

    parts = {
        "Blender (Finish)": retopo_repaint.find_blender() or Path("/missing/blender"),
    }
    if autorig_api.installed() or not (colab and autorig_api.colab_ready()):
        parts["Auto-rig (finds the head)"] = autorig_api.venv_python()
    if not colab:
        import pixal3d_generate

        from image_to_3dlab import pixal3d_native, upscale

        parts["Pixal3D (3D models)"] = pixal3d_generate.CLI
        parts["Hero face upscaler (Real-ESRGAN, 67 MB)"] = upscale.WEIGHTS_FILE
        # Upstream's old NVIDIA download lacks it; Setup's Rebuild for Pixal3D adds it.
        parts["Pixal3D's own clean-up (Setup: Rebuild Pixal3D)"] = pixal3d_native.POST_REPLAY
    return parts


NO_DUMP = ("this 3D model kept no clean-up data, so its body would stay at full size. Make the 3D "
           "model again (same picture), then make the Hero version from the new one.")


#: Checked before the run: every Hero step needs the dump, and a model made again on this
#: build would still have none.
NO_DUMP_BUILD = ("Pixal3D on this machine can't save its clean-up data, which the Hero version needs. "
                 "Open Setup and pick Rebuild for Pixal3D, then try again (nothing was downloaded).")


def bare_geometry(model: Path) -> bool:
    """True when a GLB paints nothing of its own: no image, no base-colour map, no vertex
    colours. Hi3D ships geometry only, so a Hero run over its mesh would finish onto the
    flat grey the bake falls back to, while the head gets the picture's own paint; such a
    body is better built from the picture by the route itself (which paints all round).
    """
    data = Path(model).read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        return False                    # not ours to judge; the finish says so if asked
    chunk, _ = struct.unpack_from("<II", data, 12)
    try:
        doc = json.loads(data[20:20 + chunk])
    except ValueError:
        return False
    if doc.get("images"):
        return False
    for material in doc.get("materials", []):
        if (material.get("pbrMetallicRoughness") or {}).get("baseColorTexture") is not None:
            return False
    for mesh in doc.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            if "COLOR_0" in primitive.get("attributes", {}):
                return False
    return True


def body_problem(body: Path | None, views: Path | None, dump: Path | None) -> str | None:
    """Why an existing model cannot be the Hero body, in plain words, or None.

    The cameras and the clean-up dump are optional: they are what a *Pixal3D* body brings
    (the maker's own cameras, Pixal3D's own retopo), and a body from any other engine
    (Hi3D, TRELLIS, ...) has neither -- it finishes the classic way, with a front camera
    fitted to it for Pixel Match. Only a *named* --body-views that points at no
    transforms.json is an error: someone asked for Pixel Match on a body that cannot
    have it.
    """
    if body is None:
        return None
    if views and not (Path(views) / "transforms.json").is_file():
        return "the body's cameras (--body-views) hold no transforms.json"
    return None


def pixal_command(colab: bool, image: Path, output: Path, *flags: str) -> list[str]:
    """A Pixal3D generation: this machine's trellis-cli, or the rented GPU.

    The colab form keeps its session (`--keep`): the head build, the upscale and the head
    clean-up all want the same VM within minutes of each other, and each one alone would
    otherwise stop the machine and orphan 8.4 GB of weights in /content. The idle reaper
    stops the session when the run is over.
    """
    if not colab:
        return [PYTHON, "-u", GENERATE, image, output, *flags]
    return [PYTHON, "-u", COLAB, image, output, "--backend", "pixal3d",
            "--keep", "--yes", "--", *flags]


def upscale_command(colab: bool, crop: Path, upscaled: Path, size: int = 1024) -> list[str]:
    """The head-crop upscale: local PyTorch, or the same script on the VM (--exec)."""
    if not colab:
        return [PYTHON, UPSCALE, crop, upscaled, "--size", str(size)]
    return [PYTHON, "-u", COLAB, "--backend", "pixal3d", "--exec", "scripts/upscale_esrgan.py",
            "--exec-input", crop, "--exec-output", upscaled, "--keep", "--yes",
            "--", "{in}", "{out}", "--size", str(size)]


def render_views_command(source: Path, views_dir: Path) -> list[str]:
    """Render the mesh through the fitted cameras nobody photographed — local Blender,
    the same one the Finish uses; a mesh operation this small never needs the rented GPU."""
    import retopo_repaint

    return [retopo_repaint.find_blender(), "--background", "--python", RENDER_VIEWS,
            "--", source, views_dir]


def repaint_command(colab: bool, card: Path, view: Path, render_size: tuple[int, int]) -> list[str]:
    """Recolour one render with Qwen in edit mode: locally with the repo's weights, or
    as the `qwen_image` backend's `--exec` worker on the rented VM. The card carries
    everything the worker needs but the render's size, which says where its strip ends."""
    common = ["--render", f"{render_size[0]}x{render_size[1]}"]
    if not colab:
        return [PYTHON, REPAINT_VIEWS, card, view, *common]
    return [PYTHON, "-u", COLAB, "--backend", "qwen_image", "--exec", "scripts/repaint_views.py",
            "--exec-input", card, "--exec-output", view,
            "--exec-artifacts", f"{view.stem}.provenance.json",
            "--keep", "--yes", "--", "{in}", "{out}", *common]


#: The shots the check renders beside its first one (hero_blender names them itself).
CHECK_SHOTS = ("head_three_quarter.png", "head_side.png", "head_back.png",
               "body_front.png", "body_three_quarter.png", "body_side.png")


def blender_command(remote: bool, sub: str, source: Path, target: Path) -> list[str]:
    """`hero_blender.py` wherever bpy lives: the local SkinTokens venv when installed,
    else the rented VM pattern the throwaway rigger uses — the only way a Mac with no
    local SkinTokens gets a join or a check sheet.

    `target` is the command's one fetched output: the joined GLB for `join`, the check's
    `head_front.png` for `check` (its directory is the remote output's parent, and the
    other shots ride `--exec-artifacts`)."""
    import autorig_api

    if not remote:
        local_arg = target.parent if sub == "check" else target
        return [autorig_api.venv_python(), BLENDER, sub, source, local_arg]
    command = [PYTHON, "-u", COLAB, "--backend", "autorig",
               "--exec", "scripts/hero_blender.py", "--exec-input", source,
               "--exec-output", target, "--keep", "--yes"]
    if sub == "check":
        command += ["--exec-artifacts", *CHECK_SHOTS]
    return command + ["--", sub, "{in}", "{out}" if sub != "check" else "{outdir}"]


def replay_command(colab: bool, dump: Path, output: Path, faces: int, atlas: int = 2048) -> list[str]:
    """The head clean-up: local post-replay, or the prebuilt binary on the VM (--exec-bin)."""
    from image_to_3dlab import pixal3d_native

    if not colab:
        return pixal3d_native.replay_command(dump, output, faces, atlas)
    # --exec-bin only says *how* to run the script -- without --exec the supervisor takes
    # the whole thing for a generation and dies on the missing image argument, one live
    # run deep into the hero route. Hence: both flags, always.
    return [PYTHON, "-u", COLAB, "--backend", "pixal3d",
            "--exec", "vendor/pixal3d-cpp/build/post-replay", "--exec-bin",
            "--exec-input", dump,
            "--exec-output", output, "--keep", "--yes",
            "--", "{in}", "{out}", "--no-remesh", "--faces", str(int(faces)),
            "--atlas", str(int(atlas))]


def finish(run: Run, model: Path, picture: Path, views: Path | None, out: Path, faces: int,
           dump: Path | None, classic: bool = False) -> None:
    """Finish one model: retopo to FACES, and Pixel Match when a camera exists for it.

    `views` is the body's pictures-and-cameras directory: a Pixal3D body's own
    ``.svviews``, or the fitted views `prepare_body_views` made for a foreign body
    (None when even fitting failed: no photo stage, the model keeps its own texture).
    The head's calls always bring the head run's own views. `dump` is None when the
    body is not a Pixal3D model: the clean-up is the classic route — which a *bare*
    foreign body (Hi3D, no UVs, no paint) needs anyway, `classic` or not, and main()
    passes it explicitly so the resurface is said in the run. A Pixal3D body whose dump
    was lost still keeps its mesh (--keep-mesh), as before. `classic` forces the
    classic clean-up even on a Pixal3D model — the head's fallback when post-replay
    fails.
    """
    command = [PYTHON, "-u", FINISH, model, picture, out, "--steps-dir", out.with_suffix(".steps"),
               "--faces", str(faces), *FINISH_FLAGS]
    if views is not None:
        command += ["--views", views]
    if dump:
        command += ["--native-dump", dump]
    elif not classic and views is not None:
        command += ["--keep-mesh"]
    run.run(command)
    record = json.loads(out.with_suffix(".retopo-repaint.json").read_text())
    route = (record.get("retopology") or {}).get("route")
    if dump and route == "classic":
        run.flag(f"Finish of {model.name} fell back to the classic clean-up (its frame check "
                 "objected; see run.log): hair and thin parts can shatter")


#: The canvas the angles nobody photographed are rendered at: portrait, like the
#: pictures themselves, and a size the edit returns unscaled.
RENDER_SIZE = (768, 1024)

#: The side angles every foreign body gets; the back joins them unless a real back
#: picture was supplied (a real photo beats anything made).
SIDE_ANGLES = (("left", *RENDER_SIZE), ("right", *RENDER_SIZE))


def prepare_body_views(run: Run, model: Path, picture: Path, back: Path | None,
                       out: Path, colab: bool = False) -> Path | None:
    """Pictures and fitted cameras for a body whose maker saved none; None if even
    fitting fails (the body then keeps its own texture, as before).

    The front is the real photo, cut out (Pixel Match reads the alpha as its matte).
    A supplied back picture joins it the same way. The angles nobody photographed —
    back (when not supplied) and the two sides — are *rendered from the mesh itself*
    through their fitted cameras and recoloured by Qwen with the front as the colour
    reference, so every picture agrees with the geometry underneath: a freely drawn
    view does not (Qwen invented crossed apron straps on a mesh whose back is a full
    apron, 2026-10-28). Each step is best-effort: a render or repaint that fails drops
    its view and the run says so — the front alone still paints.
    """
    from image_to_3dlab.front_view import fit_views

    front = cutout_picture(run, picture)
    pictures = [(front, "front")] + ([(cutout_picture(run, back), "back")] if back else [])
    renders = ([] if back else [("back", *RENDER_SIZE)]) + list(SIDE_ANGLES)
    fitted = out.with_suffix(".fitted-views")
    if not (fitted / "transforms.json").is_file():   # never clobber rendered/painted views
        try:
            fit_views(model, pictures, fitted, renders=renders)
        except Exception as exc:  # an unreadable mesh must not sink the whole run
            run.flag(f"could not fit cameras to {model.name} ({exc}); "
                     f"Pixel Match on the body is skipped")
            return None
    _render_body_views(run, model, fitted, front, colab)
    out.with_suffix(".views.done").write_text("ok")
    run.flag("the body's maker saved no cameras, so cameras were fitted to it and the "
             "angles nobody photographed were rendered from the mesh and repainted: "
             "Pixel Match paints it from the photo and those (fitted cameras, not the "
             "maker's)")
    return fitted


def _render_body_views(run: Run, model: Path, fitted: Path, front: Path, colab: bool) -> None:
    """Render the synthetic views with Blender, then recolour each with Qwen. Best-effort:
    a view whose render or repaint fails is dropped from the views, said in the run."""
    from PIL import Image

    from image_to_3dlab import repaint_view

    if not run.run(render_views_command(model, fitted), required=False):
        run.flag("the body's views could not be rendered (see run.log); Pixel Match "
                 "paints from the pictures that exist")
    meta = json.loads((fitted / "transforms.json").read_text())
    for frame in list(meta["frames"]):
        if not frame.get("render"):
            continue
        stem = Path(frame["file_path"]).stem
        view = fitted / frame["file_path"]
        if not (fitted / f"{stem}.rendered").is_file():
            _drop_view(fitted, meta, frame, f"{stem} was not rendered")
            continue
        if (fitted / f"{stem}.painted").is_file():
            continue
        card = fitted / f"{stem}__card.png"
        with Image.open(view) as render:
            repaint_view.reference_card(render, Image.open(front)).save(card)
        if run.run(repaint_command(colab, card, view, render.size), required=False):
            (fitted / f"{stem}.painted").write_text("ok")
            continue
        _drop_view(fitted, meta, frame, f"{stem} could not be repainted")


def _drop_view(fitted: Path, meta: dict, frame: dict, why: str) -> None:
    meta["frames"] = [f for f in meta["frames"] if f is not frame]
    (fitted / "transforms.json").write_text(json.dumps(meta, indent=2))
    print(f"[hero] view dropped: {why}", flush=True)


def cutout_picture(run: Run, picture: Path) -> Path:
    """`picture` as a cut-out, or itself when it already is one.

    The cut-out lands beside the picture at `<stem>_cutout.png` — the very name the
    hero route's own cut-out step gives the front picture, so an existing cut-out is
    reused rather than redone. Cutting needs the matte the cut-out step already used;
    a picture that cannot be cut is passed through as it is, and the silhouette fade
    keeps its background off the mesh except right at the edge.
    """
    from PIL import Image

    from image_to_3dlab import matte

    with Image.open(picture) as opened:
        if matte.is_matted(opened):
            return picture
    cut = picture.with_name(f"{picture.stem}_cutout.png")
    if not cut.is_file():
        try:
            image, used = matte.cut_out(Image.open(picture).convert("RGB"))
            image.save(cut)
            run.say(f"cut out {picture.name} with {used} for Pixel Match")
        except Exception as exc:               # the picture itself is still usable
            run.flag(f"could not cut out {picture.name} ({exc}); Pixel Match paints from "
                     f"it as it is, background and all")
            return picture
    return cut


def head_cleanup(run: Run, raw: Path, out: Path, faces: int, colab: bool = False) -> bool:
    """Pixal3D's own clean-up at FACES; False when post-replay fails. Its mirror check is a
    coin-flip on a front-facing, near-symmetric head (0.2% vs 0.3% on the elf) and would drop
    to the classic route, which shatters hair; a head crop is always front-facing, so convert
    directly and say so."""
    from image_to_3dlab import pixal3d_native as native

    replay = out.with_name(out.stem + ".replay.glb")
    if not run.run(replay_command(colab, raw.with_suffix(native.DUMP_SUFFIX), replay, faces, 2048),
                   required=False):
        return False
    out.write_bytes(native.convert_replay_glb(replay.read_bytes()))
    return True


def finish_head(run: Run, raw: Path, picture: Path, clean: Path, head: Path, faces: int,
                colab: bool = False) -> None:
    """The head, cleaned up and Pixel Matched into HEAD. When Pixal3D's own clean-up fails
    (seen on Windows), the classic Finish does both in one go instead of failing the run;
    CLEAN then holds that finished head, so a rerun skips the step."""
    views = raw.with_suffix(".svviews")
    if run.step(f"head: clean-up {faces}", clean):
        if head_cleanup(run, raw, clean, faces, colab):
            run.flag("head clean-up skips Finish's mirror check (front-facing crop); glance at the check sheet")
        else:
            run.flag("Pixal3D's own clean-up failed on the head (see run.log); it took the classic "
                     "clean-up instead: hair and thin parts can shatter")
            finish(run, raw, picture, views, head, faces, None, classic=True)
            shutil.copy(head, clean)
    if run.step("head: Pixel Match", head):
        finish(run, clean, picture, views, head, faces, None)


def stage_body_inputs(body: Path, views: Path | None, dump: Path | None, record: Path | None,
                      body_raw: Path) -> None:
    """Bring an existing model in as the body, with whatever artifacts its run saved.

    Every artifact is optional -- None is a body from another engine (a Hi3D mesh has no
    cameras, no clean-up dump, no record), and the route finishes it the classic way. The
    first live run of that path died copying a None -- hence a helper, and a test.
    """
    if views is not None:
        shutil.copytree(views, body_raw.with_suffix(".svviews"), dirs_exist_ok=True)
    if dump and dump.is_file():
        shutil.copy(dump, body_raw.with_suffix(".post.bin"))
    if record and record.is_file():
        shutil.copy(record, body_raw.with_suffix(".json"))
    shutil.copy(body, body_raw)


def main() -> int:
    # One pipeline, many legs — and each leg's child wrapper reads this: no background
    # prefetch while the pipeline runs, because an install eating the VM's disk
    # mid-flight is what killed its Pixal3D leg at 1 GB free (2026-10-28, twice).
    os.environ["I3L_PIPELINE"] = "1"
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("picture", type=Path)
    parser.add_argument("--back", type=Path, default=None, help="a back view of the same picture")
    parser.add_argument("--out", type=Path, default=None, help="run folder (default output/hero/<picture>)")
    parser.add_argument("--body-faces", type=int, default=BODY_FACES)
    parser.add_argument("--head-faces", type=int, default=HEAD_FACES)
    parser.add_argument("--match-paint", action="store_true",
                        help="also move the head's paint to the body's level (metal and roughness always are)")
    parser.add_argument("--cut-at", default=None,
                        help="swap only the head above this height (model units), or 'neck' to find it; "
                             "default swaps the whole head region")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--body", type=Path, default=None,
                        help="an existing Pixal3D model of the picture, used as the body instead of building one")
    parser.add_argument("--body-views", type=Path, default=None, help="that model's cameras (its .svviews folder)")
    parser.add_argument("--body-dump", type=Path, default=None, help="that model's clean-up dump (.post.bin)")
    parser.add_argument("--body-record", type=Path, default=None, help="that model's run record, for provenance")
    parser.add_argument("--result", type=Path, default=None, help="also copy the finished GLB here")
    parser.add_argument("--pixal", choices=("local", "colab"), default="local",
                        help="where the Pixal3D calls run: this machine's trellis-cli, or a "
                             "rented Colab GPU through scripts/colab_generate.py. The studio "
                             "passes colab for a model made by the colab-pixal3d engine.")
    args = parser.parse_args()

    missing = missing_parts(needed_parts(colab=args.pixal == "colab"))
    if missing:
        print("[hero] cannot start, not installed: " + "; ".join(missing)
              + ". Install them from the Setup page (nothing was downloaded).", file=sys.stderr)
        return 2
    import pixal3d_generate

    if args.pixal == "local" and not pixal3d_generate.dump_supported():
        print(f"[hero] {NO_DUMP_BUILD}", file=sys.stderr)
        return 2
    problem = body_problem(args.body, args.body_views, args.body_dump)
    if problem:
        print(f"[hero] {problem}", file=sys.stderr)
        return 2

    import numpy as np
    import trimesh
    from PIL import Image

    from image_to_3dlab import hero, matte
    from image_to_3dlab.pixal3d_native import smooth_normals
    from image_to_3dlab.provenance import sha256_file

    out = (args.out or REPO / "output" / "hero" / args.picture.stem[:60]).resolve()
    run = Run(out)
    colab = args.pixal == "colab"
    run.say(f"hero route: {args.picture} -> {out}"
            + (" (Pixal3D calls on a rented Colab GPU)" if colab else ""))
    picture = out / "picture.png"
    if not picture.exists():
        shutil.copy(args.picture, picture)
    back = None
    if args.back:
        back = out / "back.png"
        if not back.exists():
            shutil.copy(args.back, back)
    cutout = out / "picture_cutout.png"
    if run.step("cut-out", cutout):
        image, used = matte.cut_out(Image.open(picture).convert("RGB"))
        image.save(cutout)
        run.say(f"cut out with {used}")

    # 1. body
    body_raw = out / "body_raw.glb"
    stage(run, "body", "Building the body" if not args.body else "Finishing the body")
    body_classic = False
    if args.body and not body_raw.exists():
        # The studio's own 3D model of this picture: no second Pixal3D run for the body.
        stage_body_inputs(args.body, args.body_views, args.body_dump, args.body_record, body_raw)
        run.say(f"body: using {args.body}")
    if args.body and bare_geometry(body_raw) \
            and not (body_raw.with_suffix(".svviews") / "transforms.json").is_file():
        # Geometry only (Hi3D): no UVs, no paint, nowhere to put pixels — a direct Pixel
        # Match has no atlas to paint into (a smart-UV unwrap of the engine's own million
        # faces fragmented to ~1% of the texture, 2026-10-28). Finish's classic route
        # resurfaces it — the voxel remesh keeps its shape at the route's own face count,
        # with a dense atlas — and the fitted cameras paint it from the photo, front and
        # drawn back. The old way (build the body again with Pixal3D) threw the shape away.
        body_classic = True
        run.say(f"body: {args.body.name} paints nothing of its own (Hi3D ships geometry "
                "only), so it keeps its shape, resurfaced, and is painted from the picture")
    if run.step("body: Pixal3D", body_raw):
        run.run(pixal_command(colab, picture, body_raw, "--seed", str(args.seed),
                              *(["--back", str(back)] if back else [])))
    body = out / "body.glb"
    body_views = body_raw.with_suffix(".svviews")
    have_views = (body_views / "transforms.json").is_file()
    dump = body_raw.with_suffix(".post.bin")
    body_views_for_finish = body_views if have_views else None
    if not have_views:
        # A foreign body paints through cameras fitted to it: the front from the real
        # photo; the angles nobody photographed (back unless a real --back was supplied,
        # and the sides) rendered from the mesh itself and repainted with the photo's
        # colours — a picture that cannot disagree with the geometry underneath.
        fitted_dir = body.with_suffix(".fitted-views")
        if run.step("body: views (fit, render, repaint)", body.with_suffix(".views.done")):
            prepare_body_views(run, body_raw, picture, back, body, colab)
        if (fitted_dir / "transforms.json").is_file():
            body_views_for_finish = fitted_dir
    label = f"body: Finish {args.body_faces}"
    if have_views:
        label += " + Pixel Match"
    elif body_views_for_finish:  # a foreign body: fitted cameras, real photo + repainted views
        label += " + Pixel Match (front + repainted views)"
    if run.step(label, body):
        # A body this run just built with Pixal3D always has its dump; missing means the
        # build predates the patch, and the classic route would shatter its hair exactly
        # where the head is about to be judged. A body handed in from another engine
        # (Hi3D, TRELLIS, ...) has neither file, by design.
        if not args.body and not dump.is_file():
            raise SystemExit(f"[hero] {NO_DUMP}")
        finish(run, body_raw, picture, body_views_for_finish, body,
               args.body_faces, dump if dump.is_file() else None, classic=body_classic)

    # 2. measure
    stage(run, "measure", "Finding the head")
    rig = out / "measure_rig.glb"
    if run.step("measure: throwaway rig", rig):
        import autorig_api

        # The rigger runs wherever a rigger exists: here when installed, else on the rented
        # GPU -- the same fallback the Animate tab uses, and the only way a Mac with no
        # local SkinTokens gets a hero at all.
        commands = (autorig_api.rig_commands(body, rig) if autorig_api.installed()
                    or not autorig_api.colab_ready()
                    else autorig_api.remote_rig_commands(body, rig))
        # The rigger's own Blender server sometimes answers its first request badly
        # (BadStatusLine on the long-haired woman); a second go is cheap next to a lost run.
        for command, cwd in commands:
            run.run(command, cwd=cwd, tries=2)

    # 3. crop
    stage(run, "crop", "Cropping and upscaling the head")
    body_mesh = trimesh.load(body, force="mesh", process=False)
    positions, joints = hero.skinned_vertices(rig.read_bytes())
    bone = hero.regions(body_mesh.vertices, positions, joints)
    picked = hero.head_bones(bone, body_mesh.vertices[:, 1])
    cut = hero.load_picture(cutout)
    full_box = hero.picture_box(cut)
    px, py = hero.project(body_mesh.vertices[np.isin(bone, picked)], hero.frame(body_mesh.vertices, full_box))
    crop = hero.crop_box(px, py, cut.width, cut.height)
    run.say(f"head bones {picked}, crop box {crop} ({crop[2] - crop[0]} px)")
    # The degenerate rig that shipped a whole-body "head" (2026-10-29) exited 0 and said
    # nothing; the share of the body in the picked regions is what betrays it.
    problem = hero.crop_problem(picked, float(np.isin(bone, picked).mean()))
    if problem:
        run.flag(problem)
        raise SystemExit(f"[hero] {problem}")
    if crop[2] - crop[0] < 48:
        run.flag(f"the head is only {crop[2] - crop[0]} px wide in the picture: the crop holds little detail")
    crop_raw, crop_up = out / "head_crop_raw.png", out / "head_crop.png"
    if run.step("crop + Real-ESRGAN", crop_up):
        Image.open(picture).convert("RGB").crop(crop).save(crop_raw)
        run.run(upscale_command(colab, crop_raw, crop_up))
    crop_cutout = out / "head_crop_cutout.png"
    if run.step("crop cut-out", crop_cutout):
        matte.cut_out(Image.open(crop_up).convert("RGB"))[0].save(crop_cutout)
    head_picture, head_cutout, head_crop = crop_up, crop_cutout, crop

    # 4. head
    stage(run, "head", "Building the head on its own (the long step)")
    head_raw = out / "head_raw.glb"
    if run.step("head: Pixal3D on the crop", head_raw):
        run.run(pixal_command(colab, head_picture, head_raw, "--seed", str(args.seed)))
    head = out / "head.glb"
    finish_head(run, head_raw, head_picture, out / "head_clean.glb", head, args.head_faces, colab)

    head_baked = out / "head_baked.glb"
    if run.step("head: bake normal map from the raw head", head_baked):
        # The body's Finish bakes one; a head without would light differently next to it.
        # Finish skips the bake for a kept mesh, so run its bake step directly.
        import retopo_repaint

        weights = head.with_suffix(".steps") / "3_photo_weights.png"
        fade = (retopo_repaint.normal_fade_mask(weights, 2048, out / "head_photo_fade.png")
                if weights.is_file() else None)
        if not run.run(retopo_repaint.bake_command(head_raw, head, head_baked, 2048,
                                                   retopo_repaint.find_blender(), fade), required=False):
            # The bake refuses when the raw and cleaned heads do not line up within 5%; a hero
            # without a head bump map is still a hero, so carry on and say so.
            shutil.copy(head, head_baked)
            run.flag("the head's bump-map bake refused (see run.log); the head has no bump map")

    # 5. graft
    stage(run, "graft", "Putting the new head on the body")
    grafted = out / "grafted.glb"
    if run.step("graft", grafted):
        head_mesh = trimesh.load(head_baked, force="mesh", process=False)
        crop_pic = hero.load_picture(head_cutout)
        part_body, part_head, report = hero.graft(
            body_mesh, head_mesh, full_box, hero.picture_box(crop_pic), head_crop, crop_pic.width,
            bone, (cut.height, cut.width),
            cut_at=args.cut_at if args.cut_at in (None, "neck") else float(args.cut_at))
        run.say(f"graft {report}")
        # One piece: the head's paint and material always follow the body's own head.
        real = np.flatnonzero(np.isin(bone, picked))
        reach = 0.02 * float(np.ptp(body_mesh.vertices[real, 1]))
        head_idx, body_idx = hero.paired_points(part_head.vertices, body_mesh.vertices[real], reach)
        run.say(f"colour pairs: {len(head_idx)} head points matched to the body's own head")
        head_uv, body_uv = part_head.visual.uv[head_idx], body_mesh.visual.uv[real][body_idx]
        head_mat, body_mat = part_head.visual.material, body_mesh.visual.material
        slots = [("metallicRoughnessTexture", (1, 2))]
        if args.match_paint:
            slots.insert(0, ("baseColorTexture", (0, 1, 2)))
        for slot, channels in slots:
            head_tex, body_tex = getattr(head_mat, slot, None), getattr(body_mat, slot, None)
            if head_tex is None or body_tex is None:
                run.flag(f"no {slot} on the head or body: not matched")
                continue
            matched, how = hero.match_to_body(head_tex, hero.sample_texture(head_tex, head_uv),
                                              hero.sample_texture(body_tex, body_uv), channels)
            setattr(head_mat, slot, matched)
            run.say(f"head {slot} matched to body {how}")
            if how["big_change"]:
                run.flag(f"the head's {slot} was far from the body's ({how}); matched, worth a glance")
        if getattr(head_mat, "metallicRoughnessTexture", None) is not None:
            head_mat.metallicFactor = head_mat.roughnessFactor = 1.0
        if not 0.85 < report["scale_to_real_head"] < 1.15:
            run.flag(f"the new head needed scaling by {report['scale_to_real_head']} to fit: check the fit")
        scene = trimesh.Scene()
        for name, part in (("Body", part_body), ("Head", part_head)):
            # Without normals a glTF renders every triangle flat.
            part.vertex_normals = smooth_normals(part.vertices, part.faces)
            scene.add_geometry(part, node_name=name)
        scene.export(grafted, include_normals=True)
        (out / "graft.json").write_text(json.dumps({"crop": crop, **report}, indent=2))

    # 6. join
    stage(run, "join", "Joining head and body into one model")
    import autorig_api

    joined, final = out / "joined.glb", out / "hero.glb"
    remote_blender = not autorig_api.installed() and autorig_api.colab_ready()
    if run.step("join", joined):
        run.run(blender_command(remote_blender, "join", grafted, joined))
    if run.step("compress textures", final):
        run.run([PYTHON, COMPRESS, joined, final])

    # 7. check
    stage(run, "check", "Rendering the check sheet")
    check = out / "check"
    if run.step("check sheet", check / "head_front.png"):
        run.run(blender_command(remote_blender, "check", final, check / "head_front.png"))
        shots = [Image.open(check / f"{name}.png").convert("RGB") for name in
                 ("head_front", "head_three_quarter", "head_side", "head_back", "body_front", "body_three_quarter")]
        sheet = Image.new("RGB", (sum(s.width for s in shots), shots[0].height))
        x = 0
        for shot in shots:
            sheet.paste(shot, (x, 0))
            x += shot.width
        sheet.save(out / "check_sheet.jpg", quality=90)

    sub_runs = []
    for model in (body_raw, head_raw):
        if model.with_suffix(".json").is_file():
            sub_runs.append(json.loads(model.with_suffix(".json").read_text()))
        else:
            run.flag(f"{model.name} has no run record: its licence is not in the provenance")
            sub_runs.append({})
    record = {
        "schema_version": 1, "route": "hero", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input": {"picture": str(args.picture), "sha256": sha256_file(picture),
                  "back": str(args.back) if args.back else None},
        "output": {"path": str(final), "sha256": sha256_file(final)},
        "license": sub_runs[0].get("license"),
        "components": [c for s in sub_runs for c in s.get("components", [])] + LICENCES,
        "unsure": run.unsure,
        "parameters": {"body_faces": args.body_faces, "head_faces": args.head_faces, "seed": args.seed,
                       "crop": list(crop), "head_bones": picked},
        "sub_runs": [str(p.with_suffix(".json")) for p in (body_raw, head_raw)],
    }
    final.with_suffix(".provenance.json").write_text(json.dumps(record, indent=2))
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(final, args.result)
        args.result.with_suffix(".provenance.json").write_text(
            json.dumps({**record, "output": {"path": str(args.result), "sha256": sha256_file(args.result)}}, indent=2))
    run.say(f"done: {final} ({final.stat().st_size / 1e6:.1f} MB); check sheet {out / 'check_sheet.jpg'}")
    for note in run.unsure:
        run.say(f"  to check: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
