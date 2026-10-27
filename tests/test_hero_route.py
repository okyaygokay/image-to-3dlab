"""The hero command's own logic: what it checks before starting, and what the studio reads.

The heavy steps run real models and are exercised by hand; these pin the cheap parts that
decide whether a 25-minute run starts at all and whether the studio can follow it.
"""
from __future__ import annotations

import io
import sys

from scripts import hero_route


def test_missing_parts_names_only_what_is_absent(tmp_path):
    present = tmp_path / "here"
    present.write_text("x")
    parts = {"Pixal3D": present, "Auto-rig": tmp_path / "gone"}
    assert hero_route.missing_parts(parts) == ["Auto-rig"]
    assert hero_route.missing_parts({"Pixal3D": present}) == []


def test_stage_prints_the_marker_the_studio_reads(capsys, tmp_path):
    class Run:
        log = io.StringIO()

    hero_route.stage(Run, "head", "Building the head")
    assert capsys.readouterr().out == "I2L_STAGE::head::Building the head\n"
    assert "I2L_STAGE::head::Building the head" in Run.log.getvalue()


def test_stages_are_in_running_order():
    assert hero_route.STAGES == ("body", "measure", "crop", "head", "graft", "join", "check")


def test_hero_needs_the_upscaler_not_the_paint_install():
    parts = hero_route.needed_parts()
    assert any("upscaler" in name.lower() for name in parts)
    assert not any("hunyuan" in name.lower() or "hunyuan" in str(path).lower() for name, path in parts.items())


def test_hero_runs_its_steps_with_the_python_running_it():
    """Not a hard-coded .venv/bin/python: that path does not exist on Windows."""
    import sys
    from pathlib import Path

    assert hero_route.PYTHON == Path(sys.executable)
    assert hero_route.PYTHON.exists()


def test_hero_checks_for_pixal3ds_own_clean_up_before_starting():
    from image_to_3dlab import pixal3d_native

    assert pixal3d_native.POST_REPLAY in hero_route.needed_parts().values()


def _body(tmp_path, views=True, dump=True):
    body = tmp_path / "model.glb"
    body.write_bytes(b"glb")
    folder = tmp_path / "model.svviews"
    if views:
        folder.mkdir()
        (folder / "transforms.json").write_text("{}")
    post = tmp_path / "model.post.bin"
    if dump:
        post.write_bytes(b"dump")
    return body, folder, post


def test_a_body_with_cameras_and_a_dump_is_accepted(tmp_path):
    assert hero_route.body_problem(*_body(tmp_path)) is None


def test_a_body_from_another_engine_has_neither_file_and_is_fine(tmp_path):
    """The point of the route change: a Hi3D or TRELLIS body has no cameras and no dump,
    and is accepted -- it finishes the classic way and keeps its own texture."""
    body, _folder, _post = _body(tmp_path)
    assert hero_route.body_problem(body, None, None) is None


def test_cameras_that_point_at_nothing_are_an_error_in_plain_words(tmp_path):
    """Someone *asked* for Pixel Match on a body that cannot have it: a --body-views
    directory with no transforms.json is a mistake, not a fallback."""
    body, _folder, post = _body(tmp_path)
    empty = tmp_path / "no-cameras"
    empty.mkdir()
    assert "transforms.json" in hero_route.body_problem(body, empty, post)


def test_no_body_given_is_no_problem():
    assert hero_route.body_problem(None, None, None) is None


def test_a_failed_step_prints_the_end_of_its_own_output(capsys, tmp_path):
    import sys

    import pytest

    run = hero_route.Run(tmp_path)
    script = "import sys\nfor i in range(30): print(f'line {i}')\nsys.exit('the real error')"
    with pytest.raises(SystemExit, match=r"step failed \(1\): the real error; see "):
        run.run([sys.executable, "-u", "-c", script])   # -u, as every real step runs
    out = capsys.readouterr().out
    assert "the real error" in out and "line 29" in out
    assert "line 0\n" not in out                      # only the end, so the report keeps room
    assert "line 0" in (tmp_path / "run.log").read_text()


def test_a_step_that_works_prints_nothing_of_its_own(capsys, tmp_path):
    import sys

    hero_route.Run(tmp_path).run([sys.executable, "-c", "print('quiet')"])
    assert "quiet" not in capsys.readouterr().out.splitlines()   # the $ line names it; output does not


def test_a_step_that_works_on_its_second_try_prints_nothing_of_its_own(capsys, tmp_path):
    import sys

    flag = tmp_path / "tried"
    script = (f"import pathlib, sys; f = pathlib.Path({str(flag)!r})\n"
              "if not f.exists(): f.write_text('x'); print('first go broke'); sys.exit(1)")
    assert hero_route.Run(tmp_path).run([sys.executable, "-c", script], tries=2)
    assert "first go broke" not in capsys.readouterr().out.splitlines()


def test_the_log_takes_any_folder_name_and_any_byte(tmp_path):
    import sys

    run = hero_route.Run(tmp_path / "Ünïcødé 名前")
    run.say("hero route: 名前.png")
    run.run([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'ok \\x81\\xff\\n')"])
    run.flag("名前 looked odd")
    text = (tmp_path / "Ünïcødé 名前" / "run.log").read_text(encoding="utf-8")
    assert "名前.png" in text and "ok " in text
    assert "名前 looked odd" in hero_route.Run(tmp_path / "Ünïcødé 名前").unsure


def test_hero_stops_before_any_work_on_a_build_that_cannot_save_its_clean_up(monkeypatch, tmp_path, capsys):
    import sys

    import pixal3d_generate

    monkeypatch.setattr(hero_route, "needed_parts", lambda **_kwargs: {})
    monkeypatch.setattr(pixal3d_generate, "dump_supported", lambda cli=None: False)
    monkeypatch.setattr(sys, "argv", ["hero_route.py", str(tmp_path / "x.png"), "--out", str(tmp_path / "run")])
    assert hero_route.main() == 2
    assert "Rebuild for Pixal3D" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()                # nothing started


# --- the Colab mode: the model calls ride a rented GPU ---------------------------------------


def test_the_colab_mode_builds_supervisor_commands_instead_of_local_ones():
    """Head build, crop upscale and head clean-up are the three model calls; in the colab
    mode each becomes a supervisor invocation, and each keeps the session warm because the
    next call wants the same VM's weights."""
    from pathlib import Path

    head = hero_route.pixal_command(True, Path("crop.png"), Path("head.glb"), "--seed", "42")
    assert str(head[2]).endswith("colab_generate.py")
    assert head[head.index("--backend") + 1] == "pixal3d"
    assert "--keep" in head and "--yes" in head
    assert head[head.index("--") + 1:] == ["--seed", "42"]

    up = hero_route.upscale_command(True, Path("c.png"), Path("c_up.png"))
    assert up[up.index("--exec") + 1] == "scripts/upscale_esrgan.py"
    assert str(up[up.index("--exec-input") + 1]) == "c.png"
    assert str(up[up.index("--exec-output") + 1]) == "c_up.png"
    assert up[up.index("--") + 1:] == ["{in}", "{out}", "--size", "1024"]

    replay = hero_route.replay_command(True, Path("d.post.bin"), Path("r.glb"), 16000)
    assert replay[replay.index("--exec") + 1] == "vendor/pixal3d-cpp/build/post-replay"
    assert "--exec-bin" in replay          # the how; --exec above is the what (a live run died without it)
    assert replay[replay.index("--exec-input") + 1] == Path("d.post.bin")
    assert replay[replay.index("--exec-output") + 1] == Path("r.glb")
    assert replay[replay.index("--") + 1:] == ["{in}", "{out}", "--no-remesh",
                                               "--faces", "16000", "--atlas", "2048"]


def test_the_local_mode_unchanged_by_the_colab_routing():
    from pathlib import Path

    head = hero_route.pixal_command(False, Path("p.png"), Path("h.glb"), "--seed", "7")
    assert head == [hero_route.PYTHON, "-u", hero_route.GENERATE,
                    Path("p.png"), Path("h.glb"), "--seed", "7"]
    assert hero_route.COLAB not in head


def test_needed_parts_in_colab_mode_drops_the_model_parts(monkeypatch, tmp_path):
    """trellis-cli, the upscaler and post-replay live on the VM in that mode; a Mac with none
    of them must still pass the gate, while the rigger and Blender stay required."""
    import types

    rig = types.ModuleType("autorig_api")
    rig.venv_python = lambda root=None: tmp_path / "rig-python"
    rig.installed = lambda root=None: False
    rig.colab_ready = lambda: False   # not signed in: the rigger is demanded locally
    repaint = types.ModuleType("retopo_repaint")
    repaint.find_blender = lambda: tmp_path / "blender"
    # setitem, not a bare assignment: monkeypatch puts the real modules back afterwards,
    # so later tests (and the suite's other copies) still see them.
    monkeypatch.setitem(sys.modules, "autorig_api", rig)
    monkeypatch.setitem(sys.modules, "retopo_repaint", repaint)
    (tmp_path / "rig-python").write_text("")
    (tmp_path / "blender").write_text("")

    local = hero_route.needed_parts()
    colab = hero_route.needed_parts(colab=True)
    assert {name for name in local} - {name for name in colab} == {
        "Pixal3D (3D models)",
        "Hero face upscaler (Real-ESRGAN, 67 MB)",
        "Pixal3D's own clean-up (Setup: Rebuild Pixal3D)",
    }
    assert hero_route.missing_parts(colab) == []


def test_a_foreign_body_stages_with_no_artifacts_at_all(tmp_path):
    """A Hi3D body arrives as --body with no --body-views/--body-dump/--body-record: staging
    it must copy the mesh and skip the rest, not die on a None (the first live run did)."""
    body = tmp_path / "model.glb"
    body.write_bytes(b"glb")
    views = tmp_path / "model.svviews"
    views.mkdir()
    (views / "transforms.json").write_text("{}")
    dump = tmp_path / "model.post.bin"
    dump.write_bytes(b"dump")

    hero_route.stage_body_inputs(body, None, None, None, tmp_path / "body_raw.glb")
    raw = tmp_path / "body_raw.glb"
    assert raw.read_bytes() == b"glb"
    assert not raw.with_suffix(".svviews").exists() and not raw.with_suffix(".post.bin").exists()

    hero_route.stage_body_inputs(body, views, dump, None, raw)
    assert (raw.with_suffix(".svviews") / "transforms.json").is_file()
    assert raw.with_suffix(".post.bin").read_bytes() == b"dump"
