"""The prompt-to-motion wrapper's contract: what the studio sends is what it parses."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prompt2rig_colab as p2r


def test_h3s_frame_grid_is_respected():
    """H3 decodes 17n+5 frames at 24 fps; anything else is snapped up into the grid."""
    assert p2r.snapped_frames(5) == 124      # 5.17 s — 5 s does not divide the grid
    assert p2r.snapped_frames(6) == 158      # 6.58 s
    assert p2r.snapped_frames(15) == 362     # 15.08 s, the top of the window
    assert p2r.snapped_frames(2) == 124      # under the window: the shortest clip
    assert p2r.snapped_frames(20) == 362     # over the window: the longest


def test_the_prompt_is_framed_for_motion_recovery():
    """GVHMR wants one full body, in frame, in place — so the user's words are wrapped in
    exactly that framing, and their motion stays the sentence's subject."""
    prompt = p2r.motion_prompt("a knight swinging a sword.")
    assert prompt.startswith("One person a knight swinging a sword")
    assert "static camera" in prompt and "full body" in prompt.lower()
    assert "sword" in prompt


def test_the_studios_forwarded_flags_parse(tmp_path):
    """The API builds `--prompt … --arm-spread … --speed … --seconds … --seed`; the
    wrapper must accept exactly that argv (this is the whole handshake)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
    import autorig_api

    command = autorig_api.describe_command(
        tmp_path / "knight_rigged.glb", tmp_path / "knight_swing.glb",
        "a knight swinging a sword", 12, 1.25, 6.5, 7)
    forwarded = command[command.index("--") + 1:]
    args = p2r.parse([str(command[3]), str(command[4])] + forwarded)
    assert args.prompt == "a knight swinging a sword"
    assert args.arm_spread == 12.0 and args.speed == 1.25
    assert args.seconds == 6.5 and args.seed == 7
    assert str(args.rigged).endswith("knight_rigged.glb")
    assert str(args.result).endswith("knight_swing.glb")


def test_a_prompt_is_required():
    try:
        p2r.main(["in.glb", "out.glb"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("main must refuse to run without --prompt")


def test_colabs_notebook_backend_is_overridden():
    """Colab's kernel exports MPLBACKEND=module://matplotlib_inline.backend_inline; this
    venv's matplotlib rejects that string at import, and GVHMR imports it through
    torchmetrics. Overridden outright — the bad value is already in the environment."""
    import os

    assert os.environ["MPLBACKEND"] == "Agg"


def test_the_keyframe_selects_the_workflow(monkeypatch, tmp_path):
    """fl2va is the workflow only when a keyframe rides along: a keyframe-less fl2va chain
    noises an empty condition list and dies (`torch.cat` of nothing — found on the first
    live run). With one, the rig's own render is the clip's first frame."""
    captured: dict = {}

    def fake_run(command, *, what):
        captured["script"] = command[2]
        tmp_path.joinpath("clip.mp4").write_bytes(b"")

    monkeypatch.setattr(p2r, "run", fake_run)
    keyframe = tmp_path / "keyframe.png"
    keyframe.write_bytes(b"png")

    p2r.generate_clip("swinging a sword", 6, 0, tmp_path / "clip.mp4", keyframe)
    assert 'workflow="fl2va"' in captured["script"].replace("'", '"')
    assert f"image=Image.open({str(keyframe)!r})" in captured["script"]
    assert ".convert('RGB')" in captured["script"]

    captured.clear()
    p2r.generate_clip("swinging a sword", 6, 0, tmp_path / "clip2.mp4")
    assert 'workflow="t2va"' in captured["script"].replace("'", '"')
    assert "image=" not in captured["script"]


def test_the_keyframe_renders_through_the_only_bpy_on_the_vm(monkeypatch, tmp_path):
    """The still is rendered by hero_blender in the SkinTokens venv — the one bpy this VM
    has — and a missing venv degrades to a text-only clip rather than a dead job."""
    (tmp_path / "vendor" / "SkinTokens" / ".venv" / "bin").mkdir(parents=True)
    venv_python = tmp_path / "vendor" / "SkinTokens" / ".venv" / "bin" / "python"
    venv_python.write_text("#!/bin/sh\n")
    monkeypatch.setattr(p2r, "REPO", tmp_path)
    commands: list = []

    def render(command, *, what):
        commands.append([str(part) for part in command])
        Path(command[-1]).write_bytes(b"png")

    monkeypatch.setattr(p2r, "run", render)
    assert p2r.render_keyframe(tmp_path / "rig.glb", tmp_path / "kf.png") is True
    assert commands[0][0] == str(venv_python)
    assert commands[0][1].endswith("hero_blender.py")
    assert commands[0][commands[0].index("--") + 1:] == ["still", str(tmp_path / "rig.glb"),
                                                         str(tmp_path / "kf.png")]

    venv_python.unlink()
    assert p2r.render_keyframe(tmp_path / "rig.glb", tmp_path / "other.png") is False
