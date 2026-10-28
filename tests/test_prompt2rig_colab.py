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

    p2r.generate_clip("swinging a sword", 6, 0, tmp_path / "clip.mp4", keyframe,
                      engine="h3")
    assert 'workflow="fl2va"' in captured["script"].replace("'", '"')
    assert f"image=Image.open({str(keyframe)!r})" in captured["script"]
    assert ".convert('RGB')" in captured["script"]

    captured.clear()
    p2r.generate_clip("swinging a sword", 6, 0, tmp_path / "clip2.mp4", engine="h3")
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


def test_ltx2s_frame_grid_is_8k_plus_1():
    """LTX's video VAE compresses time 8x: a clip must be 8k+1 frames long, and the
    request is rounded up into that grid (the same deal as H3's 17n+5)."""
    import ltx2_video

    assert ltx2_video.snapped_frames(6.5) == 161      # 6.7 s — 6.5 s is off the grid
    assert ltx2_video.snapped_frames(2) == 49         # under the window: the shortest clip
    assert ltx2_video.snapped_frames(15) == 361       # the top of the window
    assert all(frames % 8 == 1 for frames in
               (ltx2_video.snapped_frames(s) for s in range(2, 16)))


def test_ltx2s_prompt_is_one_long_paragraph_with_the_same_framing():
    """LTX's card says it degrades on short prompts, so its prompt is one long
    audio-visual paragraph — carrying the same planted-in-place framing the H3 one
    does, or its actor would wander like H3's did."""
    prompt = p2r.ltx_motion_prompt("a knight swinging a sword.")
    assert prompt.startswith("A single continuous shot of one person a knight swinging "
                             "a sword")
    assert "both feet planted" in prompt and "no camera movement" in prompt
    assert prompt.count("\n") == 0 and "sound" in prompt


def test_the_video_engine_picks_who_acts_the_prompt_out(monkeypatch, tmp_path):
    """`--video-engine ltx2` swaps the video stage; every later stage is engine-agnostic
    because both engines land at 24 fps. H3 is the default — a preference call, even
    though LTX-2.5 won the A/B's clean-report metrics."""
    calls = {}

    def fake_h3(prompt, seconds, seed, out, keyframe):
        calls["h3"] = (prompt, seed)

    def fake_ltx2(prompt, seconds, seed, out, keyframe):
        calls["ltx2"] = (prompt, seed)

    monkeypatch.setattr(p2r, "generate_h3", fake_h3)
    monkeypatch.setattr(p2r, "generate_ltx2", fake_ltx2)
    p2r.generate_clip("swinging a sword", 6, 7, tmp_path / "a.mp4", None, "ltx2")
    p2r.generate_clip("swinging a sword", 6, 8, tmp_path / "b.mp4", None)
    assert list(calls) == ["ltx2", "h3"] and calls["ltx2"][1] == 7 and calls["h3"][1] == 8
    # an already-generated clip answers to neither engine
    (tmp_path / "have.mp4").write_bytes(b"clip")
    p2r.generate_clip("swinging a sword", 6, 9, tmp_path / "have.mp4", None, "ltx2")
    assert calls["ltx2"] == ("swinging a sword", 7)


def test_ltx2_installs_on_demand_once(monkeypatch, tmp_path):
    """The engine's ~71 GB is paid only when a run asks for it, and once per VM: the
    marker carries the bootstrap's build id, so both routes agree on what "installed"
    means — and a VM warm on an older build re-runs the install instead of trusting a
    marker that predates the change."""
    import bootstrap_prompt2rig_colab as bp

    monkeypatch.setattr(p2r, "REPO", tmp_path)
    driver = tmp_path / "vendor" / "prompt2rig-driver"
    driver.mkdir(parents=True)
    commands: list = []

    def record(command, *, what):
        commands.append([str(part) for part in command])

    monkeypatch.setattr(p2r, "run", record)
    p2r.ensure_ltx2()
    assert commands[0][1].endswith("bootstrap_prompt2rig_colab.py")
    assert commands[0][-2:] == ["--yes", "--ltx2-only"]
    (driver / ".ltx2-done").write_text(bp.LTX2_BUILD)
    commands.clear()
    p2r.ensure_ltx2()
    assert commands == []
    # a marker from an older build is not "installed" — the install runs again
    (driver / ".ltx2-done").write_text("ok")
    p2r.ensure_ltx2()
    assert len(commands) == 1


def test_the_video_engine_flag_parses():
    """The studio forwards the engine after `--`; H3 is the default (a preference
    call, even though LTX-2.5 won the A/B's clean-report metrics)."""
    argv = ["r.glb", "a.glb", "--prompt", "x"]
    assert p2r.parse(argv + ["--video-engine", "ltx2"]).video_engine == "ltx2"
    assert p2r.parse(argv).video_engine == "h3"


def test_every_stage_shares_one_landscape_canvas():
    """H3 stretches the rig keyframe onto its canvas and LTX resizes the keyframe to
    its own, so the keyframe render, the H3 canvas and the LTX canvas must agree on
    orientation. H3's tuple was once (960, 544) read height-first — a portrait frame
    that squashed every keyframed clip's actor to 57% width and 1.76× height, which
    motion recovery then read as the character's proportions (found live: "the
    generated video's scales are terrible")."""
    import hero_blender
    import ltx2_video

    assert p2r.CANVAS == (544, 960)                        # height, width — landscape
    assert hero_blender.still.__defaults__ == (960, 544)   # width, height
    assert ltx2_video.generate.__defaults__[-2:] == (960, 544)
    assert p2r.CANVAS[1] / p2r.CANVAS[0] == 960 / 544      # one aspect, three places
