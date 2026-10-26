"""The two thin remote wrappers (Qwen-Image, Auto-rig): routing only, no models.

Each wrapper's whole job is to turn the supervisor's positional protocol into the local
route's own command builders, imported -- never copied -- so the hosted and local routes
cannot drift. These tests pin that: the same argv the local route would run, built from
the same functions, with the preset switch choosing rig over animate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, str(REPO / "scripts"), str(REPO / "viewer")):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import qwen_image_colab as qwen


def test_the_qwen_wrapper_runs_the_local_route_s_own_command(monkeypatch, tmp_path):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"prompt": "a knight",
                                   "settings": {"steps": 4, "seed": 3, "width": 832,
                                                "height": 1248}}))
    output = tmp_path / "out.png"
    ran: dict = {}

    # Patch through the wrapper's own reference: tests/test_image_api.py hand-loads its
    # own copy into sys.modules at collection time, so `import image_api` here can be a
    # different object from the one qwen_image_colab holds.
    weights = {"diffusion_model": tmp_path / "dit.gguf", "llm": tmp_path / "llm.gguf",
               "vae": tmp_path / "vae.safetensors"}
    monkeypatch.setattr(qwen.image_api, "resolve_weights", lambda: weights)

    def fake_run(command, **_kwargs):
        ran["argv"] = command
        output.write_bytes(b"PNG")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(qwen.subprocess, "run", fake_run)
    assert qwen.main([str(request), str(output)]) == 0
    # The argv is exactly what a local run would have built for the same request.
    assert ran["argv"] == qwen.image_api.build_command(
        "a knight",
        qwen.image_api.clean_settings({"steps": 4, "seed": 3,
                                       "width": 832, "height": 1248}),
        output, weights)


def test_the_qwen_wrapper_refuses_an_unreadable_request(tmp_path, capsys):
    bad = tmp_path / "request.json"
    bad.write_text("{not json")
    assert qwen.main([str(bad), str(tmp_path / "o.png")]) == 2
    assert "unreadable" in capsys.readouterr().err


def test_the_autorig_wrapper_rigs_without_a_preset_and_plays_one_with_it(monkeypatch, tmp_path):
    import autorig_api
    import autorig_colab

    source, result = tmp_path / "m.glb", tmp_path / "r.glb"
    source.write_bytes(b"glb")
    ran: list = []

    def fake_run(command, **_kwargs):
        ran.append(command)
        result.write_bytes(b"glb")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(autorig_colab.subprocess, "run", fake_run)

    assert autorig_colab.main([str(source), str(result)]) == 0
    assert ran == [part for part, _ in autorig_api.rig_commands(source, result)]

    result.unlink()
    preset = tmp_path / "anim.glb"
    motions = REPO / "image_to_3dlab" / "motions"
    if (motions / "walk.npz").is_file():  # only where the shipped clips exist
        def fake_run_anim(command, **_kwargs):
            ran.append(command)
            preset.write_bytes(b"glb")
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(autorig_colab.subprocess, "run", fake_run_anim)
        assert autorig_colab.main([str(source), str(preset), "--", "--preset", "walk",
                                   "--arm-spread", "12", "--speed", "1.5"]) == 0
        expected = autorig_api.animate_command(source, motions / "walk.npz", preset,
                                               12.0, 1.5)
        assert ran[-1] == expected


def test_the_autorig_wrapper_names_a_preset_the_vm_does_not_have(monkeypatch, tmp_path, capsys):
    import autorig_colab

    source = tmp_path / "m.glb"
    source.write_bytes(b"glb")
    monkeypatch.setattr(autorig_colab.subprocess, "run",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("ran")))
    code = autorig_colab.main([str(source), str(tmp_path / "r.glb"),
                               "--", "--preset", "no-such-clip"])
    assert code == 2 and "no such preset" in capsys.readouterr().err
