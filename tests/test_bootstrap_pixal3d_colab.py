"""Tests for the Pixal3D Colab installer's decisions, without renting a GPU.

The installer's job on the VM is narrow but wrong-answer-expensive: the venv must carry
everything the wrapper and `bootstrap_pixal3d.py` import (the latter runs
`snapshot_download` inside that venv), while carrying none of the CUDA wheels a GGML
runtime cannot use. Those two halves are checked directly rather than by running it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, str(REPO / "scripts")):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import bootstrap_pixal3d_colab as boot


def test_the_announcement_names_the_backend_the_route_the_size_and_the_licence():
    """AGENTS.md: name the backend, name the route, state the size, require an answer."""
    text = boot.announcement()
    assert "Pixal3D" in text
    assert "Colab" in text or "VM" in text
    assert "GB" in text
    # The encoder's licence is not advice; it is a restriction on shipping the output.
    assert "DINOv3" in text


def test_the_venv_interpreter_path_is_not_doubled():
    """`host.venv_python` appends `.venv/bin/python` to the project directory, but this
    module's VENV already ends in `.venv`; the naive spelling gave `.venv/.venv/bin/python`
    and the installer gave up saying no interpreter existed (the Hi3D installer's bug)."""
    python = boot.venv_python()
    assert ".venv/.venv" not in str(python), python
    assert python.parent.parent == boot.VENV


def test_the_venv_lock_keeps_the_wrapper_s_dependencies_and_heros_upscaler():
    """The venv runs `pixal3d_generate.py`, `bootstrap_pixal3d.py` and Hero's upscale.

    torch is dead weight to a Pixal3D generation (GGML, not PyTorch), but the Hero face
    route upscales its head crop on this same VM through `image_to_3dlab/upscale.py`, so
    torch must stay or Hero breaks on exactly the machine it was routed to. The packages
    the wrapper imports must all be there too, or the first job dies at `import rembg`
    minutes into a paid session.
    """
    names = {line.split("==")[0].strip().split(";")[0].strip()
             for line in (REPO / "requirements.lock").read_text().splitlines()
             if "==" in line and not line.startswith("#")}
    for wanted in ("pillow", "numpy", "rembg", "onnxruntime", "trimesh", "huggingface-hub",
                   "torch"):
        assert wanted in names, f"{wanted} was stripped from the Colab venv"
    # And the lock's CUDA wheels ride along with torch, so the upscale has a GPU to use.
    assert any(name.startswith("nvidia-") for name in names)


def test_it_refuses_to_download_without_yes_when_nobody_can_answer(monkeypatch, capsys):
    """`--yes` is the studio's dialog, or an agent's explicit choice; a default is not
    allowed, and a non-interactive run must refuse rather than assume."""
    monkeypatch.setattr("sys.stdin", None)
    assert boot.main([]) == 1
    assert "without --yes" in capsys.readouterr().out


def test_it_writes_its_stamp_only_after_both_halves(monkeypatch, tmp_path):
    """An install that died after the venv must not look finished: the marker gate in
    `colab_generate.py` would skip the setup and hand the job a half-built backend."""
    done = []

    def fake_must(command, **_):
        done.append(command[0] if isinstance(command, list) else command)
        if len(done) == 3:  # uv venv, uv pip install, then the delegate -- die inside it
            raise SystemExit("boom")

    monkeypatch.setattr(boot, "must", fake_must)
    monkeypatch.setattr(boot, "VENDOR", tmp_path)
    monkeypatch.setattr(boot, "install_python_packages", lambda: None)
    monkeypatch.setattr(boot, "install_trellis_and_weights",
                        lambda: (_ for _ in ()).throw(SystemExit("weights failed")))
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    try:
        boot.main(["--yes"])
    except SystemExit as exit_:
        assert exit_.code == "weights failed"
    assert not boot.BUILT_MARKER.is_file()
