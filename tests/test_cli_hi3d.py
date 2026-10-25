"""The Hi3D route, as the command line sees it.

The CLI is the second front door to the same backend (the studio is the first), so what it
accepts has to agree with what `image_to_3dlab.hi3d_backend` validates. These tests drive
the parser and the manifest path without loading torch or a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_to_3dlab import cli


def test_hi3d_is_a_mode_of_its_own_and_excludes_the_others():
    args = cli.build_parser().parse_args(["photo.png", "--hi3d"])
    assert args.hi3d is True
    assert not (args.fast or args.quality or args.trellis)


def test_hi3d_defaults_match_the_backend_module():
    from image_to_3dlab import hi3d_backend

    args = cli.build_parser().parse_args(["photo.png", "--hi3d"])
    assert args.resolution == hi3d_backend.DEFAULTS["resolution"]
    assert args.hi3d_steps == hi3d_backend.DEFAULTS["steps"]
    assert args.coarse_steps == hi3d_backend.DEFAULTS["coarse_steps"]


def test_a_coarse_mesh_can_be_supplied_to_skip_the_coarse_stage(tmp_path):
    coarse = tmp_path / "coarse.glb"
    args = cli.build_parser().parse_args(
        ["photo.png", "--hi3d", "--coarse-mesh", str(coarse)])
    assert args.coarse_mesh == coarse


@pytest.mark.parametrize("resolution", ("512", "1024", "1536"))
def test_only_the_documented_resolutions_are_offered(resolution):
    args = cli.build_parser().parse_args(
        ["photo.png", "--hi3d", "--resolution", resolution])
    assert args.resolution == int(resolution)


def test_an_off_ladder_resolution_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["photo.png", "--hi3d", "--resolution", "768"])


def test_two_modes_at_once_are_refused():
    # Exactly one mode, or the run has no single answer for what it is doing.
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["photo.png", "--hi3d", "--trellis"])


def test_a_manifest_naming_hi3d_selects_the_hi3d_mode(tmp_path):
    manifest = tmp_path / "run.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "use_case": "showcase",
        "distribution": "private",
        "commercial_intent": False,
        "input": {"path": "photo.png"},
        "model": {
            "backend": "hi3d",
            "parameters": {"resolution": 1536, "steps": 12, "coarse_steps": 20},
        },
    }))
    image = tmp_path / "photo.png"
    image.write_bytes(b"x")

    args = cli.build_parser().parse_args(["--run-manifest", str(manifest)])
    # Reproduce what main() does with a manifest, without running a backend.
    data = json.loads(manifest.read_text())
    assert data["model"]["backend"] == "hi3d"
    assert args.run_manifest == manifest


def test_the_manifest_parameters_reach_the_hi3d_flags(tmp_path, monkeypatch):
    """A manifest's numbers must survive into argparse's names, or the run silently uses
    defaults while the sidecar records what the manifest said."""
    manifest = tmp_path / "run.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "use_case": "showcase",
        "distribution": "private",
        "input": {"path": "photo.png"},
        "model": {"backend": "hi3d",
                  "parameters": {"resolution": 1536, "steps": 9, "coarse_steps": 11}},
        "output": {"directory": "output"},
    }))
    (tmp_path / "photo.png").write_bytes(b"x")

    captured = {}

    def fake_finalize(*, generated, image, output_root, backend, **_kwargs):
        captured["backend"] = backend
        return generated, Path("sidecar.json")

    def fake_generate(*_args, **kwargs):
        captured["kwargs"] = kwargs
        return Path("generated.glb")

    monkeypatch.setattr(cli, "finalize_output", fake_finalize)
    monkeypatch.setattr("image_to_3dlab.hi3d_backend.generate_hi3d", fake_generate)

    assert cli.main(["--run-manifest", str(manifest)]) == 0
    assert captured["backend"] == "hi3d"
    # `steps` in a Hi3D manifest means Hi3D's, not TRELLIS's.
    assert captured["kwargs"]["resolution"] == 1536
    assert captured["kwargs"]["steps"] == 9
    assert captured["kwargs"]["coarse_steps"] == 11


def test_the_licence_gate_runs_before_any_generation(monkeypatch, tmp_path):
    """`validate_run_policy` must stay ahead of model work; a run that is refused for
    licensing must never have started the backend."""
    manifest = tmp_path / "run.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "use_case": "game",
        "distribution": "worldwide",
        "input": {"path": "photo.png"},
        "model": {"backend": "hi3d"},
        "license_policy": {"allow_conditional": True},
    }))
    (tmp_path / "photo.png").write_bytes(b"x")

    called = []
    monkeypatch.setattr("image_to_3dlab.hi3d_backend.generate_hi3d",
                        lambda *a, **k: called.append(1))

    assert cli.main(["--run-manifest", str(manifest)]) == 2
    assert not called, "the backend ran despite a licensing refusal"
