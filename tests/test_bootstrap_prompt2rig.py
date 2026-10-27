"""The prompt-to-motion installer: pins its mirrors, stages H3's disk, never ambushes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bootstrap_prompt2rig_colab as bp

from image_to_3dlab import host as host_module  # noqa: F401  (pins the index constant)


def test_announcement_names_backend_licence_and_size():
    text = bp.announcement()
    assert "MiniMax H3" in text and "GVHMR" in text and "WiLoR" in text
    assert "int8" in text
    assert "Community License" in text          # the licence with a regional caveat
    assert "research" in text.lower()
    assert "GB" in text


def test_every_pin_is_a_full_checksum_or_revision():
    for name, value in (("H3_REVISION", bp.H3_REVISION), ("SMPLX_REVISION", bp.SMPLX_REVISION),
                        ("WILOR_REVISION", bp.WILOR_REVISION),
                        ("WILOR_MINI_COMMIT", bp.WILOR_MINI_COMMIT)):
        assert len(getattr(bp, name)) == 40 and all(c in "0123456789abcdef" for c in value)
    assert len(bp.SMPLX_SHA256) == 64
    for name, sha in bp.WILOR_SHA256.items():
        assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), name


def test_the_smplx_mirror_serves_gvhmr_layout():
    """GVHMR looks the file up by its own tree; the mirror must carry it under that path."""
    assert bp.SMPLX_FILE.endswith("models_smplx_v1_1/smplx/SMPLX_NEUTRAL.npz")


def test_h3s_download_skips_the_parts_this_route_never_reads():
    """The original-format copies (FL2VA/, Ref2VA/) and the unused ref2va partition are
    ~124 GB the route would download and throw away."""
    everything = " ".join(bp.H3_SMALL) + " " + " ".join(f"{p}/*" for p in bp.H3_BIG)
    for never in ("FL2VA", "Ref2VA", "transformer_ref"):
        assert never not in everything
    assert set(bp.H3_BIG) == {"transformer", "text_encoder"}


def test_the_quantize_step_is_the_model_cards_recipe():
    script = bp.quantize_script("transformer")
    assert "Int8WeightOnlyConfig(version=2)" in script
    assert "modules_to_not_convert" in script and "proj_out" in script
    assert "save_pretrained" in script
    encoder = bp.quantize_script("text_encoder")
    assert "Qwen3VLForConditionalGeneration.from_pretrained" in encoder
    assert "MiniMaxH3Transformer3DModel.from_pretrained" not in encoder


def test_a_warm_vm_installs_nothing_twice(monkeypatch, tmp_path):
    """Every stage is marker-gated; a session that already paid for it skips straight
    through without touching the network."""

    monkeypatch.setattr(bp, "DRIVER", tmp_path / "driver")
    monkeypatch.setattr(bp, "H3_DIR", tmp_path / "h3")
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise AssertionError("a warm VM must not download again")

    monkeypatch.setattr(bp, "must", fail)
    for part in bp.H3_BIG:
        bp.marker(f"h3-{part}").parent.mkdir(parents=True, exist_ok=True)
        bp.marker(f"h3-{part}").write_text("ok")
    bp.marker("autorig").write_text("ok")
    (tmp_path / "h3").mkdir(parents=True, exist_ok=True)
    (tmp_path / "h3" / "modular_model_index.json").write_text("{}")
    bp.ensure_h3()
    bp.ensure_autorig()
    assert not calls


def test_checksum_verification_rejects_and_removes_a_bad_file(tmp_path):
    target = tmp_path / "file.bin"
    target.write_bytes(b"not the pinned bytes")
    with pytest.raises(SystemExit, match="sha256"):
        bp.verify_sha256(target, "0" * 64)
    assert not target.exists()  # a failed file must not linger as a warm-session shortcut


def test_a_checksum_pass_that_already_landed_is_not_redownloaded(tmp_path, monkeypatch):
    target = tmp_path / "SMPLX_NEUTRAL.npz"
    target.write_bytes(b"good")
    monkeypatch.setattr(bp, "must",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("download")))
    assert bp.hf_download("repo", "any/path", target, "rev", "0" * 64) is not None \
        or target.is_file()


def test_the_venv_pins_its_versions():
    assert bp.GVHMR_VERSION == "1.6.1"
    assert bp.DIFFUSERS_VERSION.count(".") >= 1  # a real release, not a branch
    assert "WiLoR-mini@" in __import__("inspect").getsource(bp.ensure_venv)
    # diffusers 0.42 lives in the hub>=1.32 era; only transformers 5 can share that venv
    # (every 4.57.x needs hub<1.0 — found on the first live install).
    assert '"transformers>=5,<6"' in __import__("inspect").getsource(bp.ensure_venv)


def test_chumpys_legacy_setup_gets_the_pip_it_imports():
    """chumpy — pulled from git by the SMPL stack — imports pip inside its setup.py,
    and uv's isolated build environment carries no pip: the whole install died on the
    first live run (reproduced and fixed against a local uv of the pinned version).
    The venv directory carries a uv.toml that adds one, and the installs run from
    there because that is the directory uv reads its config from."""
    import inspect

    source = inspect.getsource(bp.ensure_venv)
    assert '"[extra-build-dependencies]\\nchumpy = [\\"pip\\"]\\n"' in source
    assert "VENDOR.mkdir(parents=True, exist_ok=True)" in source   # the dir is born here
    assert source.count("cwd=VENDOR") == 4       # torch, gvhmr, WiLoR and roma installs
    assert len(bp.UV_VERSION.split(".")) == 3    # pinned, so the config keeps meaning this
    assert bp.HF_HUB_VERSION.count(".") == 2     # the driver's one package, pinned too
    # WiLoR-mini's own pins (ultralytics==8.1.34, torch<=2.5) cannot sit next to gvhmr's
    # preproc extra or the Blackwell torch — its code arrives without them.
    assert '"--no-deps"' in source and '"roma"' in source


def test_torch_comes_from_the_pins_own_index_before_the_rest(monkeypatch):
    """PyPI's default Linux torch is a cu126 build; a Blackwell G4 needs cu128, and
    gvhmr wants <2.8. So torch installs first, alone, and from the cu128 index — the
    pin's newest CUDA build — never the driver-suggested one (a CUDA 13 driver maps to
    cu130, an index torch 2.7.1 was never published to: found on the first live run),
    and only then the packages that must not re-resolve it."""
    import inspect

    source = inspect.getsource(bp.ensure_venv)
    assert "driver_cuda_version" in source
    assert "torch_cuda_index" not in source         # the driver's newest build is the wrong ask
    assert 'host.TORCH_INDEX + "cu128"' in source   # the pin's own newest CUDA build
    # diffusers 0.42's loaders import ScalingType, which torch first carries at 2.10 —
    # probed on the real wheels (2.9.1 lacks it), found on the first live quantize.
    assert bp.TORCH_VERSION == "2.10.0"
    assert bp.TORCHVISION_VERSION == "0.25.0"
    installs = [line for line in source.splitlines() if '"uv", "pip", "install"' in line]
    assert installs, "the venv must actually install something"
