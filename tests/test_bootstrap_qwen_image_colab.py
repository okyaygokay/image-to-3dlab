"""The Qwen-Image Colab bootstrap: what it pins, and what it says before it downloads.

A run on the VM costs money, so everything checked here is the cheap part: the compile
steps name the same commit the local prebuilt route pins, CUDA is on, the clone is
recursive (ggml is a submodule), and the announcement names the backend, the route and
the size before asking (AGENTS.md)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, str(REPO / "scripts")):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import bootstrap_qwen_image_colab as boot


def test_the_announcement_names_the_backend_the_route_the_size_and_the_licence():
    text = boot.announcement()
    for wanted in ("Qwen-Image", "Colab VM", "GB", "stable-diffusion.cpp", "licence"):
        assert wanted in text, f"the announcement does not name {wanted}"


def test_the_build_pins_the_same_commit_the_prebuilt_routes_use():
    """`master-975-95b3090` is the release the local bootstrap downloads a binary from;
    compiling a different commit would make the two routes different programs."""
    clone, configure, build = boot.build_commands("nvcc")
    assert "--branch" in clone and clone[clone.index("--branch") + 1] == boot.PINNED_REF
    assert "--recursive" in clone  # ggml is a submodule: a plain clone configures, then dies
    assert "-DGGML_CUDA=ON" in configure
    assert build[:2] == ["cmake", "--build"]


def test_the_binary_lands_where_the_wrapper_looks_for_it():
    assert boot.BINARY.name == "sd-cli"
    assert boot.BINARY.parent.name == "sdcpp"
