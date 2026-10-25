"""Tests for the Hi3D installer's decisions, without a GPU.

The installer is the one place a wrong answer costs a multi-gigabyte download on a card
that cannot run the result, so the two questions it asks about the machine -- is this card
new enough, and does it need the newer torch -- are tested directly rather than by running
it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, str(REPO / "scripts")):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import bootstrap_hi3d as boot


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("120", (12, 0)),   # RTX PRO 6000 / RTX 50-series
        ("90", (9, 0)),     # H100, H20
        ("89", (8, 9)),     # RTX 4090
        ("86", (8, 6)),     # RTX 3090 / A10
        ("80", (8, 0)),     # A100
        ("75", (7, 5)),     # Turing
        (None, None),
        ("", None),
        ("x", None),
        ((8, 6), (8, 6)),
    ],
)
def test_compute_capability_parses_both_spellings(reported, expected):
    """nvidia-smi reports CMake style ("120"); 120 must be major 12, not major 1."""
    assert boot.parse_compute_capability(reported) == expected


@pytest.mark.parametrize("capability", ["120", "90", "89", "86", "80"])
def test_ampere_and_newer_get_the_pinned_cu128_build(capability):
    route, why = boot.choose_route(capability, (12, 4))
    assert route == "cu128"
    assert "compute capability" in why


@pytest.mark.parametrize("capability", ["75", "70", "61"])
def test_cards_older_than_ampere_are_refused_with_a_reason(capability):
    route, why = boot.choose_route(capability, (12, 4))
    assert route is None
    # The reason has to name the real cause; "something went wrong" sends people hunting.
    assert "compute capability" in why
    assert "--yes" not in why


def test_an_unreadable_card_is_refused_rather_than_guessed():
    # Unlike the runtime check in hi3d_backend (which tries an unknown card), an install
    # that cannot identify the card must not start downloading.
    route, why = boot.choose_route(None, (12, 4))
    assert route is None
    assert "compute" in why


def test_an_old_driver_is_refused_before_the_download():
    route, why = boot.choose_route("86", (11, 8))
    assert route is None
    assert "12.4" in why


def test_an_unknown_driver_does_not_block_an_ampere_card():
    # nvidia-smi can be present while its header is unparsable; the card is the stronger
    # signal, and torch reports a real error if the driver is genuinely too old.
    assert boot.choose_route("86", None)[0] == "cu128"


def test_the_announcement_names_the_route_the_size_and_the_licence():
    """AGENTS.md: name the backend, the route, the size, and require an affirmative answer."""
    text = boot.announcement()
    assert "Hi3D" in text
    assert "NVIDIA" in text
    assert "GB" in text
    # The licence line is the one that must survive editing: it is not advice, it is a
    # restriction with teeth for anyone shipping the output.
    assert "Powered by Hi3D 1.0" in text
    assert "EU" in text and "UK" in text and "South Korea" in text


def test_the_announcement_states_the_pinned_stack_not_upstreams():
    text = boot.announcement()
    assert "2.9.1+cu128" in text
    assert "flash-attn" in text
    # Upstream's cu124 stack is the thing that cannot run on the card this installs onto.
    assert "2.5.0" not in text


def test_the_weights_list_covers_every_model_a_run_loads():
    """A missing entry means a first generation fetches it unannounced."""
    sources = {source for _, source, _, _ in boot.WEIGHTS}
    assert "hi3d-official/Hi3Dv1.0" in sources
    assert "facebook/dinov2-large" in sources
    assert "openai/clip-vit-large-patch14" in sources
    # The coarse stage is easy to forget; without it the first run pulls ~4 GB silently.
    assert "tencent/Hunyuan3D-2mini" in sources


def test_the_pinned_versions_satisfy_what_hi3d_requires():
    """The installer and the package metadata must not disagree.

    Hi3D declares `numpy<2` and a transformers major it was measured on. Pinning numpy 2
    or transformers 5 in the installer would install cleanly and then fail at import, many
    minutes later, with a message about a missing symbol.
    """
    pins = dict(line.split("==") for line in boot.PINNED)
    assert pins["numpy"].startswith("1."), pins["numpy"]
    assert pins["transformers"].split(".")[0] == "4", pins["transformers"]
    # flash-attn's wheel is for cp310 specifically, so the interpreter is not free to drift.
    assert boot.PYTHON_VERSION.startswith("3.10"), boot.PYTHON_VERSION


def test_the_flash_attn_wheel_is_addressed_by_content_not_by_branch():
    """`main` can be replaced under us; a commit and a sha256 cannot."""
    assert "/resolve/" in boot.FLASH_ATTN_URL
    assert "/main/" not in boot.FLASH_ATTN_URL
    assert len(boot.FLASH_ATTN_SHA256) == 64
    assert "cp310" in boot.FLASH_ATTN_URL and "cu128" in boot.FLASH_ATTN_URL


def test_the_venv_interpreter_path_is_not_doubled():
    """Real bug, found on a live VM: `host.venv_python` appends `.venv/bin/python` to the
    project directory, but this module's VENV already ends in `.venv`, so the result was
    `.venv/.venv/bin/python` and the installer gave up saying no interpreter existed."""
    python = boot.venv_python()
    assert ".venv/.venv" not in str(python), python
    assert python.name in ("python", "python.exe")
    assert python.parent.name in ("bin", "Scripts")
    assert python.parent.parent == boot.VENV
