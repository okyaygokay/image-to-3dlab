"""Tests for the Colab bridge: the decisions that cost money if they are wrong.

Nothing here talks to Colab. Every test is about a rule that, if it regressed, would either
leave a rented GPU running or refuse a session someone is actively using.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, str(REPO / "scripts")):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from image_to_3dlab import colab_backend as cb

# --- session naming ------------------------------------------------------------------------


def test_session_names_are_readable_and_unique():
    """`colab sessions` is the only view a user has; random hex is how sessions get left on."""
    name = cb.session_name("My Cool Prop.png", "hi3d")
    assert name.startswith("my-cool-prop-hi3d")
    assert " " not in name and "." not in name


def test_two_runs_of_the_same_image_get_different_names():
    first = cb.session_name("prop", "hi3d")
    second = cb.session_name("prop", "hi3d")
    assert first != second


def test_an_empty_stem_still_yields_a_usable_name():
    # No stem still has to produce something recognisable, not "-hi3d-abc123".
    name = cb.session_name("", "hi3d")
    assert name.startswith("hi3d-"), name
    assert cb.session_name("!!!", "hi3d").startswith("hi3d-")


def test_a_very_long_stem_is_bounded():
    # Colab names have a limit and a truncated-but-readable name beats a rejected one.
    name = cb.session_name("x" * 200, "hi3d")
    assert len(name) <= 60


# --- the idle rule -----------------------------------------------------------------------


def test_a_session_with_no_heartbeat_is_never_called_idle():
    """Unknown is not stale. This is the one that would kill a live run."""
    assert cb.is_idle(None) is False


def test_a_fresh_heartbeat_is_not_idle():
    now = 1_000_000.0
    assert cb.is_idle(now - 60, now=now) is False


def test_the_three_hour_boundary_is_the_boundary():
    now = 1_000_000.0
    limit = cb.IDLE_STOP_SECONDS
    assert cb.is_idle(now - limit + 1, now=now) is False
    assert cb.is_idle(now - limit, now=now) is True
    assert cb.is_idle(now - limit - 1, now=now) is True


def test_the_idle_window_is_three_hours():
    # Stated as a test because it is a billing decision, not an implementation detail.
    assert cb.IDLE_STOP_SECONDS == 3 * 60 * 60


def test_idle_sessions_are_returned_oldest_first(tmp_path, monkeypatch):
    path = tmp_path / "colab.json"
    monkeypatch.setenv(cb.HEARTBEAT_ENV, str(path))
    now = 2_000_000.0
    cb.touch("newer", path=path, now=now - cb.IDLE_STOP_SECONDS - 10)
    cb.touch("older", path=path, now=now - cb.IDLE_STOP_SECONDS - 500)
    cb.touch("live", path=path, now=now)
    assert cb.idle_sessions(path=path, now=now) == ["older", "newer"]


# --- heartbeat storage -------------------------------------------------------------------


def test_heartbeats_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "colab.json"
    monkeypatch.setenv(cb.HEARTBEAT_ENV, str(path))
    cb.touch("s1", now=123.0)
    cb.touch("s2", now=456.0)
    assert cb.read_heartbeats() == {"s1": 123.0, "s2": 456.0}
    # Nested directories are created: a first run must not fail on a missing ~/.i2l.
    assert path.is_file()


def test_forgetting_a_session_removes_only_it(tmp_path):
    path = tmp_path / "colab.json"
    cb.touch("keep", path=path, now=1.0)
    cb.touch("drop", path=path, now=2.0)
    cb.forget("drop", path=path)
    assert cb.read_heartbeats(path) == {"keep": 1.0}


def test_a_corrupt_state_file_is_treated_as_empty(tmp_path):
    # A half-written file must not make the watchdog throw and stop watching.
    path = tmp_path / "colab.json"
    path.write_text("{not json")
    assert cb.read_heartbeats(path) == {}


def test_a_non_numeric_heartbeat_is_ignored(tmp_path):
    path = tmp_path / "colab.json"
    path.write_text(json.dumps({"good": 10.0, "bad": "yes"}))
    assert cb.read_heartbeats(path) == {"good": 10.0}


# --- config validation -------------------------------------------------------------------


@pytest.mark.parametrize("accelerator", cb.ACCELERATORS)
def test_every_offered_accelerator_is_accepted(accelerator):
    assert cb.RemoteConfig("s", accelerator).validated().accelerator == accelerator


def test_an_unknown_accelerator_is_refused_before_anything_is_provisioned():
    with pytest.raises(ValueError, match="accelerator must be one of"):
        cb.RemoteConfig("s", "V100").validated()


def test_the_default_accelerator_is_the_verified_one():
    # G4 is the only card this lab has measured Hi3D and Hunyuan on.
    assert cb.DEFAULT_ACCELERATOR == "G4"
    assert cb.RemoteConfig("s").accelerator == "G4"


def test_a_session_needs_a_name_and_a_positive_timeout():
    with pytest.raises(ValueError):
        cb.RemoteConfig("   ").validated()
    with pytest.raises(ValueError):
        cb.RemoteConfig("s", timeout_seconds=0).validated()


# --- the announcement --------------------------------------------------------------------


def test_the_announcement_names_the_accelerator_the_session_and_the_bill():
    """AGENTS.md's confirmation rule, applied to someone else's machine and your money."""
    text = cb.announcement(cb.RemoteConfig("prop-1", "G4"), backend_label="Hi3D",
                           weights_gb=14.7)
    assert "G4" in text
    assert "prop-1" in text
    assert "14.7 GB" in text
    assert "Colab compute units" in text
    # The idle promise is part of what is being agreed to.
    assert "3 hours" in text


def test_the_announcement_says_when_the_vm_will_be_kept_running():
    kept = cb.announcement(cb.RemoteConfig("s", stop_when_done=False), backend_label="X",
                           weights_gb=None)
    assert "kept running" in kept
    dropped = cb.announcement(cb.RemoteConfig("s"), backend_label="X", weights_gb=None)
    assert "kept running" not in dropped


def test_durations_read_as_words():
    assert cb.human_duration(3 * 3600) == "3 hours"
    assert cb.human_duration(45 * 60) == "45 minutes"
    assert cb.human_duration(3600) == "1 hour"


# --- the CLI wrapper's bound ---------------------------------------------------------------


def test_a_missing_colab_cli_says_how_to_get_it(monkeypatch):
    monkeypatch.setattr(cb.shutil, "which", lambda name: None)
    with pytest.raises(cb.ColabError, match="google-colab-cli"):
        cb.colab_bin()


def test_a_colab_call_that_never_answers_raises_rather_than_hanging(monkeypatch):
    """An unbounded `colab exec` on a starting VM would freeze the whole job queue."""
    import subprocess

    def boom(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="colab", timeout=1)

    monkeypatch.setattr(cb.shutil, "which", lambda name: "/usr/bin/colab")
    monkeypatch.setattr(cb.subprocess, "run", boom)
    with pytest.raises(cb.ColabError, match="did not answer"):
        cb.run(["exec", "-s", "s"], timeout=1)


def test_a_failing_colab_call_carries_its_output(monkeypatch):
    class Result:
        returncode = 1
        stdout = ""
        stderr = "no such session: gone"

    monkeypatch.setattr(cb.shutil, "which", lambda name: "/usr/bin/colab")
    monkeypatch.setattr(cb.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(cb.ColabError, match="no such session"):
        cb.run(["exec", "-s", "gone"])


def test_a_silent_exec_is_retried_before_it_is_a_failure(monkeypatch):
    """A VM still bringing its kernel up does not answer at all. Found on a live run: the
    first attempt to exec during startup was reported as a dead session."""
    import subprocess

    calls = {"n": 0}

    class Ok:
        returncode = 0
        stdout = "fine"
        stderr = ""

    def flaky(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd="colab", timeout=1)
        return Ok()

    monkeypatch.setattr(cb.shutil, "which", lambda name: "/usr/bin/colab")
    monkeypatch.setattr(cb.subprocess, "run", flaky)
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)
    assert cb.run(["exec", "-s", "s"], timeout=1, attempts=3) == "fine"
    assert calls["n"] == 2


def test_retries_are_bounded(monkeypatch):
    import subprocess

    def always(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="colab", timeout=1)

    monkeypatch.setattr(cb.shutil, "which", lambda name: "/usr/bin/colab")
    monkeypatch.setattr(cb.subprocess, "run", always)
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)
    with pytest.raises(cb.ColabError, match="did not answer"):
        cb.run(["exec", "-s", "s"], timeout=1, attempts=3)


def test_a_warm_session_is_reused_instead_of_paying_the_install_again(tmp_path):
    """The GUI starts a job per click. A fresh VM installs for 15-30 minutes first, so
    every click paying that is not a usable interface -- a session still inside its idle
    window already has the venv and the weights and must be picked up."""
    path = tmp_path / "colab.json"
    now = 1_000_000.0
    cb.touch("prop-hi3d-aaaaaa", path=path, now=now - 60)
    cb.touch("prop-hunyuan_cuda-bbbbbb", path=path, now=now - 60)
    live = {"prop-hi3d-aaaaaa", "prop-hunyuan_cuda-bbbbbb"}

    assert cb.warm_session("hi3d", path=path, now=now, live=live) == "prop-hi3d-aaaaaa"
    assert cb.warm_session("hunyuan_cuda", path=path, now=now,
                          live=live) == "prop-hunyuan_cuda-bbbbbb"
    # A route with no warm session gets None, not someone else's venv.
    assert cb.warm_session("trellis_cuda", path=path, now=now, live=live) is None


def test_a_session_past_the_idle_limit_is_not_reused(tmp_path):
    """Reuse must not outlive the idle rule: the watchdog would stop it mid-job."""
    path = tmp_path / "colab.json"
    now = 1_000_000.0
    cb.touch("prop-hi3d-aaaaaa", path=path, now=now - cb.IDLE_STOP_SECONDS - 60)
    assert cb.warm_session("hi3d", path=path, now=now,
                          live={"prop-hi3d-aaaaaa"}) is None


def test_a_session_the_cli_does_not_list_is_not_reused(tmp_path):
    """The heartbeat file outlives the VM. A stopped session must not be picked."""
    path = tmp_path / "colab.json"
    now = 1_000_000.0
    cb.touch("prop-hi3d-aaaaaa", path=path, now=now - 60)
    assert cb.warm_session("hi3d", path=path, now=now, live=set()) is None


def test_the_newest_warm_session_wins(tmp_path):
    """Two live sessions of one route: the more recently used one is likelier warm."""
    path = tmp_path / "colab.json"
    now = 1_000_000.0
    cb.touch("old-hi3d-aaaaaa", path=path, now=now - 600)
    cb.touch("new-hi3d-bbbbbb", path=path, now=now - 60)
    live = {"old-hi3d-aaaaaa", "new-hi3d-bbbbbb"}
    assert cb.warm_session("hi3d", path=path, now=now, live=live) == "new-hi3d-bbbbbb"
