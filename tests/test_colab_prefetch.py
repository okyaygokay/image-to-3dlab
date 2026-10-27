"""The VM-side prefetch runner's rules: what it skips, what it waits for, what it records.

Every rule here guards a real failure mode on a paid VM: a duplicated prefetch fights
itself for the disk, a thin-disk install fails mid-download, and a heavy install that
does not wait for the running job steals its GPU (H3's components are quantized as they
download, which is GPU work).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, str(REPO / "scripts")):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import colab_prefetch as cp


class _Finished:
    """What subprocess.run returns, trimmed to what the runner reads."""

    def __init__(self, returncode=0, out="installed"):
        self.returncode = returncode
        self.stdout = out
        self.stderr = ""


def _record(ran):
    return lambda argv, **kwargs: ran.append(argv) or _Finished()


def test_a_stamped_backend_is_left_alone(tmp_path, monkeypatch):
    (tmp_path / ".setup-autorig-stamp").write_text("")
    ran = []
    monkeypatch.setattr(cp.subprocess, "run", _record(ran))
    monkeypatch.setattr(cp, "free_gb", lambda root: 200.0)
    done = cp.install([["autorig", "scripts/bootstrap_autorig_colab.py", 15]], tmp_path)
    assert done == [] and ran == []


def test_a_thin_disk_skips_the_install(tmp_path, monkeypatch):
    """The estimate plus headroom does not fit: skipping costs the next job an install,
    filling the disk mid-download costs the install itself."""
    monkeypatch.setattr(cp, "free_gb", lambda root: 20.0)    # 12 GB + 10 headroom > 20
    ran = []
    monkeypatch.setattr(cp.subprocess, "run", _record(ran))
    done = cp.install([["pixal3d", "scripts/bootstrap_pixal3d_colab.py", 12]], tmp_path)
    assert done == [] and ran == []


def test_a_light_install_does_not_wait_for_the_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "free_gb", lambda root: 200.0)
    monkeypatch.setattr(cp, "job_running", lambda root: True)
    waited = []
    monkeypatch.setattr(cp.subprocess, "run", _record([]))
    done = cp.install([["pixal3d", "scripts/bootstrap_pixal3d_colab.py", 12]], tmp_path,
                      sleep=waited.append)
    assert done == [("pixal3d", 0)] and waited == []


def test_a_heavy_install_waits_for_the_running_job(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "free_gb", lambda root: 200.0)
    busy = [True]

    def waking_sleep(seconds):
        busy[0] = False                  # the job finishes during the first wait

    monkeypatch.setattr(cp, "job_running", lambda root: busy[0])
    monkeypatch.setattr(cp.subprocess, "run", _record([]))
    done = cp.install([["hi3d", "scripts/bootstrap_hi3d_colab.py", 35]], tmp_path,
                      sleep=waking_sleep)
    assert done == [("hi3d", 0)]         # installed, and only after the job let go


def test_a_failed_install_is_recorded_not_hidden(tmp_path, monkeypatch):
    monkeypatch.setattr(cp, "free_gb", lambda root: 200.0)
    monkeypatch.setattr(cp.subprocess, "run",
                        lambda argv, **kwargs: _Finished(returncode=1, out="boom"))
    done = cp.install([["pixal3d", "scripts/bootstrap_pixal3d_colab.py", 12]], tmp_path)
    assert done == [("pixal3d", 1)]


def test_a_live_prefetch_is_not_duplicated(tmp_path, monkeypatch):
    (tmp_path / ".prefetch.pid").write_text(str(os.getpid()))    # this process is alive

    def must_not_install(*args, **kwargs):
        raise AssertionError("a second prefetch must not start")

    monkeypatch.setattr(cp, "install", must_not_install)
    assert cp.main(["--plan", "[]", "--root", str(tmp_path)]) == 0


def test_a_dead_pid_file_is_replaced_and_cleared(tmp_path, monkeypatch):
    (tmp_path / ".prefetch.pid").write_text("1")
    monkeypatch.setattr(cp, "pid_alive", lambda pid: False)
    ran = []
    monkeypatch.setattr(cp, "install", lambda plan, root: ran.append((plan, root)))
    plan = [["pixal3d", "scripts/bootstrap_pixal3d_colab.py", 12]]
    assert cp.main(["--plan", str(plan).replace("'", '"'), "--root", str(tmp_path)]) == 0
    assert not (tmp_path / ".prefetch.pid").exists()
    assert ran == [((plan), tmp_path)]
