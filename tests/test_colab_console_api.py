"""The GPU VM console: Python on the rented machine, from the page.

The console is the escape hatch for questions without buttons — disk room, a mounted
Drive, a quick look at anything — so it must behave like the job legs it stands beside:
the code rides one bounded ``colab exec`` on the lab's shared session, every output line
streams to a log the page polls, a failure is said in the log and nowhere sunk, and the
status probe answers with what it knows (an unreachable VM is an answer, not an error).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))

import colab_console_api as cc


class FakeColab:
    """Stands in for image_to_3dlab.colab_backend: records the call, answers on cue."""

    SHARED_SESSION = "i2l-lab"
    calls: list = []
    fail_with: str | None = None
    answer: str = "hello\nworld\n"

    class ColabError(Exception):
        pass

    @classmethod
    def run(cls, argv, *, timeout=180, stdin=None, attempts=1,
            silence_seconds=None, on_output=None):
        cls.calls.append({"argv": list(argv), "stdin": stdin, "timeout": timeout})
        if cls.fail_with:
            raise cls.ColabError(cls.fail_with)
        for line in cls.answer.splitlines():
            if on_output:
                on_output(line)
        return cls.answer


@pytest.fixture()
def console(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "colab", FakeColab)
    monkeypatch.setattr(cc, "CONSOLE_ROOT", tmp_path / "console")
    FakeColab.calls.clear()
    FakeColab.fail_with = None
    FakeColab.answer = "hello\nworld\n"
    return tmp_path / "console"


def _wait_done(job_id: str) -> dict:
    for _ in range(100):                          # the run happens on a thread
        answer = cc.read(job_id)
        if answer["state"] == "done":
            return answer
        time.sleep(0.05)
    raise AssertionError("run never finished")


def test_a_run_takes_the_code_on_the_shared_session_and_streams_its_output(console):
    job = cc.start("print('hi')", 30)
    assert job["state"] == "running" and job["kind"] == "exec"
    _wait_done(job["id"])                          # the call happens on a thread

    call = FakeColab.calls[0]
    assert call["argv"] == ["exec", "-s", "i2l-lab", "--timeout", "30"]
    assert call["stdin"] == "print('hi')"          # the code rides stdin, like the legs

    answer = _wait_done(job["id"])
    assert answer["exit"] == 0 and "hello" in answer["output"]
    assert "world" in answer["output"]             # every line streamed, in order
    assert answer["seconds"] >= 0


def test_nothing_to_run_and_too_much_to_run_are_refused(console):
    with pytest.raises(ValueError):
        cc.start("   ", 30)
    with pytest.raises(ValueError):
        cc.start("x = 1\n" * 40000, 30)            # past 64 KB: a script's job, not a console's
    assert not FakeColab.calls                     # and nothing was sent anywhere


def test_the_timeout_is_clamped_to_what_the_channel_tolerates(console):
    first, second = cc.start("print(1)", 1), cc.start("print(2)", 999999)
    _wait_done(first["id"]), _wait_done(second["id"])
    timeouts = [c["argv"][-1] for c in FakeColab.calls]
    assert timeouts == ["5", "1800"]


def test_a_mount_run_drives_the_cli_under_a_terminal_we_hold(console, monkeypatch):
    """The mount is not a plain call: its "Press Enter" reads /dev/tty, so it runs under
    a pty this server owns and the page's continue button types the Enter into it."""
    spawned = []
    monkeypatch.setattr(cc, "_spawn_mount",
                        lambda job_dir: spawned.append(job_dir) or
                        (job_dir / "status.json").write_text(
                            json.dumps({"state": "done", "exit": 0, "kind": "drivemount"})))
    job = cc.start(None, 30, kind="drivemount")
    assert job["kind"] == "drivemount" and not FakeColab.calls   # not through colab.run
    assert spawned and spawned[0].name == job["id"]
    assert cc.read(job["id"])["exit"] == 0


def test_the_continue_button_types_the_enter_the_mount_waits_for(console):
    read_end, write_end = os.pipe()
    cc._PTYS["0" * 31 + "1"] = write_end
    try:
        assert cc.press_enter("0" * 31 + "1") is True
        assert os.read(read_end, 1) == b"\n"          # the Enter, in the terminal
        assert cc.press_enter("0" * 31 + "2") is False  # no mount waiting there
        with pytest.raises(ValueError):
            cc.press_enter("not-an-id")               # not a run id: no path games
    finally:
        os.close(read_end), os.close(write_end)
        cc._PTYS.clear()


def test_a_failed_run_says_so_in_the_log_and_nowhere_sinks(console):
    FakeColab.fail_with = "session 'i2l-lab' not found"
    job = cc.start("print('hi')", 30)
    answer = _wait_done(job["id"])
    assert answer["exit"] == 1
    assert "not found" in answer["output"]


def test_a_run_that_never_existed_is_not_readable(console):
    with pytest.raises(ValueError):
        cc.read("0" * 32)
    with pytest.raises(ValueError):
        cc.read("../secrets")                      # not a run id: no path games


def test_vm_status_parses_the_probe_and_an_unreachable_vm_is_an_answer(console):
    FakeColab.answer = 'noise\nVMSTATUS {"free_gb": 97, "total_gb": 253, "drive": {"mount": "/content/drive", "free_gb": 4100, "total_gb": 5000}}\n'
    status = cc.vm_status()
    assert status["live"] is True and status["free_gb"] == 97
    assert status["drive"]["mount"] == "/content/drive"

    FakeColab.answer = "VMSTATUS {\"free_gb\": 97, \"total_gb\": 253}\n"
    assert cc.vm_status()["drive"] is None         # nothing mounted is said, not guessed

    FakeColab.fail_with = "session 'i2l-lab' not found"
    asleep = cc.vm_status()
    assert asleep["live"] is False and "not found" in asleep["note"]
