"""Tests for the Colab bridge: the decisions that cost money if they are wrong.

Nothing here talks to Colab. Every test is about a rule that, if it regressed, would either
leave a rented GPU running or refuse a session someone is actively using.
"""

from __future__ import annotations

import json
import sys
import time
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


def test_a_network_blip_on_a_transfer_is_retried_not_fatal(monkeypatch):
    """A large upload dying to 'Connection reset by peer'/'Broken pipe' is a flaky byte,
    not a dead session -- but it exits non-zero, which the retry used to treat as final.
    Found on live hero runs: the run died at the same upload twice."""
    calls = {"n": 0}

    class Blip:
        returncode = 1
        stdout = ""
        stderr = "[colab] Upload failed: ('Connection aborted.', BrokenPipeError(32, 'Broken pipe'))"

    class Ok:
        returncode = 0
        stdout = "uploaded"
        stderr = ""

    def flaky(*_args, **_kwargs):
        calls["n"] += 1
        return Blip() if calls["n"] == 1 else Ok()

    monkeypatch.setattr(cb.shutil, "which", lambda name: "/usr/bin/colab")
    monkeypatch.setattr(cb.subprocess, "run", flaky)
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)
    assert cb.run(["upload", "big.bin", "r", "-s", "s"], timeout=1, attempts=3) == "uploaded"
    assert calls["n"] == 2


def test_a_download_dying_mid_stream_is_retried_not_fatal(monkeypatch):
    """A download that dies 2.2 MB short printed 'Connection broken: IncompleteRead(...)'
    — words the transient list did not know, so attempts=3 never fired and an
    80-minute hero run lost its head dump to one flaky read (found live, 2026-10-29:
    66,648,328 of 68,875,439 bytes came back fine)."""
    calls = {"n": 0}

    class Blip:
        returncode = 1
        stdout = ""
        stderr = ("[colab] Download failed: ('Connection broken: "
                  "IncompleteRead(66648328 bytes read, 2226911 more expected)', "
                  "IncompleteRead(66648328 bytes read, 2226911 more expected))")

    class Ok:
        returncode = 0
        stdout = "downloaded"
        stderr = ""

    def flaky(*_args, **_kwargs):
        calls["n"] += 1
        return Blip() if calls["n"] == 1 else Ok()

    monkeypatch.setattr(cb.shutil, "which", lambda name: "/usr/bin/colab")
    monkeypatch.setattr(cb.subprocess, "run", flaky)
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)
    assert cb.run(["download", "head.post.bin", "l", "-s", "s"], timeout=1,
                  attempts=3) == "downloaded"
    assert calls["n"] == 2


def test_a_command_that_ran_and_failed_is_not_rerun(monkeypatch):
    """Only the CLI's own transport words earn a retry. An exec whose code failed has
    already made its side effects; running it again would repeat them."""
    calls = {"n": 0}

    class Boom:
        returncode = 3
        stdout = "ValueError: bad arguments, the job was not started"
        stderr = ""

    def once(*_args, **_kwargs):
        calls["n"] += 1
        return Boom()

    monkeypatch.setattr(cb.shutil, "which", lambda name: "/usr/bin/colab")
    monkeypatch.setattr(cb.subprocess, "run", once)
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)
    with pytest.raises(cb.ColabError, match="failed"):
        cb.run(["exec", "-s", "s"], timeout=1, attempts=3)
    assert calls["n"] == 1


def test_a_file_over_the_apis_ceiling_rides_in_proven_sized_pieces(monkeypatch, tmp_path):
    """The contents API rejects an upload body somewhere between 60 and 100 MB (probed
    live: a 342 MB clean-up dump died three ways). Such a file is split under the
    ceiling, and one exec concatenates, drops the parts and checks the checksum."""
    monkeypatch.setattr(cb, "UPLOAD_CHUNK", 10)
    calls = []

    def record(args, **kwargs):
        calls.append((list(args), kwargs))
        return "reassembled 3 parts -> ok"

    monkeypatch.setattr(cb, "run", record)
    local = tmp_path / "dump.post.bin"
    payload = b"0123456789abcdefghij01234"          # 25 bytes -> three parts, the last short
    local.write_bytes(payload)
    cb.upload("s", local, "/content/i2l/outputs/dump.post.bin")

    uploads = [args for args, _ in calls if args[0] == "upload"]
    assert [Path(u[1]).name for u in uploads] == ["dump.post.bin.part0000",
                                                  "dump.post.bin.part0001",
                                                  "dump.post.bin.part0002"]
    assert all(u[2].startswith("/content/i2l/outputs/dump.post.bin.part") for u in uploads)
    (exec_args, exec_kwargs) = next((a, k) for a, k in calls if a[0] == "exec")
    assert "--timeout" in exec_args
    code = exec_kwargs["stdin"]
    assert "copyfileobj" in code and "checksum mismatch" in code
    import hashlib
    assert hashlib.md5(payload).hexdigest() in code


def test_a_small_file_uploads_whole(monkeypatch, tmp_path):
    monkeypatch.setattr(cb, "UPLOAD_CHUNK", 10)
    calls = []

    def record(args, **_kwargs):
        calls.append(list(args))
        return ""

    monkeypatch.setattr(cb, "run", record)
    local = tmp_path / "crop.png"
    local.write_bytes(b"tiny")
    cb.upload("s", local, "/content/i2l/inputs/crop.png")
    assert [c[0] for c in calls] == ["upload"]
    assert calls[0][2] == "/content/i2l/inputs/crop.png"


def _vm_holding(files: dict, chunk: int, calls: list):
    """A `run` fake whose VM holds `files` (remote path -> bytes): probes are answered
    with the payload's true size and md5, downloads write the bytes (slicing parts by
    index for the split path), and every call is recorded."""
    import hashlib
    import re as _re

    def run(args, *, stdin=None, **_kwargs):
        calls.append((list(args), stdin))
        if args[0] == "exec" and stdin and "getsize" in stdin:
            remote = _re.search(r"os\.path\.exists\('([^']+)'\)", stdin).group(1)
            payload = files.get(remote)
            if payload is None:
                return "MISSING"
            return f"SIZE {len(payload)} {hashlib.md5(payload).hexdigest()}"
        if args[0] == "download":
            remote, local = args[1], Path(args[2])
            base = remote.rsplit("/", 1)[1]
            if ".part" in base:
                stem, index = base.rsplit(".part", 1)
                payload = files[f"{remote.rsplit('/', 1)[0]}/{stem}"]
                local.write_bytes(payload[int(index) * chunk:(int(index) + 1) * chunk])
            else:
                local.write_bytes(files[remote])
        return ""
    return run


def test_a_small_file_downloads_whole_and_checked(monkeypatch, tmp_path):
    """The probe runs even for small files: a truncated short download is a corrupt
    replay waiting to happen, so the checksum is always compared."""
    calls: list = []
    payload = b"a small dump"
    monkeypatch.setattr(cb, "run",
                        _vm_holding({"/o/dump.bin": payload}, cb.UPLOAD_CHUNK, calls))
    local = tmp_path / "dump.bin"
    cb.download("s", "/o/dump.bin", local)
    assert local.read_bytes() == payload
    kinds = [args[0] for args, _ in calls]
    assert kinds == ["exec", "download"]            # probe, then one plain download


def test_a_file_over_the_apis_ceiling_comes_back_in_proven_sized_pieces(
        monkeypatch, tmp_path):
    """The link is the same in both directions: what uploads split under the contents
    API's ceiling, downloads split too. Found live: a ~460 MB head dump failed all
    three one-piece attempts twenty minutes after a 208 MB one came back fine."""
    calls: list = []
    chunk = 10
    monkeypatch.setattr(cb, "UPLOAD_CHUNK", chunk)
    payload = b"0123456789abcdefghij01234"          # 25 bytes -> three parts, the last short
    monkeypatch.setattr(cb, "run", _vm_holding({"/o/dump.bin": payload}, chunk, calls))
    local = tmp_path / "dump.bin"
    cb.download("s", "/o/dump.bin", local)
    assert local.read_bytes() == payload
    downloads = [args for args, _ in calls if args[0] == "download"]
    assert [Path(d[1]).name for d in downloads] == ["dump.bin.part0000",
                                                    "dump.bin.part0001",
                                                    "dump.bin.part0002"]
    split = next(stdin for _, stdin in calls if stdin and "while block :=" in stdin)
    assert "part{index:04d}" in split
    cleanup = next(stdin for _, stdin in calls if stdin and "glob.glob" in stdin)
    assert "os.remove" in cleanup


def test_a_remote_file_that_is_not_there_is_missing_not_failed(monkeypatch, tmp_path):
    """Distinct errors on purpose: a wrapper that wrote no dump must not fail a good
    run, while a dump that exists and will not cross the link must."""
    calls: list = []
    monkeypatch.setattr(cb, "run", _vm_holding({}, 10, calls))
    with pytest.raises(cb.RemoteMissing, match="not on the session"):
        cb.download("s", "/o/none.bin", tmp_path / "none.bin")
    assert not any(args[0] == "download" for args, _ in calls)


def test_a_corrupted_download_is_refused_not_shipped(monkeypatch, tmp_path):
    """A short or corrupt transfer must fail here, not land as a subtly wrong dump."""
    calls: list = []

    def lying_vm(args, *, stdin=None, **_kwargs):
        calls.append((list(args), stdin))
        if args[0] == "exec" and stdin and "getsize" in stdin:
            import hashlib
            import re as _re
            remote = _re.search(r"os\.path\.exists\('([^']+)'\)", stdin).group(1)
            return f"SIZE 9 {hashlib.md5(b'012345678').hexdigest()}" if remote else "MISSING"
        if args[0] == "download":
            Path(args[2]).write_bytes(b"01234567X")  # one byte wrong in flight
        return ""
    monkeypatch.setattr(cb, "run", lying_vm)
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)
    with pytest.raises(cb.ColabError, match="checksum mismatch"):
        cb.download("s", "/o/dump.bin", tmp_path / "dump.bin")
    downloads = [c for c in calls if c[0][0] == "download"]
    assert len(downloads) == 3          # two whole retries, then the honest refusal


def test_a_truncated_transfer_is_tried_again_whole(monkeypatch, tmp_path):
    """The CLI's failure wordings are an open set — IncompleteRead one hour, an SSL
    record layer failure the next (both live on the same 545 MB head dump,
    2026-10-29) — so a transfer whose checksum says 'short' is simply tried again,
    whole, whatever the CLI said on the way out. A truncated body that exits zero is
    the same class of failure as a nonzero one."""
    calls = {"n": 0}
    payload = b"0123456789"

    def flaky_vm(args, *, stdin=None, **_kwargs):
        calls["n"] += 1
        if args[0] == "exec" and stdin and "getsize" in stdin:
            import hashlib
            import re as _re
            remote = _re.search(r"os\.path\.exists\('([^']+)'\)", stdin).group(1)
            digest = hashlib.md5(payload).hexdigest()
            return f"SIZE {len(payload)} {digest}" if remote else "MISSING"
        if args[0] == "download":
            Path(args[2]).write_bytes(b"01234567" if calls["n"] == 2 else payload)
        return ""

    monkeypatch.setattr(cb, "run", flaky_vm)
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)
    local = tmp_path / "dump.bin"
    cb.download("s", "/o/dump.bin", local)
    assert local.read_bytes() == payload
    assert calls["n"] == 3              # probe, the short try, the good try


def test_a_probe_that_cannot_be_read_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr(cb, "run", lambda *a, **k: "kernel hiccup")
    with pytest.raises(cb.ColabError, match="could not stat"):
        cb.download("s", "/o/dump.bin", tmp_path / "dump.bin")


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


def test_the_shared_session_is_offered_only_when_live():
    """The lab's one shared VM is what makes a second click start instantly; when it is
    not running there is nothing to offer and the supervisor rents as before."""
    assert cb.shared_session(live={"i2l-lab", "other-1"}) == "i2l-lab"
    assert cb.shared_session(live=set()) is None


def test_the_shared_session_is_not_a_warm_session_of_any_backend(tmp_path):
    """It is not named after a backend, so the per-route picker must never hand it out
    behind the sharing flag's back (its busy-guard and prefetch only run on that path)."""
    path = tmp_path / "colab.json"
    now = 1_000_000.0
    cb.touch(cb.SHARED_SESSION, path=path, now=now - 60)
    assert cb.warm_session("hi3d", path=path, now=now,
                          live={cb.SHARED_SESSION}) is None


# --- a channel that dies silently (found live 2026-10-28) -----------------------------------


def test_a_channel_that_goes_silent_is_killed_and_named(tmp_path, monkeypatch):
    """A dead exec channel prints what it has and then nothing, for as long as its
    timeout allows — found live, hanging to 3600s while the job behind it looked busy
    the whole time. Silence past the caller's threshold kills the channel and names the
    session, and is not retried: twenty missed beats in a row already is the answer."""
    cli = tmp_path / "colab"
    cli.write_text("#!/bin/sh\necho one line\nsleep 30\n")
    cli.chmod(0o755)
    monkeypatch.setattr(cb, "colab_bin", lambda: str(cli))
    began = time.monotonic()
    with pytest.raises(cb.ColabError, match="session 'i2l-lab' went silent"):
        cb.run(["exec", "-s", "i2l-lab", "--timeout", "60"], timeout=60,
               attempts=2, silence_seconds=0.5)
    assert time.monotonic() - began < 10  # killed fast — and no 20s retry backoff either


def test_a_channel_that_keeps_talking_is_left_alone(tmp_path, monkeypatch):
    """The watchdog's whole point is telling a dead link from a busy one, so a channel
    that keeps talking must run to its natural end, with its lines crossing live."""
    cli = tmp_path / "colab"
    cli.write_text(
        "#!/bin/sh\ni=0\nwhile [ $i -lt 5 ]; do echo line $i; sleep 0.2; i=$((i+1)); done\n")
    cli.chmod(0o755)
    monkeypatch.setattr(cb, "colab_bin", lambda: str(cli))
    seen = []
    output = cb.run(["exec", "-s", "s", "--timeout", "30"], timeout=30,
                    silence_seconds=5, on_output=seen.append)
    assert "line 4" in output
    assert seen and seen[0] == "line 0\n"  # crossed as it happened, not at the end
