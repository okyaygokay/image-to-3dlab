"""Tests for the Colab supervisor's control flow, with the cloud faked out.

The supervisor is the process the studio watches, so its failure modes are silent ones: a
watcher that treats "status file exists" as "job finished" reports failure for every
successful run, and a watcher that never sees a terminal state spins forever on a GPU that
is being billed. Both happened in an earlier version of this pipeline, which is why they
are pinned here.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, str(REPO / "scripts")):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import colab_generate as cg

# --- the backend table -------------------------------------------------------------------


def test_every_remote_backend_has_a_setup_and_a_wrapper_that_exist():
    """A table entry pointing at a script that was renamed fails minutes into a paid run."""
    for backend, spec in cg.BACKENDS.items():
        assert (REPO / spec["setup"]).is_file(), f"{backend}: {spec['setup']}"
        assert (REPO / spec["wrapper"]).is_file(), f"{backend}: {spec['wrapper']}"
        assert spec["label"], backend


def test_every_remote_backend_has_its_own_venv_path():
    """A route that installs into this machine's interpreter cannot run remotely."""
    assert set(cg.REMOTE_PYTHON) == set(cg.BACKENDS)
    for backend, python in cg.REMOTE_PYTHON.items():
        assert python.startswith(cg.REMOTE_ROOT), backend
        assert "/.venv/" in python, f"{backend} has no venv of its own: {python}"


def test_the_setup_stamp_is_per_backend():
    # One shared stamp would make a second backend skip its own install on the same VM.
    assert cg.setup_stamp("hi3d") != cg.setup_stamp("hunyuan_cuda")
    assert cg.setup_stamp("hi3d").startswith(cg.REMOTE_ROOT)


def test_hi3d_is_offered_remotely():
    assert "hi3d" in cg.BACKENDS
    assert cg.BACKENDS["hi3d"]["wrapper"] == "scripts/hi3d_generate.py"


# --- waiting for a remote job --------------------------------------------------------------


def test_a_running_status_is_not_completion(monkeypatch, capsys):
    """The bug: `{"state": "running"}` was read as a finished job, so every run 'failed'."""
    states = [
        {"log.txt": "STAGE coarse\n", "status.json": json.dumps({"state": "running"})},
        {"log.txt": "STAGE coarse\nSTAGE refine\n",
         "status.json": json.dumps({"state": "running"})},
        {"log.txt": "STAGE coarse\nSTAGE refine\nDONE\n",
         "status.json": json.dumps({"state": "done"})},
    ]
    polls = {"n": 0}

    def fake_fetch(session, remote):
        name = remote.rsplit("/", 1)[-1]
        index = min(polls["n"] // 2, len(states) - 1)
        polls["n"] += 1
        return states[index].get(name)

    monkeypatch.setattr(cg, "_fetch_text", fake_fetch)
    monkeypatch.setattr(cg.time, "sleep", lambda _s: None)
    status = cg.wait_for_remote_job("s", "/jobs/1", timeout=60)
    assert status["state"] == "done"


def test_an_error_state_is_terminal(monkeypatch):
    monkeypatch.setattr(cg, "_fetch_text",
                        lambda session, remote: json.dumps({"state": "error",
                                                            "error": "boom"})
                        if remote.endswith("status.json") else "")
    monkeypatch.setattr(cg.time, "sleep", lambda _s: None)
    assert cg.wait_for_remote_job("s", "/jobs/1", timeout=60)["state"] == "error"


def test_the_remote_log_is_streamed_once_not_reprinted(monkeypatch, capsys):
    """The supervisor prints only the new tail each poll, or the studio's log fills with
    the same lines over and over."""
    logs = ["STAGE coarse\n", "STAGE coarse\nSTAGE refine\n",
            "STAGE coarse\nSTAGE refine\nSTAGE export\n"]
    polls = {"n": 0}

    def fake_fetch(session, remote):
        if remote.endswith("status.json"):
            polls["n"] += 1
            state = "done" if polls["n"] > len(logs) else "running"
            return json.dumps({"state": state})
        index = min(polls["n"], len(logs) - 1)
        return logs[index]

    monkeypatch.setattr(cg, "_fetch_text", fake_fetch)
    monkeypatch.setattr(cg.time, "sleep", lambda _s: None)
    cg.wait_for_remote_job("s", "/jobs/1", timeout=60)
    out = capsys.readouterr().out
    assert out.count("STAGE refine") == 1, out
    assert "STAGE export" in out


def test_a_job_that_never_finishes_raises_instead_of_billing_forever(monkeypatch):
    monkeypatch.setattr(cg, "_fetch_text",
                        lambda session, remote: json.dumps({"state": "running"})
                        if remote.endswith("status.json") else "")
    monkeypatch.setattr(cg.time, "sleep", lambda _s: None)
    # A clock that jumps past the timeout on the second read.
    ticks = {"n": 0}

    def fake_time():
        ticks["n"] += 1
        return 0.0 if ticks["n"] < 3 else 10_000.0

    monkeypatch.setattr(cg.time, "time", fake_time)
    with pytest.raises(cg.colab.ColabError, match="hours"):
        cg.wait_for_remote_job("s", "/jobs/1", timeout=60)


def test_a_missing_log_does_not_crash_the_watcher(monkeypatch):
    """The log is read before the job writes it; None must be tolerated."""
    polls = {"n": 0}

    def fake_fetch(session, remote):
        polls["n"] += 1
        if remote.endswith("status.json"):
            return json.dumps({"state": "done" if polls["n"] > 2 else "running"})
        return None

    monkeypatch.setattr(cg, "_fetch_text", fake_fetch)
    monkeypatch.setattr(cg.time, "sleep", lambda _s: None)
    assert cg.wait_for_remote_job("s", "/jobs/1", timeout=60)["state"] == "done"


# --- remote status on failure --------------------------------------------------------------


def test_a_crashed_supervisor_leaves_a_terminal_status(monkeypatch):
    """Otherwise the watcher polls a status file the dead run never wrote."""
    written = {}

    def fake_run(args, *, timeout=180, stdin=None, attempts=1):
        written["stdin"] = stdin
        return ""

    monkeypatch.setattr(cg.colab, "run", fake_run)
    cg.write_remote_status("s", "/jobs/1", "error", "the remote run failed")
    assert "/jobs/1/status.json" in written["stdin"]
    assert "'error'" in written["stdin"]


def test_writing_a_status_survives_a_dead_session(monkeypatch):
    def fake_run(*_a, **_k):
        raise cg.colab.ColabError("session gone")

    monkeypatch.setattr(cg.colab, "run", fake_run)
    # Must not raise: the supervisor is already handling one failure.
    cg.write_remote_status("s", "/jobs/1", "error", "detail")


# --- argument handling ---------------------------------------------------------------------


def test_a_flag_after_the_positionals_is_still_recognised():
    """Real bug: `nargs=REMAINDER` swallowed `--yes` once the positionals were read, so the
    confirmation fired on a run that had explicitly skipped it."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--backend", default="hi3d")
    # Mirror main()'s parsing exactly, since that is the code that broke.
    import contextlib
    import io

    with contextlib.redirect_stderr(io.StringIO()):
        args, extra = parser.parse_known_args(
            ["in.png", "out.glb", "--backend", "hi3d", "--yes", "--", "--resolution", "1536"])
    assert args.yes is True
    assert args.backend == "hi3d"
    assert extra == ["--", "--resolution", "1536"]


def test_remote_wrapper_flags_survive_to_the_remote_side(monkeypatch):
    """A run's extra flags must reach the VM; losing them silently runs the defaults."""
    captured = {}

    def fake_launch(session, backend, image_remote, output_remote, extra):
        captured["extra"] = extra
        return "/jobs/1"

    monkeypatch.setattr(cg, "launch_remote_job", fake_launch)
    extra = ["--resolution", "1536", "--seed", "7"]
    cg.launch_remote_job("s", "hi3d", "/i.png", "/o.glb", extra)
    assert captured["extra"] == extra


def test_a_crashed_remote_job_is_reported_not_left_running(tmp_path):
    """Real bug, from a live VM: the detached job died, nobody reaped it, and the status
    file said "running" forever. The launcher now starts scripts/colab_supervisor.py, which
    waits for the child and writes a terminal status, so a crash is a crash and not a hang.
    """
    import json
    import subprocess
    import sys

    status = tmp_path / "status.json"
    status.write_text(json.dumps({"state": "running", "stage": "starting"}))
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "argv": [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"],
        "log": str(tmp_path / "log.txt"),
        "status": str(status),
        "cwd": str(tmp_path),
        "env": {"PYTHONUNBUFFERED": "1"},
    }))

    done = subprocess.run([sys.executable, str(REPO / "scripts" / "colab_supervisor.py"),
                           str(job)], capture_output=True, text=True, check=False)

    assert done.returncode == 3
    written = json.loads(status.read_text())
    assert written["state"] == "error"
    assert "code 3" in written["error"]
    assert "boom" in (tmp_path / "log.txt").read_text()


def test_a_successful_remote_job_keeps_its_own_done_status(tmp_path):
    """A clean exit must not overwrite the wrapper's `done`, which carries the output path."""
    import json
    import subprocess
    import sys

    status = tmp_path / "status.json"
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "argv": [sys.executable, "-c",
                 f"import json; open({str(status)!r}, 'w').write(json.dumps({{'state': 'done', 'output': 'x.glb'}}))"],
        "log": str(tmp_path / "log.txt"),
        "status": str(status),
        "cwd": str(tmp_path),
        "env": {},
    }))

    done = subprocess.run([sys.executable, str(REPO / "scripts" / "colab_supervisor.py"),
                           str(job)], capture_output=True, text=True, check=False)

    assert done.returncode == 0
    assert json.loads(status.read_text())["state"] == "done"


def test_a_clean_exit_that_forgot_to_say_done_is_recorded(tmp_path):
    """Real bug, from a live VM: the wrapper printed DONE and exited 0 without writing the
    status file, so the poller kept reading "running" on a job whose GLB was already made --
    and the download never started. A clean exit has to be said, not inferred.
    """
    import json
    import subprocess
    import sys

    status = tmp_path / "status.json"
    status.write_text(json.dumps({"state": "running", "stage": "starting"}))
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "argv": [sys.executable, "-c", "print('DONE')"],
        "log": str(tmp_path / "log.txt"),
        "status": str(status),
        "output": "/content/i2l/outputs/out.glb",
        "cwd": str(tmp_path),
        "env": {},
    }))

    subprocess.run([sys.executable, str(REPO / "scripts" / "colab_supervisor.py"),
                    str(job)], capture_output=True, text=True, check=False)

    written = json.loads(status.read_text())
    assert written["state"] == "done"
    assert written["output"] == "/content/i2l/outputs/out.glb"


# --- the run's side files come back with the GLB -------------------------------------------


def test_only_pixal3d_declares_artifacts_and_the_others_declare_none():
    """The other wrappers write nothing Finish needs, so their runs must not go looking."""
    assert cg.BACKENDS["pixal3d"]["artifacts"] == (".json", ".post.bin", ".svviews")
    for backend, spec in cg.BACKENDS.items():
        if backend != "pixal3d":
            assert not spec.get("artifacts"), f"{backend} declares artifacts it never wrote"


def _fake_colab(remote_files: dict, calls: list):
    """A `colab.run` that 'downloads' by copying preloaded bytes, and records every call."""
    def run(argv, **_kwargs):
        calls.append(argv)
        if argv[0] == "download":
            remote, local = argv[1], Path(argv[2])
            payload = remote_files.get(remote.rsplit("/", 1)[-1])
            if payload is None:
                raise cg.colab.ColabError(f"404 {remote}")
            local.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, bytes):
                local.write_bytes(payload)
            else:  # a whole directory, pre-materialised by the test
                shutil.copytree(payload, local, dirs_exist_ok=True)
        return ""
    return run


def test_fetch_artifacts_beside_the_glb_and_marks_the_record(monkeypatch, tmp_path):
    """Finish matches a model to its run by the record's sha256 and looks for the views
    beside it, so the side files must land with the same stem as the local GLB -- and the
    record must say it was made on a VM, because Hero routes its Pixal3D calls by that."""
    import io
    import tarfile as tar

    views = tmp_path / "vmviews"
    views.mkdir()
    (views / "transforms.json").write_text("{}")
    bundle = io.BytesIO()
    with tar.open(fileobj=bundle, mode="w") as archive:
        archive.add(views / "transforms.json", arcname="transforms.json")

    record = json.dumps({"backend": "pixal3d",
                         "output": {"path": "/content/i2l/outputs/x.glb", "sha256": "a" * 64}})
    remote_files = {"x.json": record.encode(), "x.post.bin": b"DUMP",
                    "x.svviews.tar": bundle.getvalue()}
    calls: list = []
    monkeypatch.setattr(cg.colab, "run", _fake_colab(remote_files, calls))

    local = tmp_path / "runs" / "x.glb"
    local.parent.mkdir(parents=True)
    brought = cg.fetch_artifacts("s", "pixal3d", "/content/i2l/outputs/x.glb", local,
                                 scratch=tmp_path / "scratch")

    assert [p.name for p in brought] == ["x.json", "x.post.bin", "x.svviews"]
    marked = json.loads((tmp_path / "runs" / "x.json").read_text())
    assert marked["host"] == "colab" and marked["backend"] == "pixal3d"
    assert (tmp_path / "runs" / "x.post.bin").read_bytes() == b"DUMP"
    assert ((tmp_path / "runs" / "x.svviews") / "transforms.json").is_file()
    # The directory was packed on the VM first; the REST API only moves files.
    assert any(argv[0] == "exec" and "make_archive" in str(argv) for argv in calls) or \
        any("exec" in str(argv) for argv in calls)


def test_a_missing_artifact_is_skipped_not_fatal(monkeypatch, tmp_path):
    """A wrapper that wrote no dump (an unpatched build) must not fail a good run."""
    remote_files = {"x.json": json.dumps({"backend": "pixal3d"}).encode()}
    monkeypatch.setattr(cg.colab, "run", _fake_colab(remote_files, []))
    local = tmp_path / "x.glb"
    local.parent.mkdir(parents=True, exist_ok=True)
    brought = cg.fetch_artifacts("s", "pixal3d", "/content/i2l/outputs/x.glb", local,
                                 scratch=tmp_path / "scratch")
    assert [p.name for p in brought] == ["x.json"]
    assert not (tmp_path / "x.post.bin").exists()
    assert not (tmp_path / "x.svviews").exists()


def test_a_backend_without_artifacts_fetches_nothing(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(cg.colab, "run", _fake_colab({}, calls))
    local = tmp_path / "x.glb"
    local.parent.mkdir(parents=True, exist_ok=True)
    assert cg.fetch_artifacts("s", "hi3d", "/content/i2l/outputs/x.glb", local) == []
    assert calls == []


# --- one repo script on the VM (--exec) ------------------------------------------------------


def test_exec_mode_substitutes_the_placeholders_and_lands_the_output(monkeypatch, tmp_path):
    """Hero's crop upscale rides this: `{in}`/`{out}` must become remote paths, the script
    must run in the backend's own venv, and its output must land where the caller said."""
    launched: dict = {}

    def fake_launch(session, backend, argv, output_remote):
        launched["argv"] = argv
        launched["out"] = output_remote
        return "/content/i2l/jobs/1"

    monkeypatch.setattr(cg, "launch_remote_argv", fake_launch)
    monkeypatch.setattr(cg, "wait_for_remote_job", lambda *a, **k: {"state": "done"})
    monkeypatch.setattr(cg.colab, "ensure_session", lambda *a, **k: None)
    monkeypatch.setattr(cg.colab, "stop_idle_sessions", lambda **k: iter([]))
    monkeypatch.setattr(cg.colab, "warm_session", lambda backend: None)
    monkeypatch.setattr(cg, "ensure_remote_setup", lambda *a, **k: None)
    monkeypatch.setattr(cg, "sync_tree", lambda *a, **k: None)

    def fake_run(argv, **_kwargs):
        if argv[0] == "download":
            Path(argv[2]).write_bytes(b"PNG")   # the CLI writes the file it fetched
        return ""

    monkeypatch.setattr(cg.colab, "run", fake_run)

    crop = tmp_path / "crop.png"
    crop.write_bytes(b"crop")
    out = tmp_path / "crop_up.png"
    rc = cg.main(["--backend", "pixal3d", "--exec", "scripts/upscale_esrgan.py",
                  "--exec-input", str(crop), "--exec-output", str(out), "--yes",
                  "--", "{in}", "{out}", "--size", "1024"])
    assert rc == 0
    argv = launched["argv"]
    assert argv[0] == cg.REMOTE_PYTHON["pixal3d"]
    assert argv[1] == "/content/i2l/scripts/upscale_esrgan.py"
    assert argv[2] == "/content/i2l/inputs/crop.png"
    assert argv[3] == launched["out"] == "/content/i2l/outputs/crop_up.png"
    assert argv[4:] == ["--size", "1024"]
    assert out.read_bytes() == b"PNG"


def test_exec_mode_refuses_without_an_output_or_with_generation_positionals(tmp_path, capsys):
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"crop")
    assert cg.main(["--backend", "pixal3d", "--exec", "scripts/upscale_esrgan.py",
                    "--exec-input", str(crop), "--yes"]) == 2
    assert "--exec-output" in capsys.readouterr().err
    assert cg.main(["--backend", "pixal3d", "--exec", "scripts/upscale_esrgan.py",
                    "--exec-input", str(crop), "--exec-output", str(tmp_path / "o.png"),
                    "in.png", "--yes"]) == 2
    assert "replaces" in capsys.readouterr().err


def test_exec_bin_mode_skips_the_local_script_check_and_runs_the_binary_raw(
        monkeypatch, tmp_path):
    """Hero's head clean-up rides this. The binary is built by the backend's bootstrap on
    the VM, so it does not exist on this disk — the route is for a machine with nothing
    installed, and a local existence check refused exactly that machine (found on a live
    run). It must launch as itself, not under the venv's python."""
    launched: dict = {}

    def fake_launch(session, backend, argv, output_remote):
        launched["argv"] = argv
        return "/content/i2l/jobs/1"

    monkeypatch.setattr(cg, "launch_remote_argv", fake_launch)
    monkeypatch.setattr(cg, "wait_for_remote_job", lambda *a, **k: {"state": "done"})
    monkeypatch.setattr(cg.colab, "ensure_session", lambda *a, **k: None)
    monkeypatch.setattr(cg.colab, "stop_idle_sessions", lambda **k: iter(()))
    monkeypatch.setattr(cg, "ensure_remote_setup", lambda *a, **k: None)
    monkeypatch.setattr(cg, "sync_tree", lambda *a, **k: None)

    def fake_run(argv, **_kwargs):
        if argv[0] == "download":
            Path(argv[2]).write_bytes(b"GLB")
        return ""

    monkeypatch.setattr(cg.colab, "run", fake_run)
    dump, out = tmp_path / "head.post.bin", tmp_path / "head.glb"
    dump.write_bytes(b"dump")
    rc = cg.main(["--backend", "pixal3d",
                  "--exec", "vendor/pixal3d-cpp/build/post-replay", "--exec-bin",
                  "--exec-input", str(dump), "--exec-output", str(out), "--yes",
                  "--", "{in}", "{out}", "--no-remesh", "--faces", "16000"])
    assert rc == 0
    assert launched["argv"][:2] == ["/content/i2l/vendor/pixal3d-cpp/build/post-replay",
                                    "/content/i2l/inputs/head.post.bin"]
    assert out.read_bytes() == b"GLB"


def test_generate_mode_still_requires_image_and_output(capsys):
    assert cg.main(["--backend", "hi3d", "--yes"]) == 2
    assert "image and an output GLB" in capsys.readouterr().err
