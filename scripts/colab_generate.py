#!/usr/bin/env python3
"""Run one backend on a rented Colab GPU and bring the GLB back.

    python scripts/colab_generate.py in.png out.glb --backend hi3d --accelerator G4

This is the *local* process the studio watches. It does the cloud work and streams the
remote backend's own stage lines to stdout, so the job runner sees exactly the protocol it
already understands from a local run.

What it does, in order:

1. Names the accelerator, the session and the idle timeout (and stops there, unless
   ``--yes``, because a rented GPU is a bill).
2. Allocates the session, uploads the picture, and runs the backend's setup out there --
   once per VM, guarded by a stamp, so a second job on the same session is quick.
3. Starts the real generation detached on the VM and polls its status file. Detached, not
   inline, because a refinement can take an hour and a websocket hiccup must not lose it.
4. Downloads the GLB, then releases the session (unless ``--keep``).

The remote interpreter is not this machine's, so nothing here imports a backend's
dependencies: the wrapper out there does. Locally this needs only the standard library and
the ``colab`` CLI.

Progress lines are passed through verbatim. The wrapper out there prints the same
``STAGE ...`` / ``STEP n/m`` lines it prints locally, which is the whole trick: one
protocol, two machines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import colab_backend as colab  # noqa: E402, RUF100

#: Where the VM keeps everything for this repo. `/content` is the one path Colab persists
#: for the life of a session.
REMOTE_ROOT = "/content/i2l"

#: Each backend's setup and generation on the VM, as (setup script, wrapper script). Kept
#: as data so a new route is one line, not another branch.
BACKENDS: dict[str, dict[str, str]] = {
    "hi3d": {
        "setup": "scripts/bootstrap_hi3d.py",
        "wrapper": "scripts/hi3d_generate.py",
        "label": "Hi3D 1.0 (refinement)",
    },
    "hunyuan_cuda": {
        "setup": "scripts/bootstrap_hunyuan_cuda.py",
        "wrapper": "scripts/hunyuan_cuda_generate.py",
        "label": "Hunyuan3D-2.1 (NVIDIA)",
    },
    "trellis_cuda": {
        "setup": "scripts/bootstrap_trellis_cuda.py",
        "wrapper": "scripts/trellis_cuda_generate.py",
        "label": "TRELLIS.2 (NVIDIA)",
    },
}

#: The venv each setup creates on the VM, relative to REMOTE_ROOT. A setup that installs
#: into this machine's interpreter is not usable remotely, so every supported route has
#: its own; that is why the Mac-only routes are not in BACKENDS.
REMOTE_PYTHON = {
    "hi3d": f"{REMOTE_ROOT}/vendor/hi3d/.venv/bin/python",
    "hunyuan_cuda": f"{REMOTE_ROOT}/vendor/hunyuan-cuda/.venv/bin/python",
    "trellis_cuda": f"{REMOTE_ROOT}/vendor/trellis-cuda/.venv/bin/python",
}

#: The stamp a setup writes when it is done, so a warm session skips 20 minutes of install.
def setup_stamp(backend: str) -> str:
    return f"{REMOTE_ROOT}/.setup-{backend}-stamp"


def log(message: str) -> None:
    print(message, flush=True)


def emit(line: str) -> None:
    """Pass a remote stage line straight through; the studio parses these."""
    print(line.strip(), flush=True)


#: Directories under the repo that a remote run needs. Only Python sources and the
#: requirements file go up: `output/`, `characters/` and `vendor/` are this machine's own
#: work or multi-gigabyte checkouts the VM builds for itself.
SYNC_DIRS = ("image_to_3dlab", "scripts")
SYNC_FILES = ("requirements.txt",)


def _sync_archive(destination: Path) -> Path:
    """Pack the run's code into one gzipped tarball.

    One upload instead of one per file. `colab upload` goes base64 through the REST
    contents API, and the repo has ~195 Python files -- about a minute each on a cold VM,
    which is three hours before the GPU does anything. A single tarball of the same sources
    is well under a megabyte and arrives in one call.
    """
    import tarfile

    with tarfile.open(destination, "w:gz") as archive:
        for name in SYNC_DIRS:
            folder = REPO / name
            if not folder.is_dir():
                continue
            for item in sorted(folder.rglob("*.py")):
                archive.add(item, arcname=str(item.relative_to(REPO)))
        for name in SYNC_FILES:
            item = REPO / name
            if item.is_file():
                archive.add(item, arcname=name)
    return destination


def sync_tree(session: str) -> None:
    """Upload what the VM needs: this package, plus the scripts it drives.

    The tarball is unpacked *over* the existing tree rather than replacing it, so a warm
    session keeps its venv and its downloaded weights; only code is refreshed.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="i2l-sync-") as scratch:
        archive = _sync_archive(Path(scratch) / "i2l-code.tar.gz")
        size = archive.stat().st_size
        remote = f"{REMOTE_ROOT}/i2l-code.tar.gz"
        colab.run(["upload", str(archive), remote, "-s", session], timeout=600)
    colab.run(["exec", "-s", session, "--timeout", "180"], attempts=3,
              stdin=(
                  "import os, tarfile\n"
                  f"os.makedirs({REMOTE_ROOT!r}, exist_ok=True)\n"
                  f"with tarfile.open({remote!r}) as a:\n"
                  f"    a.extractall({REMOTE_ROOT!r})\n"
                  f"os.remove({remote!r})\n"
              ),
              timeout=300)
    log(f"synced the run's code to {REMOTE_ROOT} on the VM ({size / 1024:.0f} KB, one upload)")


def ensure_remote_setup(session: str, backend: str, *, skip: bool) -> None:
    """Install the backend on the VM once per session."""
    spec = BACKENDS[backend]
    if skip:
        return
    stamp = setup_stamp(backend)
    probe = colab.run(
        ["exec", "-s", session, "--timeout", "60"], attempts=3,
        stdin=f"import os; print('STAMP' if os.path.exists({stamp!r}) else 'MISSING')",
        timeout=180,
    )
    if "STAMP" in probe:
        log(f"{spec['label']} is already installed on this VM")
        return

    log(f"installing {spec['label']} on the VM (first run on this session takes a while)")
    log("this is the slow part: 15-30 minutes, and it happens once per VM, not once per job")
    # --yes: the confirmation is the announcement above, made locally. Asking again on a
    # stdin nobody is attached to would hang the job.
    code = (
        f"import subprocess, sys\n"
        f"p = subprocess.run([sys.executable, {REMOTE_ROOT + '/' + spec['setup']!r}, "
        f"'--yes'], cwd={REMOTE_ROOT!r})\n"
        f"print('SETUP_EXIT', p.returncode)\n"
    )
    output = colab.run(["exec", "-s", session, "--timeout", "3600"],
                       stdin=code, timeout=3900, attempts=2)
    # Stream the useful lines as they arrive, but keep the whole thing: an earlier version
    # filtered to a few prefixes and threw away the one line saying why the install failed,
    # leaving "SETUP_EXIT 1" and nothing else to act on.
    for line in output.splitlines():
        if line.startswith(("error", "  $", "echo", "Cannot", "This installs", "Refusing")):
            emit(line)
    if "SETUP_EXIT 0" not in output:
        raise colab.ColabError(
            f"installing {spec['label']} on the VM failed. Full output follows.\n"
            f"{output.strip()[-4000:]}"
        )
    colab.run(["exec", "-s", session, "--timeout", "60"], attempts=3,
              stdin=f"open({stamp!r}, 'w').write('ok')", timeout=180)


def launch_remote_job(session: str, backend: str, image_remote: str,
                      output_remote: str, extra: list[str]) -> str:
    """Start the wrapper detached on the VM; return the job's remote directory."""
    spec = BACKENDS[backend]
    workdir = f"{REMOTE_ROOT}/jobs/{int(time.time() * 1000)}"
    python = REMOTE_PYTHON[backend]
    wrapper = f"{REMOTE_ROOT}/{spec['wrapper']}"
    argv = [python, wrapper, image_remote, output_remote, *extra]
    payload = json.dumps({
        "workdir": workdir,
        "log": f"{workdir}/log.txt",
        "argv": argv,
        "marker": f"{workdir}/status.json",
    })
    # `start_new_session` detaches the job so a websocket hiccup cannot kill it, which also
    # means nobody reaps it -- the kernel does not, either, because this launcher outlives it
    # only for a moment. Without a waiter the child becomes a zombie: `ps` shows a process,
    # the GPU is idle, and no exit code or traceback exists anywhere, so a crash looks like a
    # hang (which is how a real one hid). scripts/colab_supervisor.py is the waiter: it
    # streams the output and turns a crash into a status the poller can read.
    code = f"""
import json, os, subprocess
payload = json.loads({payload!r})
os.makedirs(payload['workdir'], exist_ok=True)
status = payload['marker']
open(status, 'w').write(json.dumps({{'state': 'running', 'stage': 'starting'}}))
job_file = payload['workdir'] + '/job.json'
open(job_file, 'w').write(json.dumps({{
    'argv': payload['argv'], 'log': payload['log'], 'status': status,
    'cwd': {REMOTE_ROOT!r}, 'env': {{'PYTHONUNBUFFERED': '1'}},
    'output': {output_remote!r}, 'manifest': {workdir + '/manifest.json'!r},
}}))
proc = subprocess.Popen(
    [sys.executable, {REMOTE_ROOT + '/scripts/colab_supervisor.py'!r}, job_file],
    start_new_session=True,
)
print('LAUNCHED', proc.pid)
"""
    output = colab.run(["exec", "-s", session, "--timeout", "180"], stdin=code, timeout=300,
              attempts=3)
    if "LAUNCHED" not in output:
        raise colab.ColabError(f"could not start the remote job:\n{output[-1200:]}")
    return workdir


def _fetch_text(session: str, remote: str) -> str | None:
    """Read a small remote text file, or None when it is not there yet.

    Uses `colab download`, which goes over the REST contents API and therefore still works
    while the VM's kernel is busy running the job.
    """
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / f"i2l-poll-{os.getpid()}"
    try:
        colab.run(["download", remote, str(scratch), "-s", session], timeout=180)
    except colab.ColabError:
        return None
    if not scratch.is_file():
        return None
    try:
        return scratch.read_text(errors="replace")
    finally:
        scratch.unlink(missing_ok=True)


def wait_for_remote_job(session: str, workdir: str, *, timeout: int) -> dict:
    """Stream the remote log and return its final status."""
    printed = 0
    started = time.time()
    last_beat = 0.0
    while True:
        colab.touch(session)
        log_text = _fetch_text(session, f"{workdir}/log.txt")
        if log_text and len(log_text) > printed:
            sys.stdout.write(log_text[printed:])
            sys.stdout.flush()
            printed = len(log_text)
        # A refinement can print nothing for many minutes at a time (one line per pass), and
        # the studio treats silence as "stalled". An explicit beat is honest -- the job is
        # alive and being watched -- and it keeps the panel from accusing a working GPU of
        # having hung.
        if time.time() - last_beat > 60:
            last_beat = time.time()
            log(f"still working on the VM ({int(time.time() - started)}s elapsed)")

        marker = _fetch_text(session, f"{workdir}/status.json")
        if marker:
            try:
                status = json.loads(marker)
            except json.JSONDecodeError:
                status = {}
            if status.get("state") in {"done", "error"}:
                return status

        if time.time() - started > timeout:
            raise colab.ColabError(
                f"the remote job in {workdir} passed {timeout / 3600:.1f} hours. "
                f"Stopping it rather than leaving the GPU billed."
            )
        time.sleep(colab.POLL_SECONDS)


def write_remote_status(session: str, workdir: str, state: str, detail: str) -> None:
    """Give a crashed run a terminal status, so the watcher never spins forever."""
    code = (
        "import json\n"
        f"open({workdir + '/status.json'!r}, 'w').write("
        f"json.dumps({{'state': {state!r}, 'error': {detail!r}}}))\n"
    )
    try:
        colab.run(["exec", "-s", session, "--timeout", "60"], attempts=3, stdin=code, timeout=180)
    except colab.ColabError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="hi3d")
    parser.add_argument("--accelerator", choices=colab.ACCELERATORS,
                        default=colab.DEFAULT_ACCELERATOR)
    parser.add_argument("--session", help="Use this session by name (implies --keep)")
    parser.add_argument("--new-session", action="store_true",
                        help="Rent a fresh VM instead of reusing a warm one. Pays the "
                             "install again; use it for a clean-room run.")
    parser.add_argument("--keep", action="store_true",
                        help="Leave the VM running when the job finishes.")
    parser.add_argument("--skip-setup", action="store_true",
                        help="Trust the VM's existing install; do not run the bootstrap.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. The studio passes this once its own "
                             "dialog has asked.")
    parser.add_argument("--timeout", type=int, default=6 * 60 * 60)
    # Everything the remote wrapper needs goes through untouched, so this file never has to
    # learn a backend's flags. `parse_known_args` rather than `nargs=REMAINDER`: REMAINDER
    # starts swallowing at the first positional, so `--yes` after the output path became a
    # forwarded argument and the confirmation fired anyway (found by a real run).
    args, extra = parser.parse_known_args(argv)
    extra = [value for value in extra if value != "--"]

    if not args.image.is_file():
        print(f"error: input image does not exist: {args.image}", file=sys.stderr)
        return 2

    spec = BACKENDS[args.backend]
    # Reuse a warm VM by default. A fresh one spends 15-30 minutes installing before it can
    # do anything, and the studio starts a job per click -- paying that every time is not a
    # usable GUI. `--new-session` is for a clean-room run; `--session` names one explicitly.
    session = args.session
    if session is None and not args.new_session:
        session = colab.warm_session(args.backend)
        if session:
            log(f"reusing the warm session {session!r} (no install needed)")
    session = session or colab.session_name(args.image.stem, args.backend)
    config = colab.RemoteConfig(
        session=session,
        accelerator=args.accelerator,
        stop_when_done=not args.keep and args.session is None,
        timeout_seconds=args.timeout,
    ).validated()

    print(colab.announcement(
        config,
        backend_label=spec["label"],
        weights_gb=None,
        workflow=f"{os.path.basename(spec['setup'])} then "
                 f"{os.path.basename(spec['wrapper'])}",
    ), flush=True)
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to rent a GPU without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing started.")
            return 1

    remote_image = f"{REMOTE_ROOT}/inputs/{args.image.name}"
    remote_output = f"{REMOTE_ROOT}/outputs/{args.output.name}"
    workdir = None
    try:
        # Sessions are now kept for reuse, so something has to reap the ones nobody has
        # touched for hours. This is the only long-lived process the lab runs regularly,
        # which makes it the natural place: leaving it to a cron job nobody installed is how
        # a GPU stays billed for a week.
        for stale in colab.stop_idle_sessions(log=log):
            log(f"stopped {stale!r}: idle for {colab.human_duration(colab.IDLE_STOP_SECONDS)}")
        colab.ensure_session(config, log=log)
        colab.run(["exec", "-s", session, "--timeout", "120"], attempts=3,
                  stdin=f"import os; [os.makedirs(p, exist_ok=True) for p in "
                        f"({REMOTE_ROOT + '/inputs'!r}, {REMOTE_ROOT + '/outputs'!r}, "
                        f"{REMOTE_ROOT + '/jobs'!r})]",
                  timeout=240)
        colab.run(["upload", str(args.image), remote_image, "-s", session], timeout=600)
        log(f"uploaded {args.image.name}")
        sync_tree(session)
        ensure_remote_setup(session, args.backend, skip=args.skip_setup)

        workdir = launch_remote_job(session, args.backend, remote_image, remote_output, extra)
        log(f"remote job started; watching {workdir}")
        status = wait_for_remote_job(session, workdir, timeout=args.timeout)
        if status.get("state") != "done":
            detail = status.get("error") or f"exit code {status.get('exit_code', '?')}"
            raise colab.ColabError(f"the remote run failed: {detail}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        colab.run(["download", remote_output, str(args.output), "-s", session], timeout=1800)
        if not args.output.is_file():
            raise colab.ColabError(
                f"the VM finished but {remote_output} did not come back. The file may be "
                f"missing; check with `colab ls -s {session} {REMOTE_ROOT}/outputs`."
            )
        log(f"wrote {args.output}")
        return 0
    except colab.ColabError as exc:
        if workdir:
            write_remote_status(session, workdir, "error", str(exc)[:1500])
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if config.stop_when_done:
            colab.stop_session(session, log=log)
        else:
            log(f"session {session!r} kept; it stops after "
                f"{colab.human_duration(colab.IDLE_STOP_SECONDS)} idle")


if __name__ == "__main__":
    raise SystemExit(main())
