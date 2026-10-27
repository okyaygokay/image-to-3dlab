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
import tarfile
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
    # Pixal3D is a C++/GGML binary driven by a Python wrapper, not a torch model, so its
    # setup is a venv for the wrapper plus the prebuilt trellis-cli and its weights.
    "pixal3d": {
        "setup": "scripts/bootstrap_pixal3d_colab.py",
        "wrapper": "scripts/pixal3d_generate.py",
        "label": "Pixal3D (C++/GGML)",
        # Written beside the GLB on the VM, and needed after the run: the record and views
        # are what Finish's Pixel Match matches by, the dump is what its Pixal3D route
        # reruns, and Hero reuses all three. Without them a Colab-made model could be
        # finished, but only the classic way, with no photo and no hero head.
        "artifacts": (".json", ".post.bin", ".svviews"),
    },
    # The picture engine: text-to-image, so its "image" slot carries the request JSON and
    # its output is a PNG. Upstream has no Linux CUDA prebuilt, so its setup compiles.
    "qwen_image": {
        "setup": "scripts/bootstrap_qwen_image_colab.py",
        "wrapper": "scripts/qwen_image_colab.py",
        "label": "Qwen-Image (text to picture)",
    },
    # The auto-rigger: a rig with no forwarded flags, a preset clip with --preset. The
    # wrapper runs the local route's own commands, against the VM's SkinTokens install.
    "autorig": {
        "setup": "scripts/bootstrap_autorig_colab.py",
        "wrapper": "scripts/autorig_colab.py",
        "label": "Auto-rig (SkinTokens)",
    },
    # The Animate tab's prompt box: a description becomes a clip (MiniMax H3, int8),
    # 3D motion (GVHMR), fingers (WiLoR) and finally a retargeted animation, all on the
    # VM, in one supervisor run.
    "prompt2rig": {
        "setup": "scripts/bootstrap_prompt2rig_colab.py",
        "wrapper": "scripts/prompt2rig_colab.py",
        "label": "Prompt-to-motion (H3 · GVHMR · WiLoR)",
        # The clip the motion was read from and the packed motion itself come home beside
        # the animated GLB: the clip is the evidence of what was asked for, the motion
        # replays on any other rig (the Animate tab plays it directly).
        "artifacts": (".motion.npz", ".clip.mp4"),
    },
}

#: The venv each setup creates on the VM, relative to REMOTE_ROOT. A setup that installs
#: into this machine's interpreter is not usable remotely, so every supported route has
#: its own; that is why the Mac-only routes are not in BACKENDS.
REMOTE_PYTHON = {
    "hi3d": f"{REMOTE_ROOT}/vendor/hi3d/.venv/bin/python",
    "hunyuan_cuda": f"{REMOTE_ROOT}/vendor/hunyuan-cuda/.venv/bin/python",
    "trellis_cuda": f"{REMOTE_ROOT}/vendor/trellis-cuda/.venv/bin/python",
    "pixal3d": f"{REMOTE_ROOT}/vendor/pixal3d-cpp/.venv/bin/python",
    "qwen_image": f"{REMOTE_ROOT}/vendor/sdcpp/.venv/bin/python",
    "autorig": f"{REMOTE_ROOT}/vendor/SkinTokens/.venv/bin/python",
    "prompt2rig": f"{REMOTE_ROOT}/vendor/prompt2rig/.venv/bin/python",
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
SYNC_DIRS = ("image_to_3dlab", "scripts", "viewer")
SYNC_FILES = ("requirements.txt", "requirements.lock")


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
        # The animation presets: small .npz clips the remote retarget reads. Only these
        # binaries ride along -- the sync exists to run code, and every other file the
        # wrappers need is either a weight (installed by the VM's own bootstrap) or an
        # input (uploaded per job).
        motions = REPO / "image_to_3dlab" / "motions"
        if motions.is_dir():
            for item in sorted(motions.glob("*.npz")):
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
        colab.upload(session, archive, remote, timeout=600)
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
    # The subprocess's output is captured and reprinted by the kernel itself, because the
    # exec channel only relays the kernel's own prints -- a bootstrap's stdout and stderr
    # (including the one line that says why it died) otherwise never cross the websocket,
    # which is exactly how a real failure surfaced as a bare "SETUP_EXIT 1" with nothing
    # to act on. The tail is bounded: progress bars are noise, the end is where uv and the
    # bootstraps say what went wrong. The two marker lines before it (kernel version,
    # script size) pin down the "ran nothing at all" class of failure in one shot.
    code = (
        f"import os, subprocess, sys\n"
        f"script = {REMOTE_ROOT + '/' + spec['setup']!r}\n"
        f"print('SETUP_PY', sys.version.split()[0], sys.executable)\n"
        f"print('SETUP_SCRIPT', os.path.exists(script),\n"
        f"      os.path.getsize(script) if os.path.exists(script) else -1)\n"
        f"p = subprocess.run([sys.executable, script, '--yes'], cwd={REMOTE_ROOT!r},\n"
        f"                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n"
        f"                   text=True, errors='replace')\n"
        f"print(p.stdout[-12000:])\n"
        f"print('SETUP_EXIT', p.returncode)\n"
    )
    output = colab.run(["exec", "-s", session, "--timeout", "3600"],
                       stdin=code, timeout=3900, attempts=2)
    # Stream the useful lines as they arrive, but keep the whole thing: an earlier version
    # filtered to a few prefixes and threw away the one line saying why the install failed,
    # leaving "SETUP_EXIT 1" and nothing else to act on.
    for line in output.splitlines():
        if line.startswith(("error", "  $", "echo", "Cannot", "This installs",
                            "Refusing", "SETUP_")):
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
    argv = [REMOTE_PYTHON[backend], f"{REMOTE_ROOT}/{spec['wrapper']}",
            image_remote, output_remote, *extra]
    return launch_remote_argv(session, backend, argv, output_remote)


def launch_remote_argv(session: str, backend: str, argv: list[str],
                       output_remote: str) -> str:
    """Start one command detached on the VM; return the job's remote directory.

    The command is the backend's wrapper for a generation, or any repo script for the
    `--exec` mode (Hero's crop upscale rides this). The detachment and the waiter are the
    same either way: a crash must read as a crash, whatever the command was.
    """
    workdir = f"{REMOTE_ROOT}/jobs/{int(time.time() * 1000)}"
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


def fetch_artifacts(session: str, backend: str, remote_output: str, local_output: Path,
                    scratch: Path | None = None) -> list[Path]:
    """Bring a run's side files back beside the GLB, so Finish can find them later.

    Finish matches a model to its run *by content* -- it scans the local output tree for a
    small record whose `output.sha256` matches the uploaded GLB -- so the artifacts must
    land under the same local tree with the same stem, which is exactly where the local
    route writes them. Missing ones are skipped quietly: a wrapper that wrote no dump (an
    unpatched build, say) must not fail an otherwise good run. The record is marked
    `"host": "colab"` while it is here, because Hero needs to know its Pixal3D calls must
    ride the supervisor rather than a local binary.
    """
    spec = BACKENDS[backend]
    suffixes = spec.get("artifacts", ())
    if not suffixes:
        return []
    scratch = scratch or Path(os.environ.get("TMPDIR", "/tmp")) / f"i2l-artifacts-{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    stem = Path(remote_output).name.removesuffix(Path(remote_output).suffix)
    brought: list[Path] = []
    for suffix in suffixes:
        remote = f"{Path(remote_output).parent}/{stem}{suffix}"
        target = local_output.with_suffix(suffix)
        if suffix == ".svviews":
            # A directory: the REST contents API moves files, so pack it on the VM first.
            archive = f"{Path(remote_output).parent}/{stem}.svviews.tar"
            code = (f"import shutil\n"
                    f"try:\n"
                    f"    shutil.make_archive({archive.removesuffix('.tar')!r}, 'tar', {remote!r})\n"
                    f"except (OSError, ValueError):\n"
                    f"    raise SystemExit(3)\n")
            try:
                colab.run(["exec", "-s", session, "--timeout", "120"], stdin=code,
                          timeout=180, attempts=2)
                packed = scratch / f"{stem}.svviews.tar"
                colab.run(["download", archive, str(packed), "-s", session],
                          timeout=600, attempts=3)
                target.mkdir(parents=True, exist_ok=True)
                with tarfile.open(packed) as tar:
                    tar.extractall(target)
                packed.unlink(missing_ok=True)
                brought.append(target)
            except (colab.ColabError, tarfile.TarError, OSError):
                continue
        else:
            try:
                # Three tries: these ride the same link the upload does, and a reset on
                # the one file a later step needs (the clean-up dump) must not be mistaken
                # for a wrapper that wrote nothing.
                colab.run(["download", remote, str(target), "-s", session],
                          timeout=600, attempts=3)
            except colab.ColabError:
                continue
            if not target.is_file():
                continue
            if suffix == ".json" and target.stat().st_size <= 100_000:
                try:
                    record = json.loads(target.read_text())
                    record["host"] = "colab"
                    target.write_text(json.dumps(record, indent=2))
                except (json.JSONDecodeError, OSError):
                    pass  # the wrapper's own bytes are better than nothing; ship them as-is
            brought.append(target)
    return brought


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
    parser.add_argument("image", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
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
    # One repo script run in the backend's venv on the VM, instead of a generation. Hero's
    # head-crop upscale rides this when its Pixal3D calls are Colab-hosted: `{in}` and
    # `{out}` in the forwarded args name the uploaded input and the fetched output, so the
    # script itself never learns it is running remotely.
    parser.add_argument("--exec", dest="exec_script", metavar="SCRIPT",
                        help="Run SCRIPT on the VM in the backend's venv (with --exec-input/"
                             "--exec-output and {in}/{out} placeholders in the forwarded args).")
    parser.add_argument("--exec-input", type=Path, metavar="LOCAL",
                        help="A local file to upload first; its remote path replaces {in}.")
    parser.add_argument("--exec-output", type=Path, metavar="LOCAL",
                        help="Where the remote output lands here; its remote path replaces {out}.")
    parser.add_argument("--exec-artifacts", nargs="+", metavar="NAME", default=[],
                        help="With --exec: extra file names the helper writes beside its "
                             "output on the VM (the check sheet's other shots, say), "
                             "fetched beside the output here. Missing ones are skipped.")
    parser.add_argument("--exec-bin", action="store_true",
                        help="With --exec: run SCRIPT as a binary in the backend's install "
                             "(e.g. vendor/pixal3d-cpp/build/post-replay), not under the venv's "
                             "python. Hero's head clean-up rides this.")
    # Everything the remote wrapper needs goes through untouched. The argv is split at the
    # first `--` *before* parsing, not fished out of `parse_known_args` afterwards: with
    # optional positionals (the `--exec` mode), argparse happily swallows the forwarded
    # `{in} {out}` words as the image and output, and a known flag after `--` (say `--keep`)
    # was consumed by this parser instead of forwarded -- both wrong in the same direction.
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    split = argv.index("--") if "--" in argv else None
    args, extra = parser.parse_known_args(argv if split is None else argv[:split])
    extra = extra + ([] if split is None else argv[split + 1:])

    if args.exec_script:
        if not args.exec_bin:
            # A repo script runs through the backend's venv; it must exist here to be
            # uploaded. A --exec-bin binary (post-replay) lives in the backend's install
            # on the VM, not on this disk — checking here would refuse the one machine
            # the route is for: one with nothing installed.
            script = REPO / args.exec_script
            if not script.is_file():
                print(f"error: --exec script does not exist: {args.exec_script}", file=sys.stderr)
                return 2
        if args.exec_output is None:
            print("error: --exec needs --exec-output (where the result lands here)",
                  file=sys.stderr)
            return 2
        if args.image is not None or args.output is not None:
            print("error: --exec replaces the image and output arguments", file=sys.stderr)
            return 2
        upload_source = args.exec_input
    else:
        if args.image is None or args.output is None:
            print("error: an image and an output GLB are required unless --exec is used",
                  file=sys.stderr)
            return 2
        upload_source = args.image
    if upload_source is not None and not upload_source.is_file():
        print(f"error: input file does not exist: {upload_source}", file=sys.stderr)
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
    stem = upload_source.stem if upload_source is not None else "exec"
    session = session or colab.session_name(stem, args.backend)
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
        workflow=(os.path.basename(args.exec_script) + " through " +
                  os.path.basename(spec['setup']) + "'s venv") if args.exec_script else
                 f"{os.path.basename(spec['setup'])} then "
                 f"{os.path.basename(spec['wrapper'])}",
    ), flush=True)
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to rent a GPU without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing started.")
            return 1

    if args.exec_script:
        remote_in = (f"{REMOTE_ROOT}/inputs/{args.exec_input.name}"
                     if args.exec_input is not None else None)
        remote_out = f"{REMOTE_ROOT}/outputs/{args.exec_output.name}"
    else:
        remote_in = f"{REMOTE_ROOT}/inputs/{args.image.name}"
        remote_out = f"{REMOTE_ROOT}/outputs/{args.output.name}"
    local_out = args.exec_output if args.exec_script else args.output
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
        if remote_in is not None:
            # However large it is: the 342 MB clean-up dump rides the same call as a
            # 1 MB crop, in pieces the contents API has proven it can carry.
            colab.upload(session, upload_source, remote_in, timeout=600)
            log(f"uploaded {upload_source.name}")
        sync_tree(session)
        ensure_remote_setup(session, args.backend, skip=args.skip_setup)

        if args.exec_script:
            forwarded = [value.replace("{in}", remote_in or "").replace("{out}", remote_out)
                         .replace("{outdir}", str(Path(remote_out).parent))
                         for value in extra]
            prefix = ([f"{REMOTE_ROOT}/{args.exec_script}"] if args.exec_bin
                      else [REMOTE_PYTHON[args.backend], f"{REMOTE_ROOT}/{args.exec_script}"])
            workdir = launch_remote_argv(session, args.backend, prefix + forwarded, remote_out)
        else:
            workdir = launch_remote_job(session, args.backend, remote_in, remote_out, extra)
        log(f"remote job started; watching {workdir}")
        status = wait_for_remote_job(session, workdir, timeout=args.timeout)
        if status.get("state") != "done":
            detail = status.get("error") or f"exit code {status.get('exit_code', '?')}"
            raise colab.ColabError(f"the remote run failed: {detail}")

        local_out.parent.mkdir(parents=True, exist_ok=True)
        # Three tries: this is the finished GLB, and losing it to a reset after an hour
        # of rented GPU is the most expensive byte in the whole route.
        colab.run(["download", remote_out, str(local_out), "-s", session],
                  timeout=1800, attempts=3)
        if not local_out.is_file():
            raise colab.ColabError(
                f"the VM finished but {remote_out} did not come back. The file may be "
                f"missing; check with `colab ls -s {session} {REMOTE_ROOT}/outputs`."
            )
        log(f"wrote {local_out}")
        if args.exec_script and args.exec_artifacts:
            # A helper's side files: named, not suffixed, because they are written by the
            # helper itself (the check sheet's shots), not by a wrapper's convention.
            for name in args.exec_artifacts:
                side = f"{Path(remote_out).parent}/{name}"
                try:
                    colab.run(["download", side, str(local_out.parent / name), "-s", session],
                              timeout=600, attempts=3)
                except colab.ColabError:
                    continue  # the helper decides what it writes; absent is not an error
            log(f"brought back the helper's side files beside {local_out.name}")
        if not args.exec_script:
            # A generation's side files, never an --exec helper's: the helper's output has
            # no run record, and looking would cost three REST round-trips for 404s.
            brought = fetch_artifacts(session, args.backend, remote_out, local_out)
            if brought:
                log(f"brought back {', '.join(part.name for part in brought)} beside the GLB")
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
