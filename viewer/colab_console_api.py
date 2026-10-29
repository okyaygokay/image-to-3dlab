"""Run Python on the rented GPU VM from the studio — a console with no terminal.

Every route in this viewer exists to run one worked-out thing; the console exists for
everything else: the question nobody drew a button for ("how much disk is left?", "is
Drive mounted?", "what does the mesh look like from here?"). Code typed in the page runs
on the lab's shared VM (``i2l-lab``) the same way the job legs run — one bounded
``colab exec``, the code riding stdin, each output line streamed back to a log the page
polls — so the answer lands beside the question and no terminal ever opens. It is the
same trust boundary as the job queue beside it (localhost, the studio's own key): the
machine and the VM are the user's own.

Google Drive: a button mounts it with the CLI's own ``drivemount``, which is an
interactive ceremony the page walks you through — the consent URL Google prints is
shown as a link, and the "I clicked Allow — continue" button writes the Enter the CLI
waits for. That Enter is why the mount runs under a pseudo-terminal this server owns
(found live, 2026-10-29: the CLI reads it from ``/dev/tty``, so a plain stdin pipe
never reaches it and the mount times out on the VM's 120 s clock). Mounted Drive is
FUSE: fine for copying finished models into, wrong for installs (nothing installed
there runs, and venvs and mmapped weights crawl) — the lab installs from the network,
as ever.

A run is one JSON status and one log under ``output/console/<id>/``, the shape the rest
of the lab reads. Runs are not queued here: the VM has a single kernel, so a second run
waits behind the first exactly like job legs do, its own clock ticking.
"""

from __future__ import annotations

import json
import os
import pty
import re
import select
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import colab_backend as colab

CONSOLE_ROOT = REPO / "output" / "console"

#: What a run may ask of the VM's kernel, in seconds — and what this side waits for it,
#: with margin for the channel's own chatter.
MIN_TIMEOUT = 5
MAX_TIMEOUT = 1800
CHANNEL_MARGIN = 240

#: A mount waits on consent and the FUSE handshake, neither of which hurries.
MOUNT_TIMEOUT = 900

#: Mount runs' pty masters, by job id: the CLI's "Press Enter after you have granted
#: access" reads /dev/tty, so the mount gets a terminal this server holds and the
#: page's continue button writes into it.
_PTYS: dict[str, int] = {}

#: The code the status probe runs on the VM: one JSON line about disk and Drive.
#: Drive's own capacity is deliberately not asked for: statvfs through DriveFS answers
#: with the local disk's numbers (found live, 2026-10-29), and a made-up figure is
#: worse than none — the mount point and its reality are what the page can stand on.
PROBE = (
    "import json, shutil\n"
    "def drive():\n"
    "    for line in open('/proc/mounts'):\n"
    "        parts = line.split()\n"
    "        if len(parts) > 1 and 'drive' in (parts[0] + parts[1]).lower():\n"
    "            return parts[1]\n"
    "    return None\n"
    "total, used, free = shutil.disk_usage('/content')\n"
    "answer = {'free_gb': round(free / 1e9), 'total_gb': round(total / 1e9)}\n"
    "mount = drive()\n"
    "if mount and __import__('os').path.isdir(mount + '/MyDrive'):\n"
    "    answer['drive'] = {'mount': mount}\n"
    "print('VMSTATUS ' + json.dumps(answer))\n"
)

_HEX = re.compile(r"[0-9a-f]{32}")


def _job_dir(job: str) -> Path:
    if not _HEX.fullmatch(job):
        raise ValueError("no such run")
    return CONSOLE_ROOT / job


def _write_status(job_dir: Path, **fields: Any) -> None:
    """Update the run's status atomically: the page polls this file mid-run."""
    status = json.loads((job_dir / "status.json").read_text(encoding="utf-8")) \
        if (job_dir / "status.json").is_file() else {}
    status.update(fields)
    tmp = job_dir / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    tmp.replace(job_dir / "status.json")


def _spawn(job_dir: Path, argv: list[str], label: str, timeout: float, stdin: str | None) -> None:
    """Run one colab CLI call on a thread, streaming every line to the run's log."""
    log_path = job_dir / "log.txt"

    def note(line: str) -> None:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(line.rstrip("\n") + "\n")

    def work() -> None:
        started = time.time()
        (log_path).write_text("", encoding="utf-8")
        try:
            output = colab.run(argv, timeout=timeout + CHANNEL_MARGIN, attempts=1,
                               stdin=stdin, on_output=note)
            _write_status(job_dir, state="done", exit=0,
                          seconds=round(time.time() - started))
            if not output.strip():
                note("(no output)")
        except colab.ColabError as exc:
            note(str(exc))
            _write_status(job_dir, state="done", exit=1,
                          seconds=round(time.time() - started))
        except Exception as exc:
            note(f"console bug: {exc!r}")
            _write_status(job_dir, state="done", exit=2,
                          seconds=round(time.time() - started))

    threading.Thread(target=work, daemon=True, name=f"console-{job_dir.name[:8]}").start()


def _spawn_mount(job_dir: Path) -> None:
    """Run the CLI's drivemount under a pty, streaming its output to the run's log.

    The Enter the CLI waits for ("Press Enter after you have granted access") is read
    from /dev/tty — a plain stdin pipe never reaches it (found live, 2026-10-29: three
    mounts timed out on the VM's 120 s clock while the CLI sat at a terminal nobody
    could type into). So the child gets a pty this server owns, and `press_enter`
    writes the Enter when the person says they clicked Allow.
    """
    log_path = job_dir / "log.txt"

    def work() -> None:
        started = time.time()
        log_path.write_text("", encoding="utf-8")
        pid, fd = pty.fork()
        if pid == 0:
            os.execvp("colab", ["colab", "drivemount", "-s", colab.SHARED_SESSION])
        job = job_dir.name
        _PTYS[job] = fd
        exit_code = 1
        try:
            deadline = time.time() + MOUNT_TIMEOUT + 60
            while True:
                wait = max(0.1, min(1.0, deadline - time.time()))
                ready, _, _ = select.select([fd], [], [], wait)
                if time.time() > deadline:
                    with log_path.open("a", encoding="utf-8") as log:
                        log.write("\n[console] mount timed out\n")
                    try:
                        os.kill(pid, 9)
                    except ProcessLookupError:
                        pass
                    break
                if not ready:
                    if os.waitpid(pid, os.WNOHANG)[0]:
                        break
                    continue
                try:
                    chunk = os.read(fd, 4096)
                except OSError:                     # the child closed its side
                    break
                if not chunk:
                    break
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(chunk.decode("utf-8", "replace"))
            try:
                _, status = os.waitpid(pid, 0)      # reap, whatever ended it
                code = os.waitstatus_to_exitcode(status)
                if code is not None and code > 0:
                    exit_code = code
                elif code is None or code < 0:      # killed by a signal
                    exit_code = 1
                else:
                    exit_code = 0
            except (ChildProcessError, OSError):
                pass
        finally:
            _PTYS.pop(job, None)
            try:
                os.close(fd)
            except OSError:
                pass
            _write_status(job_dir, state="done", exit=exit_code,
                          seconds=round(time.time() - started))

    threading.Thread(target=work, daemon=True, name=f"console-{job_dir.name[:8]}").start()


def press_enter(job: str) -> bool:
    """Write the Enter a mount is waiting for — the page's "I clicked Allow" button.

    False when the run is not a live mount (already finished, or never was): the page
    says so and nothing is written anywhere.
    """
    _job_dir(job)                                   # validates the id shape
    fd = _PTYS.get(job)
    if fd is None:
        return False
    try:
        os.write(fd, b"\n")
    except OSError:
        return False
    return True


def start(code: str, timeout: int, *, kind: str = "exec") -> dict[str, Any]:
    """Start one run: `code` on the VM's kernel (kind "exec"), or a Drive mount.

    The timeout is clamped to what the channel tolerates and the code to what a page
    can send; anything past that is a file's job, not a console's.
    """
    if kind not in ("exec", "drivemount"):
        raise ValueError(f"unknown run kind: {kind}")
    seconds = max(MIN_TIMEOUT, min(MAX_TIMEOUT, int(timeout or 30)))
    job = uuid.uuid4().hex
    job_dir = CONSOLE_ROOT / job
    job_dir.mkdir(parents=True, exist_ok=True)
    if kind == "exec":
        text = str(code or "").strip()
        if not text:
            raise ValueError("nothing to run")
        if len(text) > 64 * 1024:
            raise ValueError("code over 64 KB — put it in a script, not the console")
        label = "run Python on the VM"
        argv = ["exec", "-s", colab.SHARED_SESSION, "--timeout", str(seconds)]
        stdin, bound = text, seconds
    else:
        label = "mount Google Drive"
        argv, stdin, bound = None, None, MOUNT_TIMEOUT
    (job_dir / "status.json").write_text(json.dumps(
        {"state": "running", "kind": kind, "label": label, "started": time.time()}),
        encoding="utf-8")
    if kind == "exec":
        _spawn(job_dir, argv, label, bound, stdin)
    else:
        _spawn_mount(job_dir)
    return {"id": job, "state": "running", "kind": kind, "timeout": bound}


def read(job: str) -> dict[str, Any]:
    """One run's state: its status fields plus the log's tail, for the page to poll."""
    job_dir = _job_dir(job)
    status_path = job_dir / "status.json"
    if not status_path.is_file():
        raise ValueError("no such run")
    try:
        answer = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):                   # mid-write: the page asks again
        answer = {"state": "running"}
    answer["id"] = job
    log_path = job_dir / "log.txt"
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        answer["output"] = text[-65536:]
    return answer


def vm_status() -> dict[str, Any]:
    """The VM in one line: alive, disk free, Drive mounted and its room.

    A probe that cannot run is an answer too ("asleep or unreachable"), not an error:
    the page says what it knows and never invents the rest.
    """
    try:
        output = colab.run(["exec", "-s", colab.SHARED_SESSION, "--timeout", "25"],
                           timeout=90, attempts=2, stdin=PROBE)
    except colab.ColabError as exc:
        return {"live": False, "note": str(exc).strip()[-400:]}
    for line in output.splitlines():
        if line.startswith("VMSTATUS "):
            answer = json.loads(line[len("VMSTATUS "):])
            answer.setdefault("drive", None)        # absent means not mounted
            answer["live"] = True
            return answer
    return {"live": True, "note": "the VM answered without a status line"}
