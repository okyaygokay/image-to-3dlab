#!/usr/bin/env python3
"""Install the lab's other backends on this VM, one at a time, in the background.

Started detached by `scripts/colab_generate.py` while a job runs, on a session that
outlives the job (the shared VM, or a `--keep` one). Everything here is optional: a
skipped or failed install only means the next job pays the install itself.

    python scripts/colab_prefetch.py --plan '[["autorig","scripts/x.py",15], ...]' \
        --root /content/i2l

Rules, in order:

* a stamp (`.setup-<backend>-stamp`) says a backend is already installed;
* free disk must cover the estimate plus `HEADROOM_GB`;
* an install of `HEAVY_GB` or more waits until no lab job is running — big installs
  touch the GPU (H3's components are quantized as they download), and the job owns it;
* one prefetch at a time (a pid file; a dead one is replaced).

Each install's tail and exit code land in `prefetch.log` beside the root, which is where
the launcher pointed this process's stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

#: An install this big may touch the GPU, so it waits for the running job to finish.
HEAVY_GB = 20
#: Free-space floor the prefetch leaves behind, on top of every estimate. It is not
#: just unpacking room: the next job needs scratch space (a clip, motion dumps, chunked
#: artifact pieces), and a warm VM may need a venv rebuilt — 10 GB left a stuffed VM
#: with nothing left to run anything on (found live, at 0.4 GB free).
HEADROOM_GB = 30
#: A `running` job marker older than this is a waiter that died, not a job.
MARKER_STALE_SECONDS = 8 * 60 * 60


def free_gb(root: Path) -> float:
    """Free space under `root`, in GB (Colab keeps one filesystem)."""
    stat = os.statvfs(root)
    return stat.f_bavail * stat.f_frsize / 1e9


def job_running(root: Path) -> bool:
    """Whether one of this lab's jobs is still running here (one GPU, one job)."""
    now = time.time()
    for status in (root / "jobs").glob("*/status.json"):
        try:
            state = json.loads(status.read_text()).get("state")
        except (OSError, ValueError):
            continue
        if state == "running" and now - status.stat().st_mtime < MARKER_STALE_SECONDS:
            return True
    return False


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True                     # alive, just not ours
    except OSError:
        return False
    return True


def install(plan: list, root: Path, *, sleep=time.sleep, now=None) -> list[tuple[str, int]]:
    """Run the plan under the rules above; returns (backend, exit code) per install."""
    done = []
    for backend, script, estimate in plan:
        if (root / f".setup-{backend}-stamp").is_file():
            print(f"{backend}: already installed", flush=True)
            continue
        free = free_gb(root)
        if free < estimate + HEADROOM_GB:
            print(f"{backend}: skipped — {free:.0f} GB free, needs ~{estimate + HEADROOM_GB}",
                  flush=True)
            continue
        while estimate >= HEAVY_GB and job_running(root):
            print(f"{backend}: waiting for the running job (its install touches the GPU)",
                  flush=True)
            sleep(120)
        print(f"{backend}: installing (~{estimate} GB)", flush=True)
        finished = subprocess.run([sys.executable, str(root / script), "--yes"],
                                  cwd=str(root), capture_output=True, text=True,
                                  check=False)
        tail = ((finished.stdout or "") + (finished.stderr or ""))[-4000:]
        if tail.strip():
            print(tail, flush=True)
        print(f"PREFETCH_EXIT {backend} {finished.returncode}", flush=True)
        done.append((backend, finished.returncode))
    return done


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True,
                        help="JSON list of [backend, setup script, estimate GB]")
    parser.add_argument("--root", type=Path, default=Path("/content/i2l"))
    args = parser.parse_args(argv)
    plan = [entry for entry in json.loads(args.plan) if isinstance(entry, list)]
    args.root.mkdir(parents=True, exist_ok=True)
    pid_file = args.root / ".prefetch.pid"
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text().strip())
        except ValueError:
            pid = 0
        if pid and pid_alive(pid):
            print("a prefetch is already running; nothing to do", flush=True)
            return 0
    pid_file.write_text(str(os.getpid()))
    try:
        install(plan, args.root)
    finally:
        pid_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
