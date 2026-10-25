#!/usr/bin/env python3
"""Wait for one detached job on the Colab VM and record how it ended.

    python scripts/colab_supervisor.py job.json

``scripts/colab_generate.py`` starts a job with ``start_new_session`` so that a websocket
hiccup cannot kill an hour of refinement. Detaching has a cost: nobody waits for the child,
so when it exits the kernel leaves a zombie -- ``ps`` lists a process, the GPU sits idle, and
there is no exit code or traceback anywhere. A crash becomes indistinguishable from a hang.

This is the waiter. It runs in its own session, streams the child's output into the job's
log, and on a non-zero exit writes ``{"state": "error", ...}`` to the job's status file,
which is exactly what the local poller reads. On success it leaves the wrapper's own
``{"state": "done"}`` in place.

The job description is JSON on the command line rather than stdin, because this is started
through a shell that has already handed its own stdin to a kernel.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: colab_supervisor.py JOB.json", file=sys.stderr)
        return 2
    job = json.loads(Path(argv[0]).read_text())
    mark = Path(job["status"])

    def fail(message: str) -> None:
        # Written atomically: the poller reads this file between writes and must never see
        # half a JSON object and decide the job is still running.
        scratch = mark.with_suffix(".tmp")
        scratch.write_text(json.dumps({"state": "error", "error": message}))
        os.replace(scratch, mark)

    try:
        with open(job["log"], "ab") as log:
            proc = subprocess.run(
                job["argv"], stdout=log, stderr=subprocess.STDOUT,
                cwd=job.get("cwd"), env={**os.environ, **job.get("env", {})},
                check=False,
            )
    except OSError as exc:
        fail(f"could not start the remote job: {exc}")
        return 1

    if proc.returncode != 0:
        fail(f"the remote job exited with code {proc.returncode}; see the log above")
        return proc.returncode

    # A clean exit still has to be *said*, or the poller reads the launcher's
    # {"state": "running"} forever on a job that already produced its GLB. The wrapper does
    # not write this file itself: it prints DONE and leaves, which is right for a local run
    # and invisible to the watcher of a remote one.
    try:
        current = json.loads(mark.read_text())
    except (OSError, json.JSONDecodeError):
        current = {}
    if current.get("state") != "done":
        scratch = mark.with_suffix(".tmp")
        scratch.write_text(json.dumps({
            "state": "done",
            "output": job.get("output"),
            "manifest": job.get("manifest"),
        }))
        os.replace(scratch, mark)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
