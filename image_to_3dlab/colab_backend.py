"""Run a generation on a rented Google Colab GPU, driven from this machine.

The studio is local; the compute need not be. This module is the bridge: it provisions a
Colab GPU session with the `colab` CLI, uploads the picture, runs a backend's own wrapper
out there, fetches the GLB back, and releases the session.

Why a separate module rather than another `BackendSpec` branch
--------------------------------------------------------------
Every other route runs `interpreter + wrapper` as a **local** subprocess, and the studio
watches its stdout. A Colab job is that shape too -- the remote process prints the same
stage lines -- but the process is not local, so the two things the job runner assumes are
false: that `job.process` is a child it can kill, and that the working directory exists on
this filesystem. Rather than spread `if remote:` through the runner, the Colab side is
expressed here as a *local* process too: a small supervisor that shells out to `colab`,
streams the same lines, and exits with the run's status. The runner needs no changes.

Nothing here imports torch, and nothing downloads a model. Provisioning a GPU is a decision
with a bill attached, so it names the accelerator and the idle timeout before it starts.

**The idle rule.** A rented GPU costs money by the second. Every call that talks to Colab
refreshes a heartbeat, and a session with no heartbeat for ``IDLE_STOP_SECONDS`` is
stopped -- by the watchdog while you are away, or by the next command if it is not running.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

#: Colab's accelerators. G4 is an RTX PRO 6000 Blackwell (sm_120, ~96 GB) and is the only
#: one this lab has verified Hi3D and Hunyuan on; the rest are offered because the CLI
#: takes them, not because we measured anything.
ACCELERATORS = ("G4", "A100", "L4", "T4", "H100")

#: The accelerator to use when none is named. Pro+ tiers get G4; anything smaller runs
#: Hunyuan coarse + Hi3D refinement far too slowly to be worth the wait.
DEFAULT_ACCELERATOR = "G4"

#: Stop a session after three hours without a heartbeat. Long enough for a 1536 run on a
#: cold cache, short enough that a forgotten session does not cost a night's compute.
IDLE_STOP_SECONDS = 3 * 60 * 60

#: The one session the whole lab shares when sharing is on: any backend's job may land
#: on it, and while a job runs there the supervisor installs the *other* backends on it
#: in the background, so the next step starts in seconds instead of quarter-hours. It is
#: never stopped by a job -- the idle window reaps it, like any other session.
SHARED_SESSION = "i2l-lab"

#: How often the supervisor re-reads the remote status file while a job runs.
POLL_SECONDS = 15

#: `colab new` can return Service Unavailable when a scarce accelerator has no capacity.
ALLOC_ATTEMPTS = 6
ALLOC_DELAY_SECONDS = 60

HEARTBEAT_ENV = "I2L_COLAB_STATE"


@dataclass(frozen=True)
class RemoteConfig:
    """Everything a remote run needs, validated before anything is provisioned."""

    session: str
    accelerator: str = DEFAULT_ACCELERATOR
    stop_when_done: bool = True
    timeout_seconds: int = 6 * 60 * 60

    def validated(self) -> RemoteConfig:
        if self.accelerator not in ACCELERATORS:
            raise ValueError(
                f"accelerator must be one of {', '.join(ACCELERATORS)}; "
                f"got {self.accelerator!r}"
            )
        if not self.session or not self.session.strip():
            raise ValueError("the Colab session needs a name")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        return self


# ----------------------------------------------------------------------------------------
# Pure decisions, so they can be tested without a cloud account
# ----------------------------------------------------------------------------------------


def session_name(image_stem: str, backend_id: str) -> str:
    """A Colab session name for a job, so `colab sessions` is readable at a glance.

    Colab names are shown in the CLI's own listing; a wall of random hex is the thing that
    makes someone leave two sessions running because they cannot tell which is which.

    The suffix is a hash of the clock *and* the process id: truncating the clock to seconds
    collided for two runs started in the same second, which is exactly what a batch or a
    retry does.
    """
    import hashlib
    import re
    import time

    stem = re.sub(r"\.(png|jpe?g|webp|bmp|tiff?|glb|gltf)$", "", image_stem, flags=re.IGNORECASE)
    parts = [part for part in (stem, backend_id) if part and part.strip()]
    slug = re.sub(r"[^a-z0-9]+", "-", "-".join(parts).lower()).strip("-")
    salt = f"{time.time_ns()}-{os.getpid()}"
    suffix = hashlib.sha256(salt.encode()).hexdigest()[:6]
    return (slug[:40] or "i2l") + "-" + suffix


def is_idle(last_seen: float | None, now: float | None = None,
            limit: int = IDLE_STOP_SECONDS) -> bool:
    """Whether a session has gone too long without a heartbeat.

    An unknown heartbeat is *not* idle: assuming a session someone is actively using has
    expired would kill their run. The watchdog only ever acts on evidence.
    """
    if last_seen is None:
        return False
    return (now if now is not None else time.time()) - last_seen >= limit


def human_duration(seconds: float) -> str:
    """`3 hours`, `45 minutes` -- for the confirmation text."""
    if seconds >= 3600:
        hours = seconds / 3600
        return f"{hours:.0f} hour{'s' if hours != 1 else ''}"
    return f"{max(1, round(seconds / 60))} minutes"


def announcement(config: RemoteConfig, *, backend_label: str, weights_gb: float | None,
                 workflow: str | None = None) -> str:
    """What renting this GPU means, before it is rented.

    A remote run spends money and downloads weights onto someone else's disk. `AGENTS.md`'s
    rule about naming a backend, a route and a size applies with more force here, not less,
    because the disk and the bill are not the user's own machine.
    """
    lines = ["", "About to run on a rented GPU:", "",
             f"  backend: {backend_label}",
             f"  route:   Google Colab, {config.accelerator}, session {config.session!r}",
             "  billing: Colab compute units, from your own account"]
    if weights_gb is not None:
        lines.append(f"  disk:    ~{weights_gb:.1f} GB of weights on the VM, on its first run")
    if workflow:
        lines.append(f"  steps:   {workflow}")
    lines.append(f"  idle:    stopped after {human_duration(IDLE_STOP_SECONDS)} without "
                 f"activity" + ("" if config.stop_when_done
                                else ", and kept running when this job finishes"))
    lines += ["", "  Nothing runs on this machine, and nothing is downloaded here.", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------
# Heartbeat: the input to both the watchdog and the idle rule
# ----------------------------------------------------------------------------------------


def state_path() -> Path:
    """Where heartbeats live. Overridable so tests and agents can isolate themselves."""
    explicit = os.environ.get(HEARTBEAT_ENV)
    if explicit:
        return Path(explicit)
    return Path.home() / ".i2l" / "colab.json"


def read_heartbeats(path: Path | None = None) -> dict[str, float]:
    path = path or state_path()
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}


def touch(session: str, path: Path | None = None, now: float | None = None) -> None:
    """Record activity for a session.

    Written through a unique temp file in the same directory, then renamed. A fixed temp
    name raced with a concurrent `forget()` (the supervisor touches while its own finally
    stops another session) and blew up with FileNotFoundError mid-poll, killing a live job.
    """
    path = path or state_path()
    data = read_heartbeats(path)
    data[session] = now if now is not None else time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.stem, suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def forget(session: str, path: Path | None = None) -> None:
    path = path or state_path()
    data = read_heartbeats(path)
    data.pop(session, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def idle_sessions(path: Path | None = None, now: float | None = None) -> list[str]:
    """Sessions past the idle limit, oldest first -- what a watchdog should stop."""
    beats = read_heartbeats(path)
    stale = [(when, name) for name, when in beats.items()
             if is_idle(when, now=now)]
    return [name for _, name in sorted(stale)]


def live_sessions() -> set[str]:
    """The session names the CLI currently has, in one call. Empty when it cannot tell."""
    try:
        listing = run(["sessions"], timeout=120)
    except ColabError:
        return set()
    found = set()
    for line in listing.splitlines():
        line = line.strip()
        if line.startswith("[") and "]" in line:
            found.add(line[1:line.index("]")])
    return found


def shared_session(live: set[str] | None = None) -> str | None:
    """The lab's shared VM when it is live, else None.

    Sharing is the studio's default because one warm VM with every backend installed is
    what makes a second click start instantly; `--no-share` on the supervisor rents the
    old private per-job VMs instead.
    """
    running = live_sessions() if live is None else live
    return SHARED_SESSION if SHARED_SESSION in running else None


def warm_session(backend_id: str, *, path: Path | None = None,
                 now: float | None = None, live: set[str] | None = None) -> str | None:
    """A live, recently-used session for this backend, or None.

    A fresh VM costs 15-30 minutes of installing before it can do anything, which is fine
    for one command and ruinous for a GUI: every click would pay it again. A session that is
    still inside its idle window already has the venv and the weights, so the next job should
    use it.

    Only sessions this lab started for the same backend are candidates -- the name carries
    the backend, because reusing a session whose vendor checkout belongs to a different route
    is how a Hunyuan job would find Hi3D's venv and fail somewhere confusing.
    """
    now = time.time() if now is None else now
    beats = read_heartbeats(path)
    candidates = sorted(
        (when, name) for name, when in beats.items()
        if not is_idle(when, now=now) and name.rsplit("-", 1)[0].endswith(backend_id)
    )
    if not candidates:
        return None
    running = live_sessions() if live is None else live
    for _, name in reversed(candidates):
        if name in running:
            return name
    return None


# ----------------------------------------------------------------------------------------
# The colab CLI, wrapped with a hard bound
# ----------------------------------------------------------------------------------------


class ColabError(RuntimeError):
    """The CLI failed, or did not answer. Callers turn this into a job error."""


class RemoteMissing(ColabError):
    """The remote file is not on the session at all — nothing to fetch, ever.

    Distinct from a failed fetch on purpose: a wrapper that wrote no dump (an unpatched
    build, say) must not fail an otherwise good run, while a dump that exists and will
    not come back must fail it *here*, named — not three steps later as
    "input file does not exist".
    """


def colab_bin() -> str:
    found = shutil.which("colab")
    if not found:
        raise ColabError(
            "the `colab` CLI is not on PATH. Install it with "
            "`uv tool install google-colab-cli` and sign in first."
        )
    return found


def run(args: list[str], *, timeout: float = 180, stdin: str | None = None,
        attempts: int = 1, silence_seconds: float | None = None,
        on_output=None) -> str:
    """One `colab` invocation, bounded. Returns combined output.

    The bound matters: `colab exec` blocks on a fresh VM's kernel, and an unbounded child
    would hang the studio's whole job queue rather than failing one job.

    ``attempts`` is for calls whose first failure is often not a failure: a VM that
    is still bringing its kernel up does not answer at all, and a large transfer dies
    to a connection reset now and then, so callers pass a retry count rather than
    treating the first silence or reset as a dead session. A non-zero exit is retried
    only when the CLI's own words say the transport dropped (a reset, a broken pipe):
    a command that ran and failed has made its side effects, and must not run again.

    ``on_output`` and ``silence_seconds`` serve the one call that runs for the better
    part of an hour (a first-run install, whose remote side prints a beat every 30s
    while it works — see scripts/colab_generate.py). ``on_output`` receives each line
    as it crosses the channel, and a channel that goes fully quiet for
    ``silence_seconds`` is killed and named on the spot rather than waited out: found
    live on 2026-10-28, a dead exec channel hung to its 3600s timeout while the job
    behind it looked busy the whole time. Silence is not retried — twenty missed beats
    in a row already is the answer.
    """
    transient = re.compile(
        r"Connection aborted|ConnectionResetError|BrokenPipeError|Broken pipe|"
        r"Connection reset by peer|Read timed out|ReadTimeout|RemoteDisconnected|"
        r"Connection broken|IncompleteRead|"
        r"urllib3\.util\.retry|NewConnectionError|Failed to establish a new connection",
        re.IGNORECASE)
    last = ""
    for attempt in range(1, max(1, attempts) + 1):
        combined, returncode = _invoke(
            args, stdin=stdin, timeout=timeout,
            silence_seconds=silence_seconds, on_output=on_output)
        if returncode is None:  # the overall timeout hit
            last = (f"`colab {args[0]}` did not answer within {timeout:.0f}s")
            if attempt < attempts:
                time.sleep(20 * attempt)
                continue
            raise ColabError(
                f"{last}. The VM may still be starting; retry, or stop the session and "
                f"start a new one."
            )
        if returncode != 0:
            if attempt < attempts and transient.search(combined):
                last = f"`colab {args[0]}` hit a network blip (attempt {attempt})"
                time.sleep(20 * attempt)
                continue
            raise ColabError(f"`colab {args[0]}` failed:\n{combined.strip()[-1500:]}")
        return combined
    raise ColabError(last or f"`colab {args[0]}` failed")


def _invoke(args: list[str], *, stdin: str | None, timeout: float,
            silence_seconds: float | None, on_output) -> tuple[str, int | None]:
    """One CLI invocation: its output and exit code, or ``(output, None)`` on timeout.

    The plain path stays `subprocess.run` so the many short calls (and a faked CLI in
    tests) keep their exact shape. The streaming path exists to watch a long one: it
    reads line by line in a thread, hands each line to ``on_output``, and kills a
    channel whose silence has outlasted ``silence_seconds`` (raising, not returning —
    see `run`).
    """
    if on_output is None and silence_seconds is None:
        try:
            proc = subprocess.run(
                [colab_bin(), *args],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "", None
        return (proc.stdout or "") + (proc.stderr or ""), proc.returncode

    proc = subprocess.Popen(
        [colab_bin(), *args],
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},  # the CLI must not sit on its output
    )
    chunks: list[str] = []
    received = [time.monotonic()]

    def _read() -> None:
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if not line:
                return
            chunks.append(line)
            received[0] = time.monotonic()
            if on_output is not None:
                on_output(line)

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    timed_out = False
    try:
        if proc.stdin is not None:
            try:
                proc.stdin.write(stdin)
            except OSError:  # the CLI died before reading; its exit tells the story
                pass
            finally:
                proc.stdin.close()
        deadline = time.monotonic() + timeout
        while proc.poll() is None:
            quiet = time.monotonic() - received[0]
            if silence_seconds is not None and quiet > silence_seconds:
                proc.kill()
                raise _silent_channel_error(args, quiet)
            if time.monotonic() >= deadline:
                proc.kill()
                timed_out = True
                break
            time.sleep(0.2)
    finally:
        reader.join(timeout=5)
    if timed_out:
        return "", None
    return "".join(chunks), proc.returncode


def _silent_channel_error(args: list[str], quiet_for: float) -> ColabError:
    session = args[args.index("-s") + 1] if "-s" in args else None
    named = f" on session '{session}'" if session else ""
    return ColabError(
        f"the exec channel{named} went silent for {quiet_for:.0f}s with the job still "
        f"out there -- a dead link, not a slow one. The job was stopped rather than left "
        f"hanging; retry it. If the channel keeps dying, stop the session and let the "
        f"next job start a fresh VM."
    )


def session_exists(session: str) -> bool:
    try:
        listing = run(["sessions"], timeout=120)
    except ColabError:
        return False
    return session in listing


#: The contents API refuses an upload body somewhere between 60 and 100 MB (probed on a
#: live G4 session: 60 MB arrives, 100 MB dies to a broken pipe, a 342 MB clean-up dump
#: to a 400). 48 MB clears the proven floor with margin.
UPLOAD_CHUNK = 48 * 1024 * 1024


def upload(session: str, local: Path, remote: str, *, timeout: float = 600) -> None:
    """Upload `local` to `remote` on the session, however large it is.

    Small files go straight through `colab upload`. A big one is split into parts under
    the contents API's ceiling, the parts are uploaded beside the target, and one exec
    concatenates them and checks the result — so the one file a later step needs (a
    342 MB clean-up dump, say) rides the same flaky link in proven-sized pieces.
    """
    local = Path(local)
    size = local.stat().st_size
    if size <= UPLOAD_CHUNK:
        run(["upload", str(local), remote, "-s", session], timeout=timeout, attempts=3)
        return

    parent, name = remote.rsplit("/", 1)
    digest = hashlib.md5(local.read_bytes()).hexdigest()
    parts = []
    with tempfile.TemporaryDirectory(prefix="i2l-chunk-") as scratch:
        with local.open("rb") as source:
            index = 0
            while chunk := source.read(UPLOAD_CHUNK):
                part = Path(scratch) / f"{name}.part{index:04d}"
                part.write_bytes(chunk)
                parts.append(part.name)
                index += 1
        for part in parts:
            run(["upload", str(Path(scratch) / part), f"{parent}/{part}", "-s", session],
                timeout=timeout, attempts=3)
    # one block, one pass: concatenate, drop the parts, hash the result — a mismatch
    # aborts the exec, which surfaces as a failed job instead of a corrupt replay
    code = (
        "import hashlib, os, shutil\n"
        f"parts = sorted(p for p in os.listdir({parent!r}) if p.startswith({name + '.part'!r}))\n"
        f"with open({remote!r}, 'wb') as out:\n"
        "    for part in parts:\n"
        f"        shutil.copyfileobj(open(os.path.join({parent!r}, part), 'rb'), out)\n"
        f"for part in parts: os.remove(os.path.join({parent!r}, part))\n"
        f"digest = hashlib.md5(open({remote!r}, 'rb').read()).hexdigest()\n"
        f"print('reassembled', len(parts), 'parts ->', digest)\n"
        f"if digest != {digest!r}: raise SystemExit('checksum mismatch after reassembly')\n"
    )
    run(["exec", "-s", session, "--timeout", "600"], stdin=code, timeout=700, attempts=3)


def download(session: str, remote: str, local: Path, *, timeout: float = 600) -> None:
    """Download `remote` from the session to `local`, however large it is.

    Small files go straight through `colab download`. A big one is split into parts on
    the session, the parts are fetched, and they are concatenated and checksum-checked
    here — the mirror of `upload`, because the link is the same in both directions.
    Found the hard way: a head's ~460 MB clean-up dump failed all three one-piece
    attempts twenty minutes after a 208 MB one came back fine, and the silent skip cost
    an hour-old hero run its head.
    """
    local = Path(local)
    probe = run(["exec", "-s", session, "--timeout", "120"],
                stdin="import hashlib, os\n"
                      f"print('MISSING') if not os.path.exists({remote!r}) else "
                      f"print('SIZE', os.path.getsize({remote!r}), "
                      f"hashlib.md5(open({remote!r}, 'rb').read()).hexdigest())\n",
                timeout=180, attempts=3)
    match = re.search(r"SIZE (\d+) ([0-9a-f]{32})", probe or "")
    if match is None:
        if "MISSING" in (probe or ""):
            raise RemoteMissing(f"{remote} is not on the session")
        raise ColabError(f"could not stat {remote} on session {session!r}")
    size, digest = int(match.group(1)), match.group(2)

    def _check() -> None:
        got = (hashlib.md5(local.read_bytes()).hexdigest()
               if local.is_file() else "")
        if got != digest:
            raise ColabError(f"{remote} came back wrong after {local.stat().st_size if local.is_file() else 0} "
                             f"of {size} bytes (checksum mismatch)")

    if size <= UPLOAD_CHUNK:
        run(["download", remote, str(local), "-s", session], timeout=timeout, attempts=3)
        _check()
        return

    parts = (size + UPLOAD_CHUNK - 1) // UPLOAD_CHUNK
    run(["exec", "-s", session, "--timeout", "600"],
        stdin="import os\n"
              f"src, chunk = {remote!r}, {UPLOAD_CHUNK}\n"
              "index = 0\n"
              "with open(src, 'rb') as f:\n"
              "    while block := f.read(chunk):\n"
              "        with open(f'{src}.part{index:04d}', 'wb') as out:\n"
              "            out.write(block)\n"
              "        index += 1\n"
              "print('PARTS', index)\n",
        timeout=700, attempts=3)
    name = remote.rsplit("/", 1)[1]
    with tempfile.TemporaryDirectory(prefix="i2l-dl-") as scratch:
        for index in range(parts):
            run(["download", f"{remote}.part{index:04d}",
                 str(Path(scratch) / f"{name}.part{index:04d}"), "-s", session],
                timeout=timeout, attempts=3)
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.open("wb") as out:
            for index in range(parts):
                out.write((Path(scratch) / f"{name}.part{index:04d}").read_bytes())
    _check()
    # the file is safe here; leftover parts would only cost the VM disk until it stops
    try:
        run(["exec", "-s", session, "--timeout", "120"],
            stdin="import glob, os\n"
                  f"for part in glob.glob({remote + '.part*'!r}): os.remove(part)\n",
            timeout=180)
    except ColabError:
        pass


def ensure_session(config: RemoteConfig, *, log=lambda message: None) -> None:
    """A live session with the right accelerator, allocated if needed."""
    config = config.validated()
    if session_exists(config.session):
        touch(config.session)
        log(f"session {config.session!r} is live")
        return

    last = ""
    for attempt in range(1, ALLOC_ATTEMPTS + 1):
        log(f"allocating {config.accelerator} session {config.session!r} "
            f"(attempt {attempt}/{ALLOC_ATTEMPTS})")
        try:
            output = run(["new", "-s", config.session, "--gpu", config.accelerator],
                         timeout=600)
        except ColabError as exc:
            last = str(exc)
            if "quota" in last.lower() or "entitle" in last.lower():
                raise ColabError(
                    f"{config.accelerator} is not available on this Colab account "
                    f"(quota or entitlement). Colab assigns the bigger cards to paid "
                    f"tiers.\n{last[-600:]}"
                ) from exc
        else:
            if "READY" in output.upper():
                touch(config.session)
                log(f"session {config.session!r} is READY")
                return
            last = output
        if attempt < ALLOC_ATTEMPTS:
            time.sleep(ALLOC_DELAY_SECONDS)
    raise ColabError(
        f"could not allocate a {config.accelerator} session after {ALLOC_ATTEMPTS} "
        f"attempts. Last answer:\n{last[-800:]}"
    )


def stop_session(session: str, *, log=lambda message: None) -> None:
    log(f"stopping session {session!r}")
    try:
        run(["stop", "-s", session], timeout=180)
    except ColabError as exc:
        log(f"could not stop {session!r}: {exc}")
    forget(session)


def stop_idle_sessions(*, log=lambda message: None, now: float | None = None) -> list[str]:
    """Stop every session past the idle limit. Returns the ones acted on."""
    stopped = []
    live = live_sessions()
    for session in idle_sessions(now=now):
        if session not in live:
            forget(session)
            continue
        stop_session(session, log=log)
        stopped.append(session)
    return stopped
