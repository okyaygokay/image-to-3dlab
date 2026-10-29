#!/usr/bin/env python3
"""Archive the lab's installed backends to Google Drive, and restore them.

A rented VM is cattle: every fresh one re-pays each backend's download and CUDA
compile, ten to forty minutes a time, and the disk it filled is gone the moment the VM
is reclaimed. The persistent store the lab already trusts is the Google Drive mounted
at ``/content/drive`` (the studio's VM panel mounts it). So every successful install
archives itself there — one tarball per backend, its vendor trees and its Hugging Face
cache dirs and its stamp — and a fresh VM restores instead of installing: untar, verify
every piece landed, write the stamp. ``list`` says what the Drive holds.

Nothing runs FROM Drive, ever: DriveFS is a local cache in front of the cloud — the
fast numbers it shows on a hot file are the cache's, a cold read pays Google's
download speed, and venvs, symlinks and mmapped weights do not survive it. The
tarball is the store; the local disk is the runtime.

The failure rule is the whole safety of the design: a restore whose pieces do not all
appear refuses to write the stamp and says so — the next step then installs the old
way rather than running on a half-restore. A missing Drive, or a Drive with no
archive, is the same quiet answer: nothing to restore, install as before.

    python scripts/colab_archive.py archive qwen_image --root /content/i2l
    python scripts/colab_archive.py restore qwen_image --root /content/i2l --yes
    python scripts/colab_archive.py list --root /content/i2l
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import threading
import time
from pathlib import Path

#: Where the mounted Drive keeps the lab: MyDrive, so it is visible in the owner's
#: own Drive rather than a hidden app folder.
DRIVE_ROOT = Path("/content/drive/MyDrive/i2l")

#: The filesystem the pieces live on. / on the VM; a temp dir under test.
BASE = Path("/")

#: Free-space floor an archive leaves on the VM: the tarball rides DriveFS's cache
#: out to the cloud, and that cache lives on this same disk.
HEADROOM_GB = 30

#: What each backend's install consists of, as paths relative to ``BASE``: the vendor
#: trees the bootstraps build (weights land inside them — the HF cache keeps only
#: metadata stubs, verified live 2026-10-29; hunyuan's bootstrap puts its 8 GB of
#: shape weights there by design), and the stamp is added by the code. The exception
#: is autorig: SkinTokens' two checkpoints are *symlinks into the Hugging Face cache*
#: (found live, 2026-10-29 — an archive without the cache dir restored dangling links
#: and the rig died on torch.load's FileNotFoundError), so its cache dir is a piece
#: too. Every backend the wrapper can install is named — a wrong entry fails safe:
#: the restore refuses, the install runs.
PIECES = {
    "qwen_image": (("content/i2l/vendor/sdcpp",), ()),
    "autorig": (("content/i2l/vendor/SkinTokens",),
                ("root/.cache/huggingface/hub/models--VAST-AI--SkinTokens",)),
    "pixal3d": (("content/i2l/vendor/pixal3d-cpp",), ()),
    "prompt2rig": (("content/i2l/vendor/prompt2rig", "content/i2l/vendor/prompt2rig-driver"), ()),
    "hi3d": (("content/i2l/vendor/hi3d",),
             ("root/.cache/huggingface/hub/models--hi3d-official--Hi3Dv1.0",)),
    "hunyuan_cuda": (("content/i2l/vendor/hunyuan-cuda",), ()),
    "trellis_cuda": (("content/i2l/vendor/trellis-cuda",), ()),
}


def stamp_name(backend: str) -> str:
    return f"content/i2l/.setup-{backend}-stamp"


def piece_paths(backend: str) -> list[str]:
    vendor, hub = PIECES[backend]
    return [stamp_name(backend), *vendor, *hub]


def beat(started: float, note: str) -> threading.Thread:
    """A line every 30 s while tar works: the exec channel's silence watchdog would
    otherwise kill a long archive that is merely quiet (found live, 2026-10-28)."""
    stop = threading.Event()

    def tick() -> None:
        while not stop.wait(30):
            print(f"[archive] still {note} ({time.time() - started:.0f}s elapsed)", flush=True)

    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    return stop


def gb(path: Path) -> float:
    try:
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
        return total / 1e9
    except OSError:
        return 0.0


def drive_ready(drive: Path) -> bool:
    return drive.parent.is_dir()


def archive(backend: str, root: Path, drive: Path) -> int:
    """Tar the backend's pieces onto the Drive, with a manifest naming them."""
    if backend not in PIECES:
        print(f"[archive] {backend} has no pieces named for it yet; not archived")
        return 0
    if not drive_ready(drive):
        print("[archive] Google Drive is not mounted on this VM; nothing to do "
              "(mount it from the studio's VM panel)")
        return 0
    pieces = [p for p in piece_paths(backend) if (BASE / p).exists()]
    if len(pieces) != len(piece_paths(backend)):
        print(f"[archive] {backend} is not fully installed (missing "
              f"{sorted(set(piece_paths(backend)) - set(pieces))}); nothing to archive")
        return 0
    free = shutil.disk_usage(str(BASE)).free / 1e9
    size = sum(gb(BASE / p) for p in pieces[1:])
    if free - size * 0.35 < HEADROOM_GB:            # the cache's share while it flushes
        print(f"[archive] not enough headroom to archive {backend} "
              f"({free:.0f} GB free, {size:.1f} GB to tar)")
        return 0
    drive.mkdir(parents=True, exist_ok=True)
    target = drive / f"{backend}.tar.gz"
    tmp = drive / f".{backend}.tar.gz.part"
    started = time.time()
    stop = beat(started, f"archiving {backend}")
    try:
        with tarfile.open(tmp, "w:gz") as tar:
            for piece in pieces:                    # arcname keeps the base-relative path
                tar.add(str(BASE / piece), arcname=piece)
    finally:
        stop.set()
    tmp.replace(target)
    (drive / f"{backend}.json").write_text(json.dumps({
        "backend": backend,
        "gb": round(os.path.getsize(target) / 1e9, 2),
        "unpacked_gb": round(size, 1),
        "archived": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "pieces": pieces,
    }, indent=2), encoding="utf-8")
    print(f"[archive] {backend} -> {target} "
          f"({os.path.getsize(target) / 1e9:.1f} GB gz, {size:.1f} GB unpacked, "
          f"{time.time() - started:.0f}s)", flush=True)
    return 0


def restore(backend: str, root: Path, drive: Path, *, yes: bool) -> int:
    """Untar the backend from the Drive, and stamp it only if every piece landed."""
    if backend not in PIECES:
        print(f"[archive] {backend} has no pieces named for it yet; "
              f"installing the old way")
        return 1
    target = drive / f"{backend}.tar.gz"
    if not drive_ready(drive) or not target.is_file():
        print(f"[archive] no {backend} archive on Google Drive; installing the old way")
        return 1
    stamp = BASE / stamp_name(backend)
    if stamp.exists():
        print(f"[archive] {backend} is already installed here; nothing to restore")
        return 0
    started = time.time()
    stop = beat(started, f"restoring {backend}")
    # The stamp goes first: an interrupted restore must never leave the VM looking
    # installed (found live, 2026-10-29 — a stamp from a half-untarred archive sent the
    # next job into an install that tripped over the wreckage).
    stamp.unlink(missing_ok=True)
    try:
        with tarfile.open(target, "r:gz") as tar:
            # filter="tar" — the fully trusted one — because this archive is the lab's
            # own, written by this same code: a venv's bin/python is an absolute
            # symlink to the interpreter it was built on, and the "data" filter
            # refuses exactly that, aborting the restore partway (found live, same
            # day). On a python without the filter API, plain extract is the same.
            try:
                tar.extractall(str(BASE), filter="tar")
            except TypeError:                       # a python without the filter API
                tar.extractall(str(BASE))
    except tarfile.TarError as exc:
        print(f"[archive] the {backend} archive would not unpack ({exc}); "
              f"installing the old way", file=sys.stderr)
        return 1
    finally:
        stop.set()
    missing = [p for p in piece_paths(backend) if not (BASE / p).exists()]
    if missing:
        stamp.unlink(missing_ok=True)               # never look half-installed
        print(f"[archive] the {backend} archive was incomplete ({missing}); "
              f"installing the old way", file=sys.stderr)
        return 1
    print(f"[archive] {backend} restored from Google Drive "
          f"({os.path.getsize(target) / 1e9:.1f} GB gz, {time.time() - started:.0f}s)",
          flush=True)
    return 0


def listing(drive: Path) -> int:
    if not drive_ready(drive):
        print("[archive] Google Drive is not mounted on this VM")
        return 0
    if not drive.is_dir():
        print(f"[archive] nothing archived yet (nothing at {drive})")
        return 0
    for manifest in sorted(drive.glob("*.json")):
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
            print(f"[archive] {record['backend']}: {record['gb']} GB "
                  f"(gz), archived {record['archived']}")
        except (OSError, ValueError, KeyError):
            print(f"[archive] {manifest.stem}: unreadable manifest")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("archive", "restore", "list"))
    parser.add_argument("backend", nargs="?", default=None)
    parser.add_argument("--root", type=Path, default=Path("/content/i2l"))
    parser.add_argument("--drive", type=Path, default=DRIVE_ROOT)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.what == "list":
        return listing(args.drive)
    if not args.backend:
        raise SystemExit("which backend?")
    if args.what == "archive":
        return archive(args.backend, args.root, args.drive)
    return restore(args.backend, args.root, args.drive, yes=args.yes)


if __name__ == "__main__":
    raise SystemExit(main())
