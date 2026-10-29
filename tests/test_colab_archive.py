"""The Drive archive: a VM's installs, kept where the VM cannot lose them.

A rented VM is reclaimed and its disk dies; the Google Drive mounted on it is the one
persistent store. These tests pin the archiver's contract: an archive is one tarball
plus a manifest that names its backend and size; a restore brings every piece back and
stamps the install; and the failure rule that makes the whole design safe — a restore
whose pieces do not all appear refuses to write the stamp, so the next step installs
the old way instead of running on a half-restore.
"""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, str(REPO / "scripts")):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import colab_archive as ca


@pytest.fixture()
def lab(tmp_path, monkeypatch):
    """A tiny VM root and Drive, with qwen_image 'installed' under them."""
    root = tmp_path / "base"
    drive = tmp_path / "drive" / "i2l"
    (tmp_path / "drive").mkdir()                       # stands in for a mounted MyDrive
    vendor = root / "content/i2l/vendor/sdcpp"
    vendor.mkdir(parents=True)
    (vendor / "sd-cli").write_bytes(b"#!/bin/sh\n: a built binary\n")
    stamp = root / ca.stamp_name("qwen_image")
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text("ok")

    monkeypatch.setattr(ca, "BASE", root)
    monkeypatch.setattr(ca, "DRIVE_ROOT", drive)
    monkeypatch.setitem(ca.PIECES, "qwen_image", (
        ("content/i2l/vendor/sdcpp",), ()))            # one piece: HF dirs vary by VM
    monkeypatch.setattr(ca.shutil, "disk_usage", lambda _p: type("U", (), {"free": 10 ** 12})())
    return root, drive, stamp


def test_archive_then_restore_is_a_round_trip(lab):
    root, drive, stamp = lab
    # A venv-style absolute symlink: the whole point of the trusted "tar" filter —
    # the "data" filter refused exactly this and aborted the restore partway.
    bin_dir = root / "content/i2l/vendor/sdcpp/bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to("/usr/local/bin/python3.11")
    assert ca.archive("qwen_image", root, drive) == 0

    assert (drive / "qwen_image.tar.gz").is_file()
    assert (drive / "qwen_image.tar.gz").stat().st_size > 0
    manifest = json.loads((drive / "qwen_image.json").read_text())
    assert manifest["backend"] == "qwen_image" and manifest["gb"] >= 0
    assert manifest["pieces"] == [ca.stamp_name("qwen_image"), "content/i2l/vendor/sdcpp"]

    (root / "content/i2l/vendor/sdcpp").rename(root / "content/i2l/vendor/sdcpp.away")
    stamp.unlink()
    assert ca.restore("qwen_image", root, drive, yes=True) == 0

    assert (root / "content/i2l/vendor/sdcpp" / "sd-cli").read_bytes().startswith(b"#!/bin/sh")
    assert (bin_dir / "python").is_symlink()           # the venv's link survived
    assert stamp.is_file()                             # the install looks done again


def test_a_corrupt_archive_leaves_no_stamp_behind(lab):
    """The stamp untars first, so an extraction that dies partway would leave the VM
    looking installed on top of wreckage (found live, 2026-10-29: the next job's
    install tripped over the half-untarred venv). The stamp is removed before the
    untar begins and only ever comes back from a complete, verified restore."""
    root, drive, stamp = lab
    stamp.unlink()                                     # the VM was wiped; the archive is bad
    drive.mkdir(parents=True, exist_ok=True)
    (drive / "qwen_image.tar.gz").write_bytes(b"not a tarball at all")
    assert ca.restore("qwen_image", root, drive, yes=True) == 1
    assert not stamp.exists()


def test_a_restore_that_lands_incomplete_refuses_to_stamp(lab):
    """The whole safety of the design: a piece the tarball did not bring back means no
    stamp, so the wrapper installs the old way rather than running a half-restore."""
    root, drive, stamp = lab
    ca.archive("qwen_image", root, drive)
    stamp.unlink()
    (root / "content/i2l/vendor/sdcpp").rename(root / "content/i2l/vendor/sdcpp.away")

    with tarfile.open(drive / "qwen_image.tar.gz", "r:gz") as tar, \
            tarfile.open(drive / ".tampered", "w:gz") as out:   # the vendor tree dropped
        for name in [n for n in tar.getnames()
                     if not n.startswith("content/i2l/vendor")]:
            out.addfile(tar.getmember(name), tar.extractfile(name))
    (drive / ".tampered").replace(drive / "qwen_image.tar.gz")

    assert ca.restore("qwen_image", root, drive, yes=True) == 1
    assert not stamp.is_file()                         # never look half-installed


def test_a_missing_drive_or_archive_is_a_quiet_nothing(lab):
    root, drive, stamp = lab
    stamp.unlink()
    absent = root.parent / "no-drive"
    assert ca.restore("qwen_image", root, absent, yes=True) == 1   # no Drive: install the old way
    assert ca.archive("qwen_image", root, absent) == 0             # and archiving is skipped
    assert ca.restore("pixal3d", root, drive, yes=True) == 1       # a Drive with nothing on it


def test_an_unmounted_drive_is_never_reported_as_a_missing_archive(lab, capsys):
    """Found live (2026-10-29): a run said 'no pixal3d archive on Google Drive' while a
    12 GB one sat there -- the VM's Drive was simply not mounted, and the person reading
    the log concluded the archive was gone. The two truths now say their own names."""
    root, drive, stamp = lab
    stamp.unlink()
    unmounted = root.parent / "no-drive" / "i2l"      # its parent does not exist either
    assert ca.restore("qwen_image", root, unmounted, yes=True) == 1
    first = capsys.readouterr().out
    assert "not mounted" in first and "no qwen_image archive" not in first
    assert ca.restore("pixal3d", root, drive, yes=True) == 1       # mounted, nothing on it
    second = capsys.readouterr().out
    assert "no pixal3d archive" in second and "not mounted" not in second


def test_every_backend_the_wrapper_knows_has_its_pieces_named():
    """The table covers every backend the wrapper can install — a prefetch or job
    install of any of them can reach the Drive. Entries read off bootstraps still
    fail safe: a wrong entry refuses to restore, and the install runs."""
    import colab_generate as cg
    for backend in cg.PREFETCH_ORDER:
        assert backend in ca.PIECES, f"{backend} has no pieces named"
        vendor, hub = ca.PIECES[backend]
        assert vendor or hub, f"{backend} archives nothing"


def test_autorigs_checkpoints_ride_with_the_vendor_tree():
    """SkinTokens' two checkpoints are symlinks into the Hugging Face cache (found
    live, 2026-10-29: an archive without the cache dir restored dangling links and
    the rig died on torch.load's FileNotFoundError), so the cache is a named piece
    and a restore without it refuses rather than half-installing."""
    _vendor, hub = ca.PIECES["autorig"]
    assert any("models--VAST-AI--SkinTokens" in piece for piece in hub)
