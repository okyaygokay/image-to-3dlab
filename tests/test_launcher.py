"""The launchers: `./lab` on Mac and Linux, `lab.cmd` on Windows.

These exist to start the app, so their failure mode matters more than most: a raw
"cannot execute: No such file or directory" from a missing venv names a path the reader
does not recognise, and says nothing about the thing actually wrong. That is what a person
who cloned the repo with git or `gh` -- instead of running `install.sh`, which builds the
venv -- sees on their very first try.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SH = shutil.which("sh")

pytestmark = pytest.mark.skipif(SH is None, reason="needs a POSIX shell")


def _run_launcher(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run a copy of ./lab in an empty directory, so no venv can be found."""
    copied = tmp_path / "lab"
    copied.write_text((REPO / "lab").read_text())
    copied.chmod(0o755)
    return subprocess.run([SH, str(copied)], cwd=tmp_path, capture_output=True,
                          text=True, check=False, timeout=60)


def test_a_clone_without_a_venv_explains_itself(tmp_path):
    done = _run_launcher(tmp_path)

    assert done.returncode == 1
    said = done.stderr
    # What is missing, named as the user sees it on disk.
    assert ".venv/bin/python" in said
    # Both ways out: build it, or run the installer that does it for you.
    assert "uv venv" in said
    assert "install.sh" in said
    # And not the raw exec failure this replaced.
    assert "cannot execute" not in said


def test_the_launcher_never_starts_python_it_does_not_have(tmp_path):
    """The guard has to run *before* the exec, not report its failure afterwards."""
    done = _run_launcher(tmp_path)
    assert "viewer/serve.py" not in done.stdout + done.stderr


@pytest.mark.skipif(os.name == "nt", reason="checks the POSIX launcher's shape")
def test_the_launcher_is_executable():
    assert os.access(REPO / "lab", os.X_OK), "./lab must be executable to be run as ./lab"


def test_the_windows_launcher_has_the_same_guard():
    text = (REPO / "lab.cmd").read_text()
    assert ".venv\\Scripts\\python.exe" in text
    # `if not exist` before the exec lines: same ordering the POSIX launcher needs.
    assert text.index("if not exist") < text.index("viewer\\serve.py")
    assert "uv venv" in text and "install.ps1" in text


def test_the_windows_launcher_keeps_its_windows_line_endings():
    """`cmd` reads a batch file line by line; LF-only makes it misbehave, and
    `.gitattributes` says `eol=crlf` for this file.

    Editing it with an ordinary text write put LF on disk and broke the Windows launcher,
    which no Mac-side test would have caught. This catches it on any machine.
    """
    data = (REPO / "lab.cmd").read_bytes()
    assert b"\r\n" in data
    assert b"\n" not in data.replace(b"\r\n", b""), "lab.cmd has a bare LF; it must be CRLF"


def test_the_shell_launcher_stays_unix():
    """The reverse mistake: CRLF in a `#!/bin/sh` script stops it running at all."""
    data = (REPO / "lab").read_bytes()
    assert b"\r\n" not in data, "./lab must use LF line endings"
