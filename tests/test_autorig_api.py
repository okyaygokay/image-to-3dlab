"""The Animate tab's backend: what it offers, what it refuses, and the records it writes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import autorig_api as aa


def _glb(path: Path, payload: bytes = b"glTF") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.fixture
def output(tmp_path):
    root = tmp_path / "output"
    _glb(root / "commercial" / "knight" / "knight.glb")
    _glb(root / "finish" / "knight__finish__20261003-135502" / "knight_40k.glb")
    _glb(root / "finish" / "knight__finish__20261003-135502" / "steps" / "1_retopo.glb")
    _glb(root / "animate" / "knight__rig__20261005-120000" / "knight_rigged.glb")
    _glb(root / "animate" / "knight__rig__20261005-120000" / "knight_walk.glb")
    _glb(root / "props" / "sheet" / "sheet.glb")
    return root


@pytest.fixture
def motions(tmp_path):
    root = tmp_path / "motions"
    root.mkdir()
    (root / "walk.npz").write_bytes(b"x")
    (root / "kick.npz").write_bytes(b"x")
    (root / "presets.json").write_text(json.dumps({"presets": [
        {"id": "walk", "label": "Walk", "seconds": 4},
        {"id": "kick", "label": "Kick", "seconds": 2, "speed": 1.25},
        {"id": "missing", "label": "No file"},
        {"id": "../evil", "label": "Bad id"},
    ]}))
    return root


def test_offers_generated_finished_and_rigged_models_only(output):
    models = {m["path"]: m["kind"] for m in aa.pickable_models(output)}
    assert models == {
        "commercial/knight/knight.glb": "generated",
        "finish/knight__finish__20261003-135502/knight_40k.glb": "finished",
        "animate/knight__rig__20261005-120000/knight_rigged.glb": "rigged",
    }


def test_resolve_refuses_anything_not_offered(output):
    assert aa.resolve_model("commercial/knight/knight.glb", output).is_file()
    for path in ("props/sheet/sheet.glb", "../secret.glb",
                 "animate/knight__rig__20261005-120000/knight_walk.glb"):
        with pytest.raises(ValueError):
            aa.resolve_model(path, output)


def test_presets_need_a_file_and_a_safe_id(motions):
    assert [p["id"] for p in aa.presets(motions)] == ["walk", "kick"]


def test_rig_job_copies_the_source_and_records_both_licences(tmp_path):
    manager = aa.AnimateJobManager(tmp_path / "animate", tmp_path)
    source = {"license": {"name": "MIT"}, "output": {"classification": "commercial"}}
    job = manager.create_rig("My Knight.glb", b"glTF-data", source, "generated")
    assert job.directory.name.startswith("My-Knight__rig__")
    assert (job.directory / "input" / "source.glb").read_bytes() == b"glTF-data"
    assert job.result.name == "My-Knight_rigged.glb"
    (rig, rig_cwd), (clean, _) = job.commands
    assert rig[1:3] == ["demo.py", "--input"] and "--use_transfer" in rig and rig_cwd == aa.SKINTOKENS
    # The clean-up rewrites SkinTokens' result in place, after it.
    assert clean[1].endswith("clean_skin_weights.py") and clean[-1] == str(job.result)
    assert job.record["license"] == {"name": "MIT"}
    assert job.record["classification"] == "commercial"
    assert job.record["rig"]["license"]["name"] == "MIT"
    with pytest.raises(RuntimeError, match="already running"):
        manager.create_rig("other.glb", b"x", None, "uploaded")


def test_animation_needs_a_rigged_model_a_preset_and_a_sane_spread(tmp_path, motions):
    manager = aa.AnimateJobManager(tmp_path / "animate", motions)
    run = tmp_path / "animate" / "knight__rig__20261005-120000"
    rigged = _glb(run / "knight_rigged.glb")
    with pytest.raises(ValueError, match="rigged"):
        manager.create_animation(_glb(tmp_path / "plain.glb"), "walk", 0)
    with pytest.raises(ValueError, match="preset"):
        manager.create_animation(rigged, "missing", 0)
    with pytest.raises(ValueError, match="between"):
        manager.create_animation(rigged, "walk", 90)
    with pytest.raises(ValueError, match="number"):
        manager.create_animation(rigged, "walk", "wide")
    job = manager.create_animation(rigged, "walk", 12)
    assert job.result == run / "knight_walk.glb"
    (command, _), = job.commands
    assert command[command.index("--arm-spread") + 1] == "12"
    assert job.record["motion"]["made_with"] == "Kimodo"
    assert "NVIDIA" in job.record["motion"]["license"]["name"]


def _fake_job(tmp_path, script: str) -> aa.AnimateJob:
    result = tmp_path / "out.glb"
    command = [sys.executable, "-c", script.format(out=result)]
    return aa.AnimateJob("0" * 32, "rig", tmp_path, result, [(command, tmp_path)], 1.0,
                         {"license": {"name": "MIT"}})


def test_a_finished_job_writes_its_provenance(tmp_path):
    job = _fake_job(tmp_path, "open(r'{out}', 'wb').write(b'glb'); print('hello')")
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert job.status == "done"
    assert json.loads(job.result.with_suffix(".provenance.json").read_text())["license"]
    assert job.events[-1]["phase"] == "done"
    assert "hello" in job.log_path.read_text()


def test_a_job_that_writes_nothing_is_an_error_with_its_log(tmp_path):
    job = _fake_job(tmp_path, "import sys; print('boom'); sys.exit(3)")
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert job.status == "error"
    assert "code 3" in job.events[-1]["message"]
    assert "boom" in job.events[-1]["log_tail"]


def test_progress_estimate_never_claims_done():
    assert aa.estimated_pct(0, 70) == 0
    assert aa.estimated_pct(35, 70) == 50
    assert aa.estimated_pct(500, 70) == 95


def test_source_record_prefers_a_sidecar_that_names_a_licence(tmp_path):
    glb = _glb(tmp_path / "m.glb")
    (tmp_path / "m.json").write_text(json.dumps({"seed": 1}))
    assert aa.source_record_for(glb) is None
    (tmp_path / "m.retopo-repaint.json").write_text(json.dumps({"license": {"name": "X"}}))
    assert aa.source_record_for(glb)["license"]["name"] == "X"


def test_friendly_names_drop_run_stamps_and_keep_the_face_count(output):
    names = {m["kind"]: m["name"] for m in aa.pickable_models(output)}
    assert names["finished"] == "knight · 40k"
    assert names["rigged"] == "knight"
    long = aa.friendly_name("a-cute-anime-girl-standing-in-a-perfect-t-pose__20261003__pixal3d")
    assert long.startswith("a cute anime girl") and long.endswith("…") and len(long) == 40


def test_a_failing_first_step_stops_the_chain(tmp_path):
    result = tmp_path / "out.glb"
    marker = tmp_path / "second-ran"
    commands = [([sys.executable, "-c", "import sys; sys.exit(2)"], tmp_path),
                ([sys.executable, "-c", f"open(r'{marker}', 'w')"], tmp_path)]
    job = aa.AnimateJob("1" * 32, "rig", tmp_path, result, commands, 1.0, {})
    aa.run_job(job, aa.AnimateJobManager(tmp_path, tmp_path))
    assert job.status == "error" and not marker.exists()


def test_speed_defaults_to_the_presets_own_and_can_be_overridden(tmp_path, motions):
    run = tmp_path / "animate" / "knight__rig__20261005-120000"
    rigged = _glb(run / "knight_rigged.glb")
    for preset, asked, expected in (("walk", None, "1"), ("kick", None, "1.25"), ("kick", 1.5, "1.5")):
        manager = aa.AnimateJobManager(tmp_path / "animate", motions)
        job = manager.create_animation(rigged, preset, 0, asked)
        (command, _), = job.commands
        assert command[command.index("--speed") + 1] == expected
        assert job.record["motion"]["speed"] == float(expected)
    with pytest.raises(ValueError, match="between"):
        aa.AnimateJobManager(tmp_path / "animate", motions).create_animation(rigged, "walk", 0, 5)


def test_the_shipped_preset_moves_are_all_in_the_repo():
    """The moves live in git: a fresh clone with an empty motions folder offers no moves."""
    shipped = aa.presets()
    assert len(shipped) >= 42, len(shipped)  # 21 moves, takes A and B
    assert {"idle", "walk", "run", "dance"} <= {p["id"] for p in shipped}


# --- the Colab fallback: a machine with no local rigger ---------------------------------------


def test_a_machine_with_no_local_rigger_rigs_on_a_rented_gpu(tmp_path, monkeypatch):
    """SkinTokens is a local install; when it is missing and the Colab CLI is signed in,
    the same job shape runs one supervisor command instead of the venv's two."""
    monkeypatch.setattr(aa, "installed", lambda root=None: False)
    monkeypatch.setattr(aa, "colab_ready", lambda: True)
    manager = aa.AnimateJobManager(tmp_path / "animate", tmp_path)
    job = manager.create_rig("knight.glb", b"glTF", None, "uploaded")
    assert len(job.commands) == 1
    command, _cwd = job.commands[0]
    assert str(command[2]).endswith("colab_generate.py")
    assert command[command.index("--backend") + 1] == "autorig"
    assert command[4] == str(job.result)


def test_a_local_rigger_still_runs_locally_even_when_colab_is_signed_in(tmp_path, monkeypatch):
    monkeypatch.setattr(aa, "installed", lambda root=None: True)
    manager = aa.AnimateJobManager(tmp_path / "animate", tmp_path)
    job = manager.create_rig("knight.glb", b"glTF", None, "uploaded")
    assert len(job.commands) == 2
    assert "demo.py" in job.commands[0][0]


def test_an_animation_falls_back_to_the_same_rented_gpu(tmp_path, motions, monkeypatch):
    run = tmp_path / "animate" / "knight__rig__20261005-120000"
    rigged = _glb(run / "knight_rigged.glb")
    (rigged.with_suffix(".provenance.json")).write_text(json.dumps(
        {"rig": {"model": "knight.glb", "license": {"name": "MIT"}}}))
    monkeypatch.setattr(aa, "installed", lambda root=None: False)
    monkeypatch.setattr(aa, "colab_ready", lambda: True)
    manager = aa.AnimateJobManager(tmp_path / "animate", motions)
    job = manager.create_animation(rigged, "walk", 0)
    command, _ = job.commands[0]
    forwarded = command[command.index("--") + 1:]
    assert forwarded[:2] == ["--preset", "walk"]


# --- a described motion: the prompt box's job ----------------------------------------------


def _rigged_run(tmp_path, motions):
    run = tmp_path / "animate" / "knight__rig__20261005-120000"
    rigged = _glb(run / "knight_rigged.glb")
    (rigged.with_suffix(".provenance.json")).write_text(json.dumps(
        {"rig": {"model": "knight.glb", "license": {"name": "MIT"}}}))
    return rigged, aa.AnimateJobManager(tmp_path / "animate", motions)


def test_a_described_motion_is_one_rented_gpu_run(tmp_path, motions, monkeypatch):
    """One supervisor command for the whole chain — H3, GVHMR, WiLoR, retarget — because
    the video model only exists there. The prompt, the seed and the finishing settings
    ride the forwarded argv; the result lands in the rig's folder, slug-named."""
    rigged, manager = _rigged_run(tmp_path, motions)
    job = manager.create_describe(rigged, "a knight  swinging a sword", 0, 1.25, 6, 7)
    assert len(job.commands) == 1
    command, _cwd = job.commands[0]
    assert str(command[2]).endswith("colab_generate.py")
    assert command[command.index("--backend") + 1] == "prompt2rig"
    forwarded = command[command.index("--") + 1:]
    assert forwarded[:2] == ["--prompt", "a knight swinging a sword"]  # whitespace folded
    assert forwarded[forwarded.index("--seed") + 1] == "7"
    assert forwarded[forwarded.index("--speed") + 1] == "1.25"
    assert job.result.name == "knight_a-knight-swinging-a-sword.glb"
    assert job.result.parent.name == "knight__rig__20261005-120000"
    assert job.expected_seconds == aa.DESCRIBE_SECONDS


def test_the_same_prompt_twice_makes_a_second_take(tmp_path, motions):
    rigged, manager = _rigged_run(tmp_path, motions)
    first = manager.create_describe(rigged, "swing", 0)
    _glb(first.result)  # the first take exists on disk
    manager.finish(first)
    second = manager.create_describe(rigged, "swing", 0)
    assert second.result.name != first.result.name
    assert second.result.stem.startswith(first.result.stem)


def test_a_described_motion_is_validated_like_any_other_setting(tmp_path, motions):
    rigged, manager = _rigged_run(tmp_path, motions)
    for prompt in ("", "  ", "no", "x" * 601):
        with pytest.raises(ValueError):
            manager.create_describe(rigged, prompt, 0)
    with pytest.raises(ValueError):
        manager.create_describe(rigged, "a good prompt", 0, seconds=2)
    with pytest.raises(ValueError):
        manager.create_describe(rigged, "a good prompt", 0, seconds="soon")
    with pytest.raises(ValueError):
        manager.create_describe(rigged, "a good prompt", 31)  # arm spread out of range


def test_a_described_motions_record_names_every_piece_of_the_chain(tmp_path, motions):
    """Four models, four licences: H3's community licence with its regional caveat,
    GVHMR, WiLoR and the SMPL-X body model's research terms, plus the prompt and seed."""
    rigged, manager = _rigged_run(tmp_path, motions)
    job = manager.create_describe(rigged, "a cartwheel to the left", 4, None, 8, 99)
    record = job.record
    assert record["motion"]["prompt"] == "a cartwheel to the left"
    assert record["motion"]["seed"] == 99
    assert record["motion"]["arm_spread_degrees"] == 4
    assert set(record["motion"]["licenses"]) == {"video", "motion", "hands", "body_model"}
    assert "MiniMax" in record["motion"]["licenses"]["video"]["name"]
    assert "USA, EU, UK" in record["motion"]["licenses"]["video"]["note"]
    assert record["rig"]["license"] == {"name": "MIT"}  # the rig's own source carries over


def test_a_describe_seed_comes_out_of_thin_air_when_not_given(tmp_path, motions):
    rigged, manager = _rigged_run(tmp_path, motions)
    one = manager.create_describe(rigged, "a spin", 0)
    manager.finish(one)
    two = manager.create_describe(rigged, "a spin", 0)
    assert one.record["motion"]["seed"] != two.record["motion"]["seed"]


def test_describe_refuses_to_share_the_tab_with_a_running_job(tmp_path, motions):
    rigged, manager = _rigged_run(tmp_path, motions)
    job = manager.create_describe(rigged, "a spin", 0)
    with pytest.raises(RuntimeError):
        manager.create_describe(rigged, "another spin", 0)
    with pytest.raises(RuntimeError):
        manager.create_animation(rigged, "walk", 0)
    manager.finish(job)
    assert manager.create_describe(rigged, "another spin", 0)
