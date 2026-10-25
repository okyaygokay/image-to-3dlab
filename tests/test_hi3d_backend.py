"""Tests for the Hi3D backend's pure half.

Hi3D is the one route here that is not image-to-mesh: it refines a coarse mesh that some
other generator made. That makes two things worth pinning down before any GPU is
involved -- the settings the studio may send, and the licence record, because Hi3D's
commercial conditions (branding, and a threshold) travel with every output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, str(REPO / "viewer")):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from image_to_3dlab import hi3d_backend as hi3d
from image_to_3dlab.provenance import (
    HI3D_BRANDING,
    LICENSES,
    validate_run_policy,
)


def test_defaults_round_trip_through_the_validator():
    validated = hi3d.validate_settings(hi3d.DEFAULTS)
    assert validated == hi3d.DEFAULTS
    assert hi3d.validate_settings(validated) == validated


def test_empty_settings_become_the_defaults():
    # The Generate tab sends {} for an untouched form; that must mean "the defaults",
    # not an error and not a run with missing keys.
    assert hi3d.validate_settings(None) == hi3d.DEFAULTS
    assert hi3d.validate_settings({}) == hi3d.DEFAULTS


@pytest.mark.parametrize("resolution", hi3d.RESOLUTIONS)
def test_every_advertised_resolution_is_accepted(resolution):
    assert hi3d.validate_settings({"resolution": resolution})["resolution"] == resolution


@pytest.mark.parametrize(
    "settings",
    [
        {"resolution": 768},          # not on Hi3D's ladder
        {"resolution": 2048},
        {"steps": 0},
        {"steps": 101},
        {"coarse_steps": 0},
        {"coarse_octree": 300},       # not a power-of-two-ish octree the VAE accepts
        {"seed": -1},
        {"resolution": "big"},
    ],
)
def test_out_of_range_settings_are_refused_with_a_value_error(settings):
    # ValueError specifically: the studio turns it into a 400 with the message shown.
    with pytest.raises(ValueError):
        hi3d.validate_settings(settings)


def test_non_object_settings_are_refused():
    with pytest.raises(ValueError):
        hi3d.validate_settings(["not", "a", "dict"])


def test_the_pass_ladder_matches_the_project_documentation():
    # 512^3 is one pass, 1024 two, 1536 all three. Progress weighting depends on this.
    assert hi3d.passes_for(512) == 1
    assert hi3d.passes_for(1024) == 2
    assert hi3d.passes_for(1536) == 3
    with pytest.raises(ValueError):
        hi3d.passes_for(999)


@pytest.mark.parametrize(
    ("capability", "ok"),
    [((8, 0), True), ((8, 6), True), ((8, 9), True), ((9, 0), True), ((12, 0), True),
     ((7, 5), False), ((6, 1), False)],
)
def test_cards_older_than_ampere_are_refused(capability, ok):
    # There is no CPU path and flash-attn has no fallback kernel, so a pre-Ampere card
    # fails deep inside attention. Say so before the download instead.
    can, reason = hi3d.gpu_can_run(capability)
    assert can is ok
    assert (reason is None) is ok


def test_an_unreadable_card_is_tried_rather_than_refused():
    # Not knowing is not evidence of a bad card; torch will give a real error if it is.
    assert hi3d.gpu_can_run(None) == (True, None)


def test_vram_warning_is_advice_and_stays_quiet_when_it_fits():
    assert hi3d.vram_warning(512, 40 * 1024 ** 3) is None
    assert hi3d.vram_warning(1536, 96 * 1024 ** 3) is None
    warning = hi3d.vram_warning(1536, 20 * 1024 ** 3)
    assert warning is not None and "1536" in warning
    # An unknown free-memory figure must never produce a warning from nothing.
    assert hi3d.vram_warning(1024, None) is None


def test_the_licence_profile_is_conditional_and_states_the_branding():
    profile = LICENSES["hi3d"]
    assert profile.classification == "commercial-conditional"
    conditions = " ".join(profile.conditions)
    assert HI3D_BRANDING in conditions
    assert "200,000" in conditions and "1,000,000" in conditions


def test_the_policy_gate_treats_hi3d_like_other_conditional_routes():
    assert validate_run_policy("hi3d", "showcase", "private", True).classification == (
        "commercial-conditional")
    with pytest.raises(ValueError):
        validate_run_policy("hi3d", "showcase", "private", False)


def test_hi3d_outputs_are_not_cleared_for_worldwide_game_distribution():
    # It refines a Hunyuan mesh, so it inherits Hunyuan's territorial restriction.
    with pytest.raises(ValueError):
        validate_run_policy("hi3d", "game", "worldwide", True)
    # A private game run is fine; the restriction is about distribution.
    assert validate_run_policy("hi3d", "game", "private", True)


def test_provenance_names_all_three_models_and_the_branding(tmp_path):
    image = tmp_path / "picture.png"
    image.write_bytes(b"not really a png, but hashing does not care")

    record = hi3d.build_provenance(
        image_path=image,
        parameters={"resolution": 1024},
        matte_model="birefnet-general-lite",
        backend_revision="13657ad",
        sha256_file=lambda path: "deadbeef",
        now="2026-10-25T00:00:00+00:00",
    )

    assert record["license"]["name"] == LICENSES["hi3d"].license_name
    assert record["branding"] == HI3D_BRANDING
    assert record["input"]["sha256"] == "deadbeef"
    assert record["software"]["backend_revision"] == "13657ad"

    components = {c["component"]: c for c in record["components"]}
    # The coarse generator, both conditioners named by Hi3D, and the remover.
    for name in ("tencent/Hunyuan3D-2mini", "facebook/dinov2-large",
                 "openai/clip-vit-large-patch14", "rembg/birefnet-general-lite"):
        assert name in components, name
    # The coarse stage's territory restriction must survive into the record.
    assert "EU" in components["tencent/Hunyuan3D-2mini"]["note"]


def test_provenance_omits_the_remover_when_none_ran(tmp_path):
    image = tmp_path / "picture.png"
    image.write_bytes(b"x")
    record = hi3d.build_provenance(
        image_path=image,
        parameters={},
        matte_model=None,
        backend_revision=None,
        sha256_file=lambda path: "0" * 64,
        now="2026-10-25T00:00:00+00:00",
    )
    names = [c["component"] for c in record["components"]]
    assert not any(n.startswith("rembg/") for n in names)


def test_manifest_records_the_coarse_stage_it_depended_on(tmp_path):
    manifest = hi3d.build_manifest(
        image="in.png",
        output="out.glb",
        coarse_mesh="coarse.glb",
        settings=hi3d.DEFAULTS,
        artifacts={"faces": 1_000_000},
        timings={"total": 12.5},
        coarse_model="tencent/Hunyuan3D-2mini",
        matte_model="u2net",
    )
    assert manifest["schema_version"] == 1
    assert manifest["backend"] == "hi3d"
    # The coarse mesh is an input, not a detail: without it the record is not reproducible.
    assert manifest["coarse_mesh"] == "coarse.glb"
    assert manifest["coarse_model"] == "tencent/Hunyuan3D-2mini"


def test_the_weights_repo_is_not_the_code_checkout():
    """Real bug, found on a live VM: `from_pretrained` was given the git checkout, so the
    loader read the repository's own files as a 'model' -- 160 KB of config and LICENSE --
    and then stalled in the first transformer with no error at all."""
    repo = hi3d.MODEL_REPO
    assert repo == "hi3d-official/Hi3Dv1.0"
    # A local path would be absolutised by the loader; a Hub id is not a directory.
    assert "/" in repo and not repo.startswith(("/", "."))


def test_materialised_weights_are_real_files_not_links(tmp_path):
    """Real bug, from a live VM: Hi3D's loader refuses every symlink in the model directory.

    A Hugging Face cache is nothing *but* symlinks, so `from_pretrained` on a snapshot dies
    with "model repository has an unexpected symlink" -- and hard-linking the link itself
    produces another link, so the copy has to resolve first.
    """
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    snapshot = tmp_path / "snapshots" / "abc123"
    for relative in hi3d.MODEL_FILES:
        blob = blobs / relative.replace("/", "_")
        blob.write_bytes(b"x" * 32)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(blob)

    built = hi3d.materialise_weights(snapshot, tmp_path / "materialised")

    for relative in hi3d.MODEL_FILES:
        made = built / relative
        assert made.is_file()
        assert not made.is_symlink(), f"{relative} is still a link"
        assert made.stat().st_size == 32


def test_a_misspelt_materialised_copy_is_rebuilt(tmp_path):
    """A half-built copy must be refilled rather than trusted."""
    source = tmp_path / "source"
    for relative in hi3d.MODEL_FILES:
        file = source / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"x" * 16)
    built = hi3d.materialise_weights(source, tmp_path / "out")
    (built / hi3d.MODEL_FILES[0]).unlink()
    again = hi3d.materialise_weights(source, tmp_path / "out")
    assert (again / hi3d.MODEL_FILES[0]).is_file()


def test_the_exported_glb_has_normals_and_a_material(tmp_path):
    """Real bug, from a live VM: the GLB was valid glTF 2.0 with 1M triangles and rendered
    as a black silhouette. trimesh drops NORMAL unless asked, and a mesh with no visual gets
    no material -- neither of which shows in vertex counts, so this reads the file back.
    """
    import json
    import struct
    import sys

    trimesh = pytest.importorskip("trimesh")
    sys.path.insert(0, str(REPO / "scripts"))
    import hi3d_generate

    mesh = trimesh.creation.icosphere(subdivisions=1)
    out = tmp_path / "mesh.glb"
    hi3d_generate.export_mesh(mesh, out)

    data = out.read_bytes()
    assert data[:4] == b"glTF"
    json_length = struct.unpack("<I", data[12:16])[0]
    document = json.loads(data[20:20 + json_length])
    primitives = [p for m in document["meshes"] for p in m["primitives"]]
    attributes = {key for p in primitives for key in p["attributes"]}
    assert "NORMAL" in attributes, "the GLB has no normals and will render black"
    assert document.get("materials"), "the GLB has no material and will render black"
