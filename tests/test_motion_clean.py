"""The packed-clip cleaner: planted feet hold, the floor stops drifting, shimmer comes off.

The fixtures are identity-rotation stick figures (every joint hanging 0.18 m below its
parent), so world joints are exactly the hips track plus a fixed shape: injected drift,
bob and jitter then show up in the output as plainly as they went in.
"""

from __future__ import annotations

import numpy as np

from scripts import motion_clean as mc
from scripts.pack_smplx_clip import aa_to_mat


def _figure(frames: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """(quats [T, 77, 4] identity, joints0 [77, 3]) a stick figure hanging from the hips."""
    joints0 = np.zeros((len(mc.SOMA77_PARENTS), 3))
    for j, parent in enumerate(mc.SOMA77_PARENTS):
        if parent >= 0:
            joints0[j] = joints0[parent] + np.array([0.0, -0.18, 0.02])
    quats = np.zeros((frames, len(mc.SOMA77_PARENTS), 4))
    quats[..., 0] = 1.0
    return quats, joints0


def test_planted_feet_hold_their_ground():
    """A constant slow drift (the skate) is pulled back out of the root: the feet landed
    somewhere and stay there."""
    quats, joints0 = _figure()
    t = np.arange(len(quats)) / (len(quats) - 1)
    hips = np.stack([0.04 * t, np.zeros(len(t)), np.zeros(len(t))], axis=1)
    out, report = mc.clean(quats, joints0, hips, 30.0)
    assert report["planted_frames"] > 0
    assert report["slide_after"] < 0.5                       # cm: the anchors held
    assert np.allclose(out["quats"][0].astype(np.float64), quats[0], atol=1e-3)
    assert np.linalg.norm(out["hips"][0] - hips[0]) < 0.01   # playback is hips-relative anyway


def test_the_floor_stops_wandering():
    """A slow vertical bob (the float) comes off the root's height, never the steps."""
    quats, joints0 = _figure()
    t = np.arange(len(quats)) / (len(quats) - 1)
    hips = np.stack([np.zeros(len(t)), 0.06 * np.sin(2 * np.pi * t * 1.2), np.zeros(len(t))], axis=1)
    _out, report = mc.clean(quats, joints0, hips, 30.0)
    assert report["bob_after"] < report["bob_before"] / 2


def test_jitter_comes_off():
    rng = np.random.default_rng(7)
    _quats, joints0 = _figure(120)
    mats = aa_to_mat(rng.normal(0.0, 0.010, (120, len(mc.SOMA77_PARENTS), 3)))
    noisy = mc.matrices_to_quaternions(mats).reshape(120, -1, 4)
    _out, report = mc.clean(noisy, joints0, np.zeros((120, 3)), 30.0)
    assert report["shimmer_after"] < report["shimmer_before"] * 0.7


def test_a_travelling_motion_is_not_pinned():
    """A clip whose root really travels (a walk) must keep its travel: pinning it would
    make the character march in place."""
    quats, joints0 = _figure()
    t = np.arange(len(quats)) / (len(quats) - 1)
    hips = np.stack([1.5 * t, np.zeros(len(t)), np.zeros(len(t))], axis=1)
    out, report = mc.clean(quats, joints0, hips, 30.0)
    assert report["pinned"] is False
    assert np.allclose(out["hips"][:, 0], hips[:, 0], atol=0.01)


def test_the_cli_reports_and_leaves_the_file_alone(tmp_path):
    quats, joints0 = _figure(60)
    t = np.arange(60) / 59
    hips = np.stack([0.05 * t, np.zeros(60), np.zeros(60)], axis=1)
    path = tmp_path / "motion.npz"
    np.savez_compressed(path, quats=quats.reshape(60, -1), joints0=joints0, hips=hips,
                        fps=np.float32(30.0))
    before = path.read_bytes()
    assert mc.main([str(path), "--report-only"]) == 0
    assert path.read_bytes() == before


def test_finger_flicker_comes_off():
    """48 of the 77 joints are finger bones, and the hand tracker's per-frame noise
    lands there and plays as broken fingers — curling and flicking faster than a real
    hand moves. The finger pass takes it off."""
    rng = np.random.default_rng(3)
    quats, joints0 = _figure(140)
    mats = mc.quaternions_to_matrices(quats.astype(np.float64))
    noise = np.zeros((140, len(mc.SOMA77_PARENTS), 3))
    noise[:, mc.FINGERS] = rng.normal(0.0, 0.05, (140, len(mc.FINGERS), 3))
    noisy = mc.matrices_to_quaternions(aa_to_mat(noise) @ mats).reshape(140, -1, 4)
    _out, report = mc.clean(noisy, joints0, np.zeros((140, 3)), 30.0)
    assert report["finger_after"] < report["finger_before"] * 0.4
    # frame 0 is still joints0's own frame, fingers included
    assert np.allclose(_out["quats"][0].astype(np.float64), noisy[0], atol=1e-3)


def test_the_finger_set_is_the_48_bones_below_the_wrists():
    assert len(mc.FINGERS) == 48
    assert mc.SOMA_INDEX["LeftHand"] not in mc.FINGERS     # the wrist itself stays body
    assert mc.SOMA_INDEX["LeftHandThumb1"] in mc.FINGERS
