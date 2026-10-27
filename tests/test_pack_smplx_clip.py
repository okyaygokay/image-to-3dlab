"""pack_smplx_clip: GVHMR's SMPL-X motion → the packed clip kimodo_retarget plays.

The packer is the bridge between two conventions that both happen to be Y-up T-poses,
so its output can be checked with the retargeter's own loaders: `load_motion` reads the
packed npz back, and `soma_tpose` un-poses frame 0 into the rest skeleton. A synthetic
SMPL-X model file stands in for the real one (registration-gated), built with the real
kintree and a plausible T-pose.
"""

from __future__ import annotations

from pathlib import Path

import kimodo_retarget
import numpy as np
import pack_smplx_clip as psc
import pytest

# The SMPL-X kintree, parents before children: root, legs, spine, arms, hands, face.
PARENTS = [-1] * 55
for j, p in {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9,
             13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 21: 19,
             52: 12, 53: 15, 54: 15}.items():
    PARENTS[j] = p
for base in (22, 37):  # each hand: index, middle, ring, pinky, thumb, three joints each
    for f in range(5):
        base_joint = base + 3 * f
        PARENTS[base_joint] = base - 2  # the wrist (20 or 21)
        for k in (1, 2):
            PARENTS[base_joint + k] = base_joint + k - 1

FINGER_TIPS = {"Index": 0, "Middle": 3, "Ring": 6, "Pinky": 9, "Thumb": 12}


def rest_joints() -> np.ndarray:
    """A plausible SMPL-X neutral T-pose: Y-up, facing +Z, arms straight out along ±X
    with palms down, fingers continuing outward and slightly spread."""
    j = np.zeros((55, 3))
    j[0] = (0, 0.90, 0)
    j[1], j[2] = (-0.09, 0.85, 0), (0.09, 0.85, 0)
    j[4], j[5] = (-0.09, 0.45, 0), (0.09, 0.45, 0)
    j[7], j[8] = (-0.09, 0.05, 0), (0.09, 0.05, 0)
    j[10], j[11] = (-0.09, 0.02, 0.12), (0.09, 0.02, 0.12)
    j[3], j[6], j[9] = (0, 1.00, 0), (0, 1.10, 0), (0, 1.20, 0)
    j[12], j[15] = (0, 1.38, 0), (0, 1.55, 0.02)
    j[13], j[14] = (-0.04, 1.30, 0), (0.04, 1.30, 0)
    j[16], j[17] = (-0.16, 1.32, 0), (0.16, 1.32, 0)
    j[18], j[19] = (-0.36, 1.32, 0), (0.36, 1.32, 0)
    j[20], j[21] = (-0.54, 1.32, 0), (0.54, 1.32, 0)
    for side, wrist in (("L", 20), ("R", 21)):
        out = -1.0 if side == "L" else 1.0
        for f, spread in enumerate((0.01, 0.0, -0.01, -0.02, 0.03)):
            for k in range(3):
                joint = (22 if side == "L" else 37) + 3 * f + k
                along = out * (0.07 + 0.035 * k)
                z = spread * (0.5 + k) * (0.6 if f == 4 else 1.0)
                j[joint] = j[wrist] + (along, -0.01 * k, z)
    j[52], j[53], j[54] = (0, 1.52, 0.08), (-0.03, 1.58, 0.10), (0.03, 1.58, 0.10)
    return j


@pytest.fixture(scope="module")
def smplx_file(tmp_path_factory: Path) -> Path:
    path = tmp_path_factory.mktemp("smplx") / "SMPLX_NEUTRAL.npz"
    np.savez(path, kintree_table=np.stack([PARENTS, np.arange(55)]), J=rest_joints())
    return path


@pytest.fixture(scope="module")
def smplx_rest(smplx_file: Path) -> np.ndarray:
    return psc.smplx_rest(smplx_file)[0]


def soma_rest(smplx_rest: np.ndarray) -> np.ndarray:
    return psc._soma_rest_joints(smplx_rest)


def load_packed(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g, joints0, hips = kimodo_retarget.load_motion(path)
    return g, joints0, hips


# --------------------------------------------------------------------- the bridge itself


def test_identity_motion_is_the_t_pose(smplx_file: Path, smplx_rest, tmp_path: Path):
    out = tmp_path / "idle.npz"
    np.savez(out, **psc.pack(np.zeros((3, 3)), np.zeros((3, 21, 3)),
                             np.tile([0, 0.9, 0], (3, 1)), smplx_file))
    g, joints0, hips = load_packed(out)
    assert hips[:, 1] == pytest.approx(0.9, abs=1e-3)      # standing at the origin
    identity = np.eye(3)
    assert np.abs(g[0, kimodo_retarget.SOMA_INDEX["Hips"]] - identity).max() < 2e-3
    # every joint sits where its rest joint says, carried by the hips
    base = soma_rest(smplx_rest)
    expect = base - base[kimodo_retarget.SOMA_INDEX["Hips"]] + np.array([0, 0.9, 0])
    assert np.abs(joints0 - expect).max() < 2e-3


def test_unposing_frame_zero_recovers_the_rest_skeleton(smplx_file, smplx_rest, tmp_path):
    """The retargeter rebuilds its T-pose from frame 0; the packer must let it."""
    rng = np.random.default_rng(7)
    go = rng.normal(0, 0.4, (2, 3))
    go[0] = [0, np.pi / 2, 0]                              # facing +X, say
    bp = rng.normal(0, 0.5, (2, 21, 3))
    out = tmp_path / "posed.npz"
    np.savez(out, **psc.pack(go, bp, np.array([[0.1, 0.9, 0.2], [0.2, 0.95, 0.3]]),
                             smplx_file))
    g, joints0, _ = load_packed(out)
    tpose = kimodo_retarget.soma_tpose(g[:1], joints0[None])
    assert np.abs(tpose - (soma_rest(smplx_rest) - soma_rest(smplx_rest)[0])).max() < 5e-3


def test_a_bent_left_elbow_lowers_the_left_wrist(smplx_file, tmp_path):
    """Left arm runs along -X; a +90° turn about Z folds the forearm down (-Y)."""
    bp = np.zeros((1, 21, 3))
    bp[0, 17] = [0, 0, np.pi / 2]                          # joint 18, the left elbow
    out = tmp_path / "bend.npz"
    np.savez(out, **psc.pack(np.zeros((1, 3)), bp, np.array([[0, 0.9, 0]]), smplx_file))
    _, joints0, _ = load_packed(out)
    hand = joints0[kimodo_retarget.SOMA_INDEX["LeftHand"]]
    rest_hand = soma_rest(psc.smplx_rest(smplx_file)[0])[kimodo_retarget.SOMA_INDEX["LeftHand"]]
    forearm = 0.18                                          # wrist minus elbow, along X
    assert hand[1] == pytest.approx(rest_hand[1] - forearm, abs=5e-3)
    assert hand[0] == pytest.approx(rest_hand[0] + forearm, abs=5e-3)


# ------------------------------------------------------------------------------- fingers


def test_untracked_fingers_follow_the_wrist(smplx_file, tmp_path):
    out = tmp_path / "stiff.npz"
    np.savez(out, **psc.pack(np.zeros((1, 3)), np.zeros((1, 21, 3)),
                             np.array([[0, 0.9, 0]]), smplx_file, hands=None))
    g, _, _ = load_packed(out)
    wrist = g[0, kimodo_retarget.SOMA_INDEX["LeftHand"]]
    for name in ("LeftHandIndex1", "LeftHandIndex3", "LeftHandThumb2", "LeftHandPinkyEnd"):
        assert np.abs(g[0, kimodo_retarget.SOMA_INDEX[name]] - wrist).max() < 2e-3


def left_hand_keypoints(smplx_rest, curl: float = 0.0) -> np.ndarray:
    """21 wrist-relative keypoints for the left hand, optionally curled toward the palm.

    Built from the rest joints themselves (MediaPipe/OpenPose order), so an uncurled
    hand must produce identity turns — the same directions as rest."""
    kp = np.zeros((21, 3))
    wrist = smplx_rest[20]
    chains = psc.HAND_CHAINS
    for finger, ids in chains.items():
        offset = FINGER_TIPS[finger]
        base = 22 + offset
        kp[ids[0]] = smplx_rest[base] - wrist
        for k in (1, 2):
            seg = smplx_rest[base + k] - smplx_rest[base + k - 1]
            # fold each further joint `curl` radians toward the palm (-Y at rest)
            angle = curl * k
            kp[ids[k]] = kp[ids[k - 1]] + seg @ psc.aa_to_mat([angle, 0, 0]).T \
                if finger != "Thumb" else kp[ids[k - 1]] + seg
        kp[ids[3]] = kp[ids[2]] + (smplx_rest[base + 2] - smplx_rest[base + 1])
    return kp


def test_a_tracked_hand_poses_its_own_fingers(smplx_file, smplx_rest, tmp_path):
    straight = {"L": {"kp": left_hand_keypoints(smplx_rest, 0.0)[None], "valid": [1]}}
    curled = {"L": {"kp": left_hand_keypoints(smplx_rest, 1.1)[None], "valid": [1]}}
    args = (np.zeros((1, 3)), np.zeros((1, 21, 3)), np.array([[0, 0.9, 0]]), smplx_file)
    np.savez(tmp_path / "straight.npz", **psc.pack(*args, hands=straight))
    np.savez(tmp_path / "curled.npz", **psc.pack(*args, hands=curled))
    g_straight, _, _ = load_packed(tmp_path / "straight.npz")
    g_curled, _, _ = load_packed(tmp_path / "curled.npz")
    wrist = g_straight[0, kimodo_retarget.SOMA_INDEX["LeftHand"]]
    index1 = kimodo_retarget.SOMA_INDEX["LeftHandIndex1"]
    # straight along rest: the measured frames are the rest frames, so no extra turn
    assert np.abs(g_straight[0, index1] - np.eye(3)).max() < 0.05
    # curled: the knuckle carries the fold, and it is a real turn (orthonormal, not wrist)
    folded = g_curled[0, index1]
    assert np.abs(folded @ folded.T - np.eye(3)).max() < 1e-3
    assert np.abs(folded - wrist).max() > 0.1


def test_fingers_never_fold_past_their_limits(smplx_file, smplx_rest, tmp_path):
    """A keypoint hand bent *backwards* at the knuckle is clamped to nearly straight."""
    backwards = left_hand_keypoints(smplx_rest, 0.0)
    across = backwards[5] - backwards[17]
    along = backwards[9]
    palm = psc.palm_normal(along, across, "L")
    seg = backwards[6] - backwards[5]
    hinge = psc.aa_to_mat(-1.2 * palm) @ seg                 # 70° away from the palm
    backwards[6] = backwards[5] + hinge
    clamped = psc._hinge(psc._unit(hinge), psc._unit(seg), palm, knuckle=True)
    cos_fold = float(clamped @ psc._unit(seg))
    assert cos_fold > np.cos(psc.HAND_LIMITS["back_knuckle"] + 1e-6)
    assert float(clamped @ palm) < 1e-9                      # no palm-side component left


# ------------------------------------------------------------------------------ robustness


def test_an_upside_down_clip_is_righted(smplx_file, tmp_path):
    out = tmp_path / "flip.npz"
    np.savez(out, **psc.pack([np.pi, 0, 0], np.zeros((1, 21, 3)),
                             np.array([[0, 0.9, 0]]), smplx_file))
    _, joints0, _ = load_packed(out)
    feet = [kimodo_retarget.SOMA_INDEX[n] for n in ("LeftFoot", "RightFoot")]
    hips = kimodo_retarget.SOMA_INDEX["Hips"]
    assert joints0[hips][1] > joints0[feet, 1].min()         # upright again


def test_somas_extra_joints_inherit_their_parent(smplx_file, tmp_path):
    out = tmp_path / "fill.npz"
    np.savez(out, **psc.pack(np.zeros((1, 3)), np.zeros((1, 21, 3)),
                             np.array([[0, 0.9, 0]]), smplx_file))
    g, _, _ = load_packed(out)
    neck1 = g[0, kimodo_retarget.SOMA_INDEX["Neck1"]]
    for name in ("Neck2", "Jaw", "LeftEye"):
        assert np.abs(g[0, kimodo_retarget.SOMA_INDEX[name]] - neck1).max() < 1e-3 \
            or np.abs(g[0, kimodo_retarget.SOMA_INDEX[name]]).max() < 1e-3


def test_the_cli_packs_a_gvhmr_npz(smplx_file, smplx_rest, tmp_path):
    source = tmp_path / "recovered.npz"
    np.savez(source, global_orient=np.zeros((4, 1, 3)), body_pose=np.zeros((4, 21, 3)),
             transl=np.tile([0, 0.9, 0], (4, 1)))
    hands = tmp_path / "hands.npz"
    kp = left_hand_keypoints(smplx_rest, 0.6)
    np.savez(hands, L_kp=kp[None].repeat(4, 0), L_valid=np.ones(4, int))
    target = tmp_path / "packed.npz"
    assert psc.main([str(source), str(target), "--smplx", str(smplx_file),
                     "--hands", str(hands), "--fps", "30"]) == 0
    with np.load(target) as data:
        assert data["quats"].shape == (4, 77, 4)
        assert float(data["fps"]) == 30.0
        assert data["joints0"].shape == (77, 3)
