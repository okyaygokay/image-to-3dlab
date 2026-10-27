#!/usr/bin/env python3
"""Pack GVHMR's SMPL-X motion into the clip format `kimodo_retarget.py` plays.

    python scripts/pack_smplx_clip.py recovered.npz packed.npz --smplx SMPLX_NEUTRAL.npz

The Animate tab plays Kimodo clips, and `kimodo_retarget.load_motion` already reads a
small packed form: per-frame **world** rotations (as quaternions, in SOMA77 order), one
frame's joint positions and the hips track. GVHMR recovers SMPL-X parameters — local
rotations away from the SMPL-X rest pose — so this module is the bridge:

* body: forward-kinematics `global_orient` + `body_pose` along SMPL-X's tree into world
  rotations and rigid joint positions. Both rest poses are T-poses, so a joint's world
  rotation *is* the "away from the T-pose" turn the retargeter wants, and un-posing
  frame 0 (`soma_tpose`) recovers the rest joints exactly.
* fingers: GVHMR leaves the hands at rest. Finger tracking (WiLoR or MediaPipe, see
  `prompt2rig_colab.py`) supplies 21 wrist-relative keypoints per hand per frame, in
  world axes, MediaPipe/OpenPose order. Each finger segment's world frame is rebuilt
  from its measured direction — with the anatomical clamps prompt2rig found necessary
  (fingers fold toward the palm, never backwards, and only so far) — and emitted as a
  turn away from the segment's rest frame. Untracked frames let the fingers follow the
  wrist rigidly, the same trick as the retargeter's `--stiff-fingers`.
* joints SOMA has and SMPL-X does not (a second neck bone, the jaw, eyes, finger tips)
  inherit their parent's turn, which the world-space retarget absorbs for free.

Pure numpy; `kimodo_retarget` supplies the SOMA77 joint table and the quaternion helper,
so the two sides of the hand-off cannot drift.

The finger maths is adapted from prompt2rig (`hq/hands.py`, `public/retarget.js`),
Apache-2.0, (c) universal-modder, with thanks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kimodo_retarget import SOMA77, SOMA77_PARENTS, SOMA_INDEX, matrices_to_quaternions

# SMPL-X body joints that have a SOMA name; everything else is filled or driven by hands.
# SMPL-X order: 0 pelvis, 1 L_hip, 2 R_hip, 3 spine1, 4 L_knee, 5 R_knee, 6 spine2,
# 7 L_ankle, 8 R_ankle, 9 spine3, 10 L_foot, 11 R_foot, 12 neck, 13 L_collar,
# 14 R_collar, 15 head, 16 L_shoulder, 17 R_shoulder, 18 L_elbow, 19 R_elbow,
# 20 L_wrist, 21 R_wrist, 22-36 left hand, 37-51 right hand, 52 jaw, 53-54 eyes.
SMPLX_TO_SOMA = {
    0: "Hips", 3: "Spine1", 6: "Spine2", 9: "Chest", 12: "Neck1", 15: "Head",
    13: "LeftShoulder", 16: "LeftArm", 18: "LeftForeArm", 20: "LeftHand",
    14: "RightShoulder", 17: "RightArm", 19: "RightForeArm", 21: "RightHand",
    1: "LeftLeg", 4: "LeftShin", 7: "LeftFoot", 10: "LeftToeBase",
    2: "RightLeg", 5: "RightShin", 8: "RightFoot", 11: "RightToeBase",
}
SMPLX_WRIST = {"L": 20, "R": 21}
SMPLX_HAND_BASE = {"L": 22, "R": 37}

# MediaPipe/OpenPose hand order: 0 wrist, thumb 1-4, index 5-8, middle 9-12,
# ring 13-16, pinky 17-20. Four points per finger = three measurable segments.
HAND_CHAINS = {
    "Thumb": (1, 2, 3, 4), "Index": (5, 6, 7, 8), "Middle": (9, 10, 11, 12),
    "Ring": (13, 14, 15, 16), "Pinky": (17, 18, 19, 20),
}
# Where each finger's three SMPL-X joints sit inside the hand block (index, middle,
# ring, pinky, thumb — three joints each), and the same finger in keypoint order.
FINGER_OFFSET = {"Index": 0, "Middle": 3, "Ring": 6, "Pinky": 9, "Thumb": 12}
# Anatomical clamps (prompt2rig's HAND_LIMITS, in radians): knuckles fold ~90° and never
# more than ~20° backwards; the middle joints a little further. The fold plane is free.
HAND_LIMITS = {"knuckle": 1.6, "joint": 1.75, "back_knuckle": 0.35, "back_joint": 0.05}


def aa_to_mat(aa: np.ndarray) -> np.ndarray:
    """(…, 3) axis-angle → (…, 3, 3) rotation matrices (Rodrigues)."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    safe = np.where(theta < 1e-8, 1.0, theta)
    k = aa / safe
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    return np.eye(3) + np.sin(theta)[..., None] * K + (1 - np.cos(theta)[..., None]) * (K @ K)


def smplx_rest(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(rest joints [55, 3], parents [55]) from an SMPL-X model file, in metres, Y-up.

    `J` directly when the file carries it (some mirrors do), else regressed from the
    template the way the model itself does. The kintree is authoritative either way.
    """
    with np.load(npz_path) as data:
        kintree = np.asarray(data["kintree_table"], dtype=int)
        if "J" in data.files:
            joints = np.asarray(data["J"], dtype=np.float64)
        else:
            regressor = np.asarray(data["J_regressor"], dtype=np.float64)
            joints = regressor @ np.asarray(data["v_template"], dtype=np.float64)
    return joints, np.asarray(kintree[0], dtype=int)


def fk_world(rest: np.ndarray, parents: np.ndarray, global_orient: np.ndarray,
             body_pose: np.ndarray, frames: int | None = None
             ) -> tuple[np.ndarray, np.ndarray]:
    """SMPL-X parameters → (world rotations [T, 55, 3, 3], rigid joints [T, 55, 3]).

    `global_orient` is [T, 3] or [T, 1, 3] and `body_pose` [T, n, 3] with n ≥ 21 (GVHMR's
    hands are zeros), both axis-angle, both away from the rest pose — GVHMR's
    `smpl_params_world`. Joint centres move rigidly, the same convention Kimodo's own
    clips carry, so un-posing any frame recovers the rest pose. `frames`, when given, is
    the authoritative count (the transl track's length).
    """
    go = np.asarray(global_orient, dtype=np.float64).reshape(-1, 3)
    frames = len(go) if frames is None else frames
    assert len(go) == frames, "global_orient and transl disagree on the frame count"
    local = np.zeros((frames, len(parents), 3, 3))
    local[:, 0] = aa_to_mat(go)
    bp = np.asarray(body_pose, dtype=np.float64).reshape(frames, -1, 3)[:, :21]
    local[:, 1:22] = aa_to_mat(bp)
    world = np.zeros_like(local)
    joints = np.zeros((frames, len(parents), 3))
    world[:, 0] = local[:, 0]
    for j in range(1, len(parents)):  # SMPL-X's kintree lists parents before children
        p = int(parents[j])
        world[:, j] = world[:, p] @ local[:, j]
        joints[:, j] = joints[:, p] + world[:, p] @ (rest[j] - rest[p])
    return world, joints


def _basis(x: np.ndarray, y_hint: np.ndarray) -> np.ndarray:
    """Orthonormal [3, 3] frame: X along `x`, Y as near `y_hint` as possible."""
    x = x / (np.linalg.norm(x) + 1e-12)
    z = np.cross(x, y_hint)
    z = z / (np.linalg.norm(z) + 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def palm_normal(along: np.ndarray, across: np.ndarray, side: str) -> np.ndarray:
    """Palm-side normal from the knuckle line (`along`) and the index→pinky line
    (`across`), mirrored between hands so it always points the way fingers fold."""
    palm = -np.cross(along, across) if side == "R" else np.cross(along, across)
    return _unit(palm)


def _hinge(dir_: np.ndarray, parent_dir: np.ndarray, palm: np.ndarray,
           knuckle: bool) -> np.ndarray:
    """A finger segment direction, folded only toward the palm and only so far."""
    along = float(dir_ @ parent_dir)
    perp = dir_ - parent_dir * along
    if np.linalg.norm(perp) < 1e-10:
        return _unit(dir_)
    bend = np.arctan2(float(np.linalg.norm(perp)), along)
    if float(perp @ palm) < 0:  # bending away from the palm: nearly straight, at most
        bend = min(bend, HAND_LIMITS["back_knuckle" if knuckle else "back_joint"])
    bend = min(bend, HAND_LIMITS["knuckle" if knuckle else "joint"])
    perp = _unit(perp)
    return _unit(parent_dir * np.cos(bend) + perp * np.sin(bend))


def hand_turns(rest: np.ndarray, hand_base: int, kp: np.ndarray,
               side: str) -> dict[str, np.ndarray]:
    """World turns for one hand's 15 SOMA finger bones on one frame, keyed by SOMA name.

    `kp` is 21 wrist-relative keypoints in world axes. Each segment's turn is its
    measured frame against its rest frame; direction-only, so hand size never matters.
    """
    wrist = rest[SMPLX_WRIST[side]]
    along = kp[9] - kp[0]                       # to the middle-finger knuckle
    across = kp[5] - kp[17]                     # index knuckle minus pinky knuckle
    palm = palm_normal(along, across, side)
    rest_along = rest[hand_base + 3] - wrist    # middle-finger MCP
    rest_across = rest[hand_base + 0] - rest[hand_base + 9]
    rest_palm = palm_normal(rest_along, rest_across, side)
    prefix = "Left" if side == "L" else "Right"

    turns: dict[str, np.ndarray] = {}
    parent_dir = _unit(along)
    for finger, ids in HAND_CHAINS.items():
        offset = FINGER_OFFSET[finger]
        for k in range(3):                      # three segments per finger
            seg = kp[ids[k + 1]] - kp[ids[k]]
            dir_ = _unit(seg) if np.linalg.norm(seg) > 1e-9 else parent_dir
            if finger != "Thumb":
                dir_ = _hinge(dir_, parent_dir, palm, k == 0)
            # rest segment k: joint k → joint k+1; the tip extends the last one
            a, b = offset + k, offset + min(k + 1, 2)
            rest_seg = rest[hand_base + b] - rest[hand_base + a] if k < 2 \
                else rest[hand_base + 2] - rest[hand_base + 1]
            turns[f"{prefix}Hand{finger}{k + 1}"] = \
                _basis(dir_, palm) @ _basis(_unit(rest_seg), rest_palm).T
            parent_dir = _unit(dir_)
    return turns


def _soma_rest_joints(rest: np.ndarray) -> np.ndarray:
    """SMPL-X rest joints in SOMA77 order, with the joints SOMA adds filled in by
    extending their parent's rest direction — a short neutral stub is enough, the
    retargeter only uses these to undo frame 0."""
    soma_rest = np.full((len(SOMA77), 3), np.nan)
    for smplx_joint, soma_name in SMPLX_TO_SOMA.items():
        soma_rest[SOMA_INDEX[soma_name]] = rest[smplx_joint]
    for i in range(len(SOMA77)):  # SOMA77 lists parents before children
        if np.isnan(soma_rest[i]).any():
            parent = SOMA77_PARENTS[i]
            stub = soma_rest[parent] + np.array([0.0, 0.0, 0.02])
            soma_rest[i] = stub
    return soma_rest


def pack(global_orient: np.ndarray, body_pose: np.ndarray, transl: np.ndarray,
         smplx_path: Path, hands: dict | None = None, fps: float = 30.0) -> dict:
    """Arrays for `np.savez_compressed`, in `kimodo_retarget.load_motion`'s packed shape.

    `hands` is per side: {"R": {"kp": [T, 21, 3], "valid": [T]}, "L": …} — wrist-relative
    world keypoints (see `prompt2rig_colab.py`). Missing hands, or missing frames, ride
    the wrist.
    """
    rest, smplx_parents = smplx_rest(smplx_path)
    frames = len(transl)
    transl = np.asarray(transl, dtype=np.float64).reshape(frames, 3)
    world, _ = fk_world(rest, smplx_parents, global_orient, body_pose, frames)

    # Every SOMA joint starts as its SMPL-X source's turn, or its SOMA parent's.
    turns = np.zeros((frames, len(SOMA77), 3, 3))
    filled = np.zeros(len(SOMA77), dtype=bool)
    for smplx_joint, soma_name in SMPLX_TO_SOMA.items():
        turns[:, SOMA_INDEX[soma_name]] = world[:, smplx_joint]
        filled[SOMA_INDEX[soma_name]] = True
    for i in range(len(SOMA77)):  # parents first: Neck2 after Neck1, tips after tips
        if not filled[i]:
            turns[:, i] = turns[:, SOMA77_PARENTS[i]]
            filled[i] = True
    # Fingers, where tracking has them: measured segment turns instead of the wrist's.
    if hands:
        for side in ("L", "R"):
            track = hands.get(side) or {}
            kp = np.asarray(track.get("kp") if track.get("kp") is not None
                            else np.zeros((frames, 21, 3)), dtype=np.float64)
            valid_raw = track.get("valid")
            valid = np.asarray(valid_raw if valid_raw is not None else np.zeros(frames),
                               dtype=bool)[:frames]
            valid = valid[:len(kp)] & (np.linalg.norm(kp.reshape(len(kp), -1), axis=1) > 1e-6)
            for t in np.flatnonzero(valid)[:frames]:
                for name, turn in hand_turns(rest, SMPLX_HAND_BASE[side], kp[t], side).items():
                    turns[t, SOMA_INDEX[name]] = turn
            # The 4th SOMA finger joint and the tips follow their finger's last turn.
            prefix = "Left" if side == "L" else "Right"
            for finger in HAND_CHAINS:
                last = SOMA_INDEX[f"{prefix}Hand{finger}3"]
                for extra in (f"{prefix}Hand{finger}4", f"{prefix}Hand{finger}End"):
                    if extra in SOMA_INDEX:
                        turns[:, SOMA_INDEX[extra]] = turns[:, last]

    # Frame-0 joint positions posed by exactly these turns, so un-posing recovers `rest`.
    soma_rest = _soma_rest_joints(rest)
    joints0 = np.zeros((len(SOMA77), 3))
    joints0[0] = transl[0]
    for i in range(1, len(SOMA77)):  # parents first
        parent = SOMA77_PARENTS[i]
        joints0[i] = joints0[parent] + turns[0, parent] @ (soma_rest[i] - soma_rest[parent])

    hips = np.asarray(transl, dtype=np.float64)
    # GVHMR's world is gravity-aligned Y-up; if a clip ever lands upside down (a different
    # convention upstream), flip it here rather than animate a character on the ceiling.
    feet = [SOMA_INDEX[n] for n in ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase")]
    if np.mean(joints0[feet, 1]) > joints0[SOMA_INDEX["Hips"], 1]:
        flip = np.diag([1.0, -1.0, -1.0])
        turns = flip @ turns
        joints0 = joints0 @ flip.T
        hips = hips @ flip.T

    return {
        "quats": matrices_to_quaternions(turns.reshape(-1, 3, 3)).reshape(frames, -1, 4)
                      .astype(np.float16),
        "joints0": joints0.astype(np.float32),
        "hips": hips.astype(np.float32),
        "fps": np.float32(fps),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path,
                        help="GVHMR-style .npz: global_orient, body_pose, transl")
    parser.add_argument("target", type=Path, help="packed .npz for kimodo_retarget.py")
    parser.add_argument("--smplx", required=True, type=Path, help="SMPLX_NEUTRAL.npz")
    parser.add_argument("--hands", type=Path, help="optional hands .npz: <side>_kp/<side>_valid")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args(argv)

    with np.load(args.source) as data:
        go, bp, tr = data["global_orient"], data["body_pose"], data["transl"]
    hands = None
    if args.hands and args.hands.is_file():
        with np.load(args.hands) as data:
            hands = {s: {"kp": data[f"{s}_kp"], "valid": data[f"{s}_valid"]}
                     for s in ("L", "R") if f"{s}_kp" in data.files}
    packed = pack(go, bp, tr, args.smplx, hands, fps=args.fps)
    args.target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.target, **packed)
    print(f"{args.target}: {len(packed['hips'])} frames @ {args.fps:g}fps, {len(SOMA77)} joints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
