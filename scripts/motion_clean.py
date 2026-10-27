#!/usr/bin/env python3
"""Clean the packed motion clip so it plays planted and smooth on the rig.

Post-processes the `motion.npz` prompt2rig writes: planted feet stop skating, the
floor stops drifting, and the frame-to-frame shimmer from motion recovery comes off.

    python scripts/motion_clean.py motion.npz [--report-only]

The packed form is per-frame **world** rotations, one frame's joint positions and the
hips track — the same three things `kimodo_retarget.load_motion` reads. World joint
centres for every frame follow from them (the same construction the packer used), so
the three fixes work without touching the retargeter:

* **Slide** — a foot is planted when it is at floor height and barely moving. While a
  foot is planted its anchor is where it landed, and the mean breach of the planted
  feet's anchors is subtracted from the hips track (smoothed, so one frame's noise
  cannot jerk the root). The prompt pipeline asks H3 for in-place action, so holding
  the ground here is right; a clip whose root genuinely travels (a walk) is detected
  and its feet are left to fall behind the root as they should.
* **Bob** — the floor itself (the lowest foot) wanders over a clip; the slow wander is
  removed from the hips' height, never the steps.
* **Shimmer** — recovery noise sits frame-to-frame. Rotations are smoothed with a short
  triangular window and re-orthonormalised (SVD), which averages rotations safely.
  Frame 0 is never smoothed: `joints0` is frame 0's posed joints, and the pair must
  stay consistent or the rest pose recovered from them drifts.

Every step reports numbers (planted-foot drift, floor wander, shimmer energy) before
and after, so quality is a measurement, not an opinion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kimodo_retarget import (
    SOMA77_PARENTS,
    SOMA_INDEX,
    matrices_to_quaternions,
    quaternions_to_matrices,
    soma_tpose,
)

#: Ankle joints, left then right.
FOOT = ("LeftFoot", "RightFoot")
#: A foot is at the floor within this many metres of the clip's low foot heights. The
#: floor is the 5th percentile, not the minimum: one penetrating frame would otherwise
#: lift the floor away from every real stance.
CONTACT_HEIGHT = 0.06
#: ... and moving no faster than this horizontally (m/s): recovery jitter alone moves a
#: planted foot ~0.1 m/s, so a tight gate here reads a stance as a shuffle. Anchor
#: re-planting (below) covers the frames a swing slips through.
CONTACT_SPEED = 0.35
#: Net root travel above which the clip is a travelling motion: planted feet are *meant*
#: to fall behind the root, and pinning them would make a walk march in place. Measured
#: on the smoothed track — the raw path is mostly jitter (a standing clip "travels" a
#: metre without moving).
TRAVEL_METRES = 0.50
#: Smoothing widths: the root correction narrower than the floor correction.
DRIFT_SIGMA, FLOOR_SIGMA, ROTATION_WINDOW = 0.10, 0.75, 5


def world_joints(mats: np.ndarray, joints0: np.ndarray, hips: np.ndarray) -> np.ndarray:
    """[T, 77, 3] world joint centres: `joints0` un-posed by frame 0's rotations gives
    the rest offsets, then the same parent-first walk the packer and retargeter use."""
    rest = soma_tpose(mats[:1], joints0[None])          # hips at the origin
    out = np.zeros((len(mats), len(SOMA77_PARENTS), 3))
    out[:, 0] = np.asarray(hips, dtype=np.float64)
    for j in range(1, len(SOMA77_PARENTS)):
        p = SOMA77_PARENTS[j]
        out[:, j] = out[:, p] + np.einsum("tpi,i->tp", mats[:, p], rest[j] - rest[p])
    return out


def feet_of(joints: np.ndarray) -> np.ndarray:
    """[T, 2, 3] the two ankle joints, left then right."""
    return np.stack([joints[:, SOMA_INDEX[name], :] for name in FOOT], axis=1)


def planted(feet: np.ndarray, fps: float) -> np.ndarray:
    """[T, 2] bool: each foot near the clip's floor height and barely moving sideways."""
    floor = np.percentile(feet[:, :, 1], 5)
    speed = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=2) * fps
    speed = np.vstack([speed[:1], speed])               # frame 0 borrows frame 1's speed
    return (feet[:, :, 1] < floor + CONTACT_HEIGHT) & (speed < CONTACT_SPEED)


def _smooth1(track: np.ndarray, fps: float, sigma: float) -> np.ndarray:
    """Gaussian-smoothed [T] track, edges held (mode='edge')."""
    sigma = max(sigma * fps, 1.0)
    x = np.arange(-3 * sigma, 3 * sigma + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    return np.convolve(np.pad(track, pad, mode="edge"), kernel, mode="valid")[: len(track)]


def drift(feet: np.ndarray, contact: np.ndarray) -> np.ndarray:
    """[T, 2] the horizontal breach of each planted foot from its landing anchor (the
    unplanted frames are 0): what a viewer calls skating."""
    out = np.zeros((len(feet), 2))
    for side in (0, 1):
        anchor = None
        for t in range(len(feet)):
            if contact[t, side]:
                pos = feet[t, side, :2]
                if anchor is None or np.linalg.norm(pos - anchor) > 0.25:
                    anchor = pos                        # a new planting, or a real step
                out[t, side] = np.linalg.norm(pos - anchor)
            else:
                anchor = None
    return out


def root_correction(feet: np.ndarray, contact: np.ndarray, hips: np.ndarray, fps: float,
                    travel: float) -> np.ndarray:
    """The hips track with planted feet pinned and the floor's wander removed.

    Only the hips move (the packed form has no per-leg IK), so this is a root fix: the
    mean horizontal breach of the planted feet's anchors, and the slow wander of the
    floor. A clip whose root really travels keeps its travel.

    Two passes, because the smoothing that keeps one noisy frame from jerking the root
    also leaves the correction lagging at the clip's ends; the second pass measures what
    is left and removes that too.
    """
    def correction_of(positions: np.ndarray) -> np.ndarray:
        breach = np.zeros((len(hips), 2))
        for side in (0, 1):
            anchor = None
            for t in range(len(hips)):
                if contact[t, side]:
                    pos = positions[t, side]
                    if anchor is None or np.linalg.norm(pos - anchor) > 0.25:
                        anchor = pos                    # a new planting, or a real step
                    breach[t] += (anchor - pos) * 0.5   # each planted foot: half a vote
                else:
                    anchor = None
        # hold the last correction through airborne stretches so the root does not snap
        for t in range(1, len(breach)):
            if not contact[t].any():
                breach[t] = breach[t - 1]
        return np.stack([_smooth1(breach[:, i], fps, DRIFT_SIGMA) for i in range(2)], axis=1)

    total = np.zeros((len(hips), 2))
    positions = feet[:, :, :2].astype(np.float64).copy()
    for _ in range(2):
        step = correction_of(positions)
        total += step
        positions += step[:, None, :]
    if travel > TRAVEL_METRES:
        total[:] = 0.0
    hips_out = np.array(hips, dtype=np.float64, copy=True)
    hips_out[:, :2] += total

    floor = feet[:, :, 1].min(axis=1)
    wander = _smooth1(floor - floor[0], fps, FLOOR_SIGMA)
    hips_out[:, 1] -= wander - wander[0]
    return hips_out


def smooth_rotations(mats: np.ndarray, window: int = ROTATION_WINDOW) -> np.ndarray:
    """Triangular-window average of nearby rotations, re-orthonormalised by SVD.

    Rotation matrices are continuous in time, so a short average plus the closest
    rotation (polar decomposition) is a safe low-pass. Frame 0 is kept exactly: the
    packed `joints0` is frame 0's posed joints and the pair must stay consistent.
    """
    mats = np.asarray(mats, dtype=np.float64)
    kernel = np.bartlett(max(window, 3))
    kernel = kernel / kernel.sum()
    pad = len(kernel) // 2
    padded = np.pad(mats, ((pad, pad), (0, 0), (0, 0), (0, 0)), mode="edge")
    means = np.zeros_like(mats)
    for k, weight in enumerate(kernel):
        means += weight * padded[k: k + len(mats)]
    u, _, vt = np.linalg.svd(means)                     # batched over frames and joints
    rot = u @ vt
    flip = np.linalg.det(rot) < 0                       # keep rotations, not mirrors
    if flip.any():
        u[flip, :, -1] *= -1
        rot = u @ vt
    rot[0] = mats[0]                                    # joints0's frame, untouched
    return rot


def shimmer(joints: np.ndarray) -> float:
    """Mean energy of the position second difference (mm): recovery noise shows as a
    constant floor here, real motion as peaks."""
    if len(joints) < 3:
        return 0.0
    second = joints[:-2] - 2 * joints[1:-1] + joints[2:]
    return float(np.linalg.norm(second, axis=-1).mean() * 1000)


def clean(quats: np.ndarray, joints0: np.ndarray, hips: np.ndarray, fps: float
          ) -> tuple[dict, dict]:
    """(packed arrays with the fixes applied, a report of before/after numbers)."""
    quats = np.asarray(quats, dtype=np.float64).reshape(len(hips), -1, 4)
    joints0 = np.asarray(joints0, dtype=np.float64)
    hips = np.asarray(hips, dtype=np.float64)
    frames = len(hips)

    raw = quaternions_to_matrices(quats)
    joints_raw = world_joints(raw, joints0, hips)
    feet_raw = feet_of(joints_raw)
    contact = planted(feet_raw, fps)
    before = drift(feet_raw, contact)

    mats = smooth_rotations(raw)
    joints = world_joints(mats, joints0, hips)
    # Net travel of the slow root motion: the raw path is mostly jitter, and gating on
    # it would read a standing clip as a walk.
    slow = np.stack([_smooth1(hips[:, i], fps, FLOOR_SIGMA) for i in range(2)], axis=1)
    travel = float(np.linalg.norm(slow[-1] - slow[0])) if frames > 1 else 0.0
    hips_out = root_correction(feet_of(joints), contact, hips, fps, travel)
    after = drift(feet_of(world_joints(mats, joints0, hips_out)), contact)

    pinned = travel <= TRAVEL_METRES

    def wander(floor: np.ndarray) -> float:
        slow = _smooth1(floor - floor[0], fps, FLOOR_SIGMA)
        return float(slow.max() - slow.min())

    report = {
        "planted_frames": int(contact.sum()),
        "slide_before": round(float(before[contact].mean()) * 100, 1) if contact.any() else 0.0,
        "slide_after": round(float(after[contact].mean()) * 100, 1) if contact.any() else 0.0,
        "bob_before": round(wander(feet_raw[:, :, 1].min(axis=1)) * 100, 1),
        "bob_after": round(wander(feet_of(world_joints(mats, joints0, hips_out))[:, :, 1]
                                  .min(axis=1)) * 100, 1),
        "shimmer_before": round(shimmer(joints_raw), 2),
        "shimmer_after": round(shimmer(joints), 2),
        "travel": round(travel, 2), "pinned": pinned,
    }
    return {"quats": matrices_to_quaternions(mats).astype(np.float16),
            "joints0": joints0.astype(np.float32), "hips": hips_out.astype(np.float32),
            "fps": np.float32(fps)}, report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("motion", type=Path, help="a packed motion.npz")
    parser.add_argument("--report-only", action="store_true",
                        help="print the numbers, write nothing")
    args = parser.parse_args(argv)
    with np.load(args.motion) as data:
        quats, joints0, hips, fps = data["quats"], data["joints0"], data["hips"], data["fps"]
    cleaned, report = clean(quats, joints0, hips, float(fps))
    for key, value in report.items():
        print(f"{key}: {value}")
    if not args.report_only:
        np.savez_compressed(args.motion, **cleaned)
        print(f"cleaned {args.motion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
