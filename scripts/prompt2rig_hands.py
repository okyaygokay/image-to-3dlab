#!/usr/bin/env python3
"""Hand and finger tracking for the prompt-to-motion route: a clip → per-hand keypoints.

Runs a hand model over the generated clip, assigns detections to the subject's wrists
(from GVHMR's 2D pose cache), rotates the keypoints from camera axes into GVHMR's world
frame, resamples them onto the motion's frames, fills short gaps and smooths. The output
per hand is what `pack_smplx_clip.pack(hands=…)` eats: 21 wrist-relative keypoints per
frame in world axes (Y up) plus a validity mask, MediaPipe/OpenPose order — wrist,
thumb 1-4, index 5-8, middle 9-12, ring 13-16, pinky 17-20.

Two backends: "wilor" (WiLoR-mini, better on the small blurry hands a video model draws)
and "mediapipe" (the fallback; smaller, faster, occasionally earlier). The route picks
WiLoR when its weights are installed and MediaPipe otherwise, exactly like prompt2rig.

Adapted from prompt2rig's `hq/hands.py` and `hq/worker.py`, Apache-2.0, (c) universal-modder.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SIDES = ("R", "L")
MAX_GAP_S = 0.4      # interpolate missing stretches up to this long
SMOOTH_SIGMA = 1.2   # temporal Gaussian (frames at 30fps)


def read_frames(path: Path) -> tuple[list, float]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    return frames, fps


# ---- backends: frame list → per-frame candidates of (kp2d (21,2) px, kp3d (21,3), score) ----

class MediaPipeHands:
    name = "mediapipe"

    def __init__(self, task: Path):
        import mediapipe as mp

        self.mp = mp
        opts = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(task)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3)
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(opts)

    def __call__(self, frames, fps):
        h, w = frames[0].shape[:2]
        out = []
        for t, frame in enumerate(frames):
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = self.detector.detect_for_video(image, int(t * 1000 / fps))
            out.append([(np.array([(p.x * w, p.y * h) for p in lm], np.float32),
                         np.array([(p.x, p.y, p.z) for p in wl], np.float32), hd[0].score)
                        for lm, wl, hd in zip(result.hand_landmarks, result.hand_world_landmarks,
                                              result.handedness)])
        return out


class WiLoRHands:
    name = "wilor"

    def __init__(self, weights: Path):
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )

        pretrained = weights / "pretrained_models"
        if not (pretrained / "MANO_RIGHT.pkl").is_file():
            raise FileNotFoundError(f"WiLoR needs MANO_RIGHT.pkl in {pretrained}")
        # Every file is present locally, so the pipeline never reaches for its mirror repo.
        self.pipe = WiLorHandPose3dEstimationPipeline(
            device=torch.device("cuda"), dtype=torch.float16, verbose=False,
            wilor_pretrained_dir=str(weights), WILOR_MINI_REPO_ID="__offline__")

    def __call__(self, frames, fps, det_batch: int = 16, model_batch: int = 32):
        """Batched WiLoR: same weights and maths as WiLoR-mini's predict(), faster."""
        import torch
        from wilor_mini.utils import utils as wu

        pipeline = self.pipe
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        height, width = rgb[0].shape[:2]
        detections = []
        for i in range(0, len(rgb), det_batch):          # 1) hand boxes, padded batches
            chunk = rgb[i:i + det_batch]
            results = pipeline.hand_detector(chunk + [chunk[-1]] * (det_batch - len(chunk)),
                                             conf=0.3, verbose=False, half=True)
            for r in results[:len(chunk)]:
                boxes = r.boxes
                detections.append(list(zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(),
                                           boxes.conf.cpu().numpy())))
        patches, meta = [], []                           # 2) 256px crops, lefts mirrored
        for t, hands in enumerate(detections):
            for box, cls, conf in sorted(hands, key=lambda d: -d[2])[:4]:
                centre = (box[:2] + box[2:]) / 2.0
                size = 2.5 * float((box[2:] - box[:2]).max())
                right = int(cls > 0.5)
                patch, _ = wu.generate_image_patch_cv2(
                    rgb[t], centre[0], centre[1], size, size, 256, 256, right == 0, 1.0, 0,
                    border_mode=cv2.BORDER_CONSTANT)
                patches.append(patch)
                meta.append((t, centre, size, right, float(conf)))
        out = [[] for _ in frames]
        if not patches:
            return out
        cams, keypoints = [], []                         # 3) the model, batched
        with torch.no_grad():
            for i in range(0, len(patches), model_batch):
                batch = torch.from_numpy(np.stack(patches[i:i + model_batch]))
                result = pipeline.wilor_model(batch.to(pipeline.device, dtype=pipeline.dtype))
                cams.append(result["pred_cam"].float().cpu().numpy())
                keypoints.append(result["pred_keypoints_3d"].float().cpu().numpy())
        cams = np.concatenate(cams)
        keypoints = np.concatenate(keypoints)
        focal = pipeline.FOCAL_LENGTH / pipeline.IMAGE_SIZE * max(width, height)
        for j, (t, centre, size, right, _conf) in enumerate(meta):   # 4) back to the frame
            kp3 = keypoints[j].copy()
            cam = cams[j:j + 1].copy()
            cam[:, 1] *= (2 * right - 1)
            if right == 0:
                kp3[:, 0] = -kp3[:, 0]
            translation = wu.cam_crop_to_full(cam, centre[None], size,
                                              np.array([[width, height]], np.float32), focal)
            kp2 = wu.perspective_projection(
                kp3[None], translation=translation, focal_length=np.array([[focal, focal]]),
                camera_center=np.array([[width / 2.0, height / 2.0]]))[0]
            out[t].append((kp2.astype(np.float32), kp3.astype(np.float32), float(_conf)))
        return out


def backend(weights: Path, task: Path):
    """WiLoR when its weights are installed, MediaPipe otherwise (its landmarker is small)."""
    if (weights / "pretrained_models" / "MANO_RIGHT.pkl").is_file():
        return WiLoRHands(weights)
    return MediaPipeHands(task)


# ---- post-processing -----------------------------------------------------------------------

def assign_to_wrists(detections, wrists):
    """Per-frame candidates + wrists (T, 2, 2) px [right, left] → {side: (kp3d (T,21,3),
    valid (T,))}. Detections far from the wrist they were given are rejected outright."""
    frames = len(detections)
    kp = {s: np.zeros((frames, 21, 3), np.float32) for s in SIDES}
    valid = {s: np.zeros(frames, bool) for s in SIDES}
    for t, candidates in enumerate(detections):
        here = t  # bind the loop variable: the helper below must not see it change

        def distance(candidate, index, at=here):
            return float(np.linalg.norm(candidate[0][0] - wrists[at, index]))

        candidates = sorted(candidates, key=lambda c: -c[2])[:2]
        if not candidates:
            continue
        if len(candidates) == 2:
            a, b = candidates
            pairs = ([(SIDES[0], a), (SIDES[1], b)] if distance(a, 0) + distance(b, 1)
                     <= distance(a, 1) + distance(b, 0) else [(SIDES[0], b), (SIDES[1], a)])
        else:
            c = candidates[0]
            pairs = [(SIDES[0] if distance(c, 0) <= distance(c, 1) else SIDES[1], c)]
        forearm = np.linalg.norm(wrists[here, 0] - wrists[here, 1]) * 0.6 + 40
        for side, (kp2, kp3, _score) in pairs:
            if np.linalg.norm(kp2[0] - wrists[here, SIDES.index(side)]) < forearm:
                kp[side][here], valid[side][here] = kp3, True
    return kp, valid


def resample(kp, valid, src_fps, frames, fps):
    """Linear resample (T,21,3) from src_fps to `frames` frames at `fps`; gaps longer
    than MAX_GAP_S stay invalid."""
    count = len(kp)
    t_src = np.arange(count) / src_fps
    t_dst = np.arange(frames) / fps
    out = np.zeros((frames, 21, 3), np.float32)
    ok = np.zeros(frames, bool)
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return out, ok
    flat = kp[indices].reshape(len(indices), -1)
    for j in range(flat.shape[1]):
        out.reshape(frames, -1)[:, j] = np.interp(t_dst, t_src[indices], flat[:, j])
    nxt = np.searchsorted(t_src[indices], t_dst)
    prev_t = t_src[indices][np.clip(nxt - 1, 0, len(indices) - 1)]
    next_t = t_src[indices][np.clip(nxt, 0, len(indices) - 1)]
    exact = np.isin(np.round(t_dst * src_fps).astype(int), indices)
    ok = exact | ((next_t - prev_t) <= MAX_GAP_S) & (t_dst >= t_src[indices][0]) \
        & (t_dst <= t_src[indices][-1])
    return out, ok


def smooth(kp, ok, sigma: float = SMOOTH_SIGMA):
    """Gaussian smoothing within each valid run — nothing is blended across a gap."""
    if sigma <= 0:
        return kp
    radius = int(3 * sigma)
    window = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    out = kp.copy()
    frames = len(kp)
    i = 0
    while i < frames:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < frames and ok[j]:
            j += 1
        segment = kp[i:j]
        for t in range(len(segment)):
            lo, hi = max(0, t - radius), min(len(segment), t + radius + 1)
            weights = window[lo - t + radius:hi - t + radius]
            out[i + t] = (segment[lo:hi] * weights[:, None, None]).sum(0) / weights.sum()
        i = j
    return out


def aa_to_mat(axis_angle):
    """(N, 3) axis-angle → (N, 3, 3)."""
    from pack_smplx_clip import aa_to_mat as rodrigues

    return rodrigues(np.asarray(axis_angle, np.float32))


def frame_size(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def finish(detections, src_fps, hw, vitpose_px, work_long, orient_world, orient_incam,
           frames: int, fps: float) -> dict:
    """Assign to wrists, rotate camera → world, resample to the motion frames, smooth.

    `orient_world`/`orient_incam` are GVHMR's per-frame pelvis rotations (axis-angle,
    world and camera); their product maps camera axes into the world frame.
    """
    count = len(detections)
    scale = max(hw) / float(work_long)
    pose_frames = len(vitpose_px)
    idx = np.clip(np.round(np.arange(count) / src_fps * fps).astype(int), 0, pose_frames - 1)
    wrists = np.stack([vitpose_px[idx, 10, :2], vitpose_px[idx, 9, :2]], 1) * scale  # [R, L] px
    kp_cam, valid = assign_to_wrists(detections, wrists)

    world = aa_to_mat(orient_world) @ np.transpose(aa_to_mat(orient_incam), (0, 2, 1))
    out = {}
    for side in SIDES:
        k, ok = resample(kp_cam[side], valid[side], src_fps, frames, fps)
        k = k - k[:, :1]                                # wrist-relative
        k = np.einsum("tij,tkj->tki", world[:frames], k)  # into world axes
        k = smooth(k, ok)
        out[side] = {"kp": k, "valid": ok}
    return out
