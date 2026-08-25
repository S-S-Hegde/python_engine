"""
ace/utils/calibrate.py - Modern MediaPipe Tasks auto-calibration for baseline head pose.
Captures resting pitch, yaw, and roll over a 5-second window to personalize
proctoring thresholds.
"""

import os
import time
import json
from typing import Dict, Any, Optional
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ace.config import Config
from ace.utils.assets import ensure_face_landmarker_model


MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),             # Nose tip (Landmark 1)
        (0.0, -330.0, -65.0),        # Chin (Landmark 152)
        (-225.0, 170.0, -135.0),     # Left eye outer corner (Landmark 33)
        (225.0, 170.0, -135.0),      # Right eye outer corner (Landmark 263)
        (-150.0, -150.0, -125.0),    # Left mouth corner (Landmark 61)
        (150.0, -150.0, -125.0),     # Right mouth corner (Landmark 291)
    ],
    dtype=np.float64,
)

HEAD_POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]


def compute_angles_from_landmarks(landmarks, img_w: int, img_h: int):
    """Estimate Euler angles from landmarks."""
    image_points = []
    for idx in HEAD_POSE_LANDMARKS:
        lm = landmarks[idx]
        image_points.append((lm.x * img_w, lm.y * img_h))

    image_points = np.array(image_points, dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, trans_vec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    rot_matrix, _ = cv2.Rodrigues(rot_vec)

    sy = np.sqrt(rot_matrix[0, 0] * rot_matrix[0, 0] + rot_matrix[1, 0] * rot_matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rot_matrix[2, 1], rot_matrix[2, 2])
        y = np.arctan2(-rot_matrix[2, 0], sy)
        z = np.arctan2(rot_matrix[1, 0], rot_matrix[0, 0])
    else:
        x = np.arctan2(-rot_matrix[1, 2], rot_matrix[1, 1])
        y = np.arctan2(-rot_matrix[2, 0], sy)
        z = 0.0

    pitch = float(np.degrees(x))
    yaw = float(np.degrees(y))
    roll = float(np.degrees(z))

    return pitch, yaw, roll


def run_calibration(
    cap_or_camera=None,
    duration_sec: int = Config.CALIBRATION_DURATION_SEC,
    output_file: str = Config.CALIBRATION_FILE,
    model_path: str = Config.FACE_LANDMARKER_MODEL_PATH,
    show_ui: bool = True,
) -> Dict[str, Any]:
    """
    Executes a calibration session for `duration_sec` seconds using the MediaPipe Tasks API.
    Can accept an OpenCV VideoCapture or a ThreadedCamera instance.
    """
    is_external_cam = cap_or_camera is not None
    cam = cap_or_camera

    if not is_external_cam:
        cam = cv2.VideoCapture(Config.CAMERA_INDEX)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        if not cam.isOpened():
            raise RuntimeError(f"Failed to open camera index {Config.CAMERA_INDEX}")

    # Ensure model asset is present
    validated_model_path = ensure_face_landmarker_model(model_path)

    # Initialize MediaPipe Tasks FaceLandmarker
    base_options = python.BaseOptions(model_asset_path=validated_model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    pitch_samples = []
    yaw_samples = []
    roll_samples = []

    print(f"\n[ACE Calibrate] Starting {duration_sec}-second baseline calibration (MediaPipe Tasks)...")
    print("[ACE Calibrate] Please look straight at your screen and maintain a relaxed neutral posture.")

    start_time = time.time()

    while True:
        # Fetch frame depending on camera type
        if hasattr(cam, "get_frame"):
            ret, frame = cam.get_frame()
        else:
            ret, frame = cam.read()

        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)  # User mirror view
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        results = landmarker.detect(mp_image)

        elapsed = time.time() - start_time
        remaining = max(0.0, duration_sec - elapsed)

        face_detected = False
        cur_p, cur_y, cur_r = None, None, None

        if results and results.face_landmarks:
            face_detected = True
            landmarks = results.face_landmarks[0]
            cur_p, cur_y, cur_r = compute_angles_from_landmarks(landmarks, w, h)
            if cur_p is not None:
                pitch_samples.append(cur_p)
                yaw_samples.append(cur_y)
                roll_samples.append(cur_r)

        if show_ui:
            # Modern UI Overlay
            overlay = frame.copy()
            # Top Banner
            cv2.rectangle(overlay, (0, 0), (w, 80), (15, 15, 25), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Title
            cv2.putText(
                frame,
                f"ACE CALIBRATION: Look Straight at Screen ({remaining:.1f}s)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Subtitle
            status_text = (
                "HOLD STEADY - Calibrating baseline pose..."
                if face_detected
                else "ALIGN YOUR FACE INSIDE TARGET BOX"
            )
            color = (0, 255, 0) if face_detected else (0, 0, 255)
            cv2.putText(
                frame,
                status_text,
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

            # Center Target Box
            box_w, box_h = int(w * 0.40), int(h * 0.52)
            bx1, by1 = (w - box_w) // 2, (h - box_h) // 2
            cv2.rectangle(frame, (bx1, by1), (bx1 + box_w, by1 + box_h), color, 2)

            # Live angle indicators
            if cur_p is not None:
                angle_txt = f"Live Pose -> Pitch: {cur_p:+.1f} deg | Yaw: {cur_y:+.1f} deg | Roll: {cur_r:+.1f} deg"
                cv2.putText(
                    frame,
                    angle_txt,
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

            # Progress Bar
            progress = min(1.0, elapsed / duration_sec)
            cv2.rectangle(frame, (0, 76), (int(w * progress), 80), (0, 255, 0), -1)

            cv2.imshow("ACE Exam Proctor - Auto Calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[ACE Calibrate] Calibration aborted by user.")
                break

        if elapsed >= duration_sec:
            break

    if show_ui:
        cv2.destroyWindow("ACE Exam Proctor - Auto Calibration")

    if not is_external_cam and cam is not None:
        cam.release()

    landmarker.close()

    # Aggregate baseline metrics
    if len(pitch_samples) < 5:
        print("[ACE Calibrate] Warning: Few face samples captured. Using default center baseline.")
        baseline = {
            "baseline_pitch": 0.0,
            "baseline_yaw": 0.0,
            "baseline_roll": 0.0,
            "std_pitch": 5.0,
            "std_yaw": 5.0,
            "std_roll": 5.0,
            "samples": len(pitch_samples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    else:
        baseline = {
            "baseline_pitch": float(np.mean(pitch_samples)),
            "baseline_yaw": float(np.mean(yaw_samples)),
            "baseline_roll": float(np.mean(roll_samples)),
            "std_pitch": float(np.std(pitch_samples)),
            "std_yaw": float(np.std(yaw_samples)),
            "std_roll": float(np.std(roll_samples)),
            "samples": len(pitch_samples),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    # Save to disk
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=4)

    print("\n" + "=" * 60)
    print("ACE CALIBRATION COMPLETE:")
    print(f"  Pitch Baseline: {baseline['baseline_pitch']:+.2f}° (±{baseline['std_pitch']:.2f})")
    print(f"  Yaw Baseline:   {baseline['baseline_yaw']:+.2f}° (±{baseline['std_yaw']:.2f})")
    print(f"  Roll Baseline:  {baseline['baseline_roll']:+.2f}° (±{baseline['std_roll']:.2f})")
    print(f"  Saved to:       {output_file}")
    print("=" * 60 + "\n")

    return baseline


if __name__ == "__main__":
    run_calibration()
