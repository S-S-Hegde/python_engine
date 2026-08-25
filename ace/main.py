"""
ace/main.py - Main Orchestrator and Standalone Proctoring Engine.
Integrates threaded camera streaming, asynchronous YOLOv10 object detection,
MediaPipe FaceLandmarker + PoseLandmarker, 60-second timer, and debounced screenshot logging.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ace.config import Config
from ace.core.camera import ThreadedCamera
from ace.core.detector import YOLODetectorWorker
from ace.core.tracker import FaceProctorTracker
from ace.core.logger import ViolationLogger
from ace.utils.calibrate import run_calibration


def draw_hud(
    frame: np.ndarray,
    tracker_results: dict,
    yolo_results: dict,
    active_warnings: list,
    fps: float,
    baseline: dict,
    time_remaining: int,
    exam_finished: bool,
):
    """Renders a sleek HUD overlay on the video frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 1. Top Navigation Bar
    cv2.rectangle(overlay, (0, 0), (w, 60), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    is_violation = len(active_warnings) > 0
    if exam_finished:
        status_text = "EXAM SESSION COMPLETED (60s)"
        status_color = (0, 215, 255)
    elif is_violation:
        status_text = "STATUS: VIOLATION DETECTED"
        status_color = (0, 0, 255)
    else:
        status_text = "STATUS: EXAM SECURE"
        status_color = (0, 255, 100)

    # Title & Badge
    cv2.putText(
        frame,
        "ACE PROCTOR ENGINE",
        (20, 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.60,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        status_text,
        (20, 48),
        cv2.FONT_HERSHEY_DUPLEX,
        0.50,
        status_color,
        1,
        cv2.LINE_AA,
    )

    # 60-Second Exam Countdown Timer (Top Center)
    mins, secs = divmod(max(0, time_remaining), 60)
    timer_str = f"TIME: {mins:02d}:{secs:02d}"
    timer_color = (0, 0, 255) if time_remaining <= 10 else ((0, 255, 255) if time_remaining <= 30 else (0, 255, 100))
    if exam_finished:
        timer_str = "TIME: 00:00 (FINISHED)"
        timer_color = (0, 215, 255)

    timer_size, _ = cv2.getTextSize(timer_str, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
    timer_x = (w - timer_size[0]) // 2
    cv2.putText(
        frame,
        timer_str,
        (timer_x, 38),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        timer_color,
        2,
        cv2.LINE_AA,
    )

    # Live FPS & Face Count
    face_count = tracker_results.get("face_count", 0)
    fc_color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | Faces: {face_count}",
        (w - 230, 36),
        cv2.FONT_HERSHEY_DUPLEX,
        0.52,
        fc_color,
        1,
        cv2.LINE_AA,
    )

    # 2. Draw YOLO Detection Boxes
    for det in yolo_results.get("detections", []):
        x1, y1, x2, y2 = det["box"]
        label = f"{det['class_name'].upper()} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(frame, (x1, max(0, y1 - 25)), (x1 + len(label) * 11, y1), (0, 0, 200), -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 3. Bottom Telemetry Bar
    overlay_bot = frame.copy()
    cv2.rectangle(overlay_bot, (0, h - 55), (w, h), (15, 15, 20), -1)
    cv2.addWeighted(overlay_bot, 0.85, frame, 0.15, 0, frame)

    pitch = tracker_results.get("pitch", 0.0)
    yaw = tracker_results.get("yaw", 0.0)
    roll = tracker_results.get("roll", 0.0)
    p_dev = tracker_results.get("pitch_dev", 0.0)
    y_dev = tracker_results.get("yaw_dev", 0.0)
    lum = tracker_results.get("mean_luminance", 128.0)
    dist_pct = tracker_results.get("face_area_ratio", 0.0) * 100.0

    pose_str = (
        f"Pose -> Pitch: {pitch:+.1f} (dev: {p_dev:.1f}) | "
        f"Yaw: {yaw:+.1f} (dev: {y_dev:.1f}) | "
        f"Roll: {roll:+.1f}"
    )
    cv2.putText(
        frame,
        pose_str,
        (20, h - 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    env_str = (
        f"Gaze: {'OFF' if tracker_results.get('gaze_violation') else 'OK'} | "
        f"Dist: {dist_pct:.1f}% | "
        f"Lum: {lum:.0f} | "
        f"Posture: {'BAD' if tracker_results.get('posture_violation') else 'OK'} | "
        f"Hands: {'HIDDEN' if tracker_results.get('hands_hidden') else 'OK'}"
    )
    cv2.putText(
        frame,
        env_str,
        (20, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 180, 255),
        1,
        cv2.LINE_AA,
    )

    # 4. Active Warning Badges
    for i, warn_txt in enumerate(active_warnings):
        y_pos = 95 + i * 36
        cv2.rectangle(frame, (15, y_pos - 22), (420, y_pos + 8), (0, 0, 180), -1)
        cv2.rectangle(frame, (15, y_pos - 22), (420, y_pos + 8), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"! {warn_txt}",
            (25, y_pos),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main():
    print("=" * 65)
    print("       ACE (Anti Cheat Exam) - Zero-Lag Python Engine")
    print("=" * 65)

    # 1. Initialize Thread-safe Violation Logger
    logger = ViolationLogger()

    # 2. Start Asynchronous High-Speed Camera Stream
    print("[ACE Main] Initializing Camera Stream...")
    camera = ThreadedCamera().start()
    time.sleep(0.5)

    # 3. Check or Run Baseline Calibration
    baseline = {}
    if os.path.exists(Config.CALIBRATION_FILE):
        try:
            with open(Config.CALIBRATION_FILE, "r", encoding="utf-8") as f:
                baseline = json.load(f)
            print(f"[ACE Main] Loaded existing calibration from {Config.CALIBRATION_FILE}")
        except Exception:
            baseline = {}

    if not baseline:
        print("[ACE Main] Running initial 5-second calibration...")
        baseline = run_calibration(cap_or_camera=camera, duration_sec=Config.CALIBRATION_DURATION_SEC)

    # 4. Initialize Multi-modal Tracker with Baseline
    tracker = FaceProctorTracker(baseline=baseline)

    # 5. Start Background YOLO Detector Worker
    detector = YOLODetectorWorker().start()

    # 6. Initialize 60-second Exam Timer
    exam_duration = Config.EXAM_DURATION_SEC
    exam_start_time = time.time()

    print(f"[ACE Main] 60-Second Exam Session Started (Duration: {exam_duration}s).")
    print("[ACE Main] Press 'q' to exit, 'r' to restart 60s timer, 'c' to recalibrate.\n")

    # Metrics
    prev_time = time.time()
    fps = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame = camera.get_frame(timeout=1.0)
            if not ret or frame is None:
                continue

            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                frame_count = 0
                prev_time = now

            # Calculate 60-Second Exam Countdown
            elapsed_exam = now - exam_start_time
            time_remaining = max(0, int(exam_duration - elapsed_exam))
            exam_finished = elapsed_exam >= exam_duration

            display_frame = cv2.flip(frame, 1)

            # Asynchronously send newest frame to YOLO worker
            detector.update_frame(display_frame)

            # Process lightweight face, pose, distance, and lighting tracking
            tracker_results = tracker.process_frame(display_frame)
            yolo_results = detector.get_latest_results()

            active_warnings: List[str] = []

            # 1. No face
            if tracker_results["face_status"] == "NO_FACE":
                logger.report_violation(
                    violation_type="no_face",
                    frame=display_frame,
                    details="Student is not visible in camera frame",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_FACE,
                )
                active_warnings.append("WARNING: NO FACE DETECTED")
            else:
                logger.reset_violation("no_face")

            # 2. Multiple faces
            if tracker_results["face_status"] == "MULTIPLE_FACES":
                logger.report_violation(
                    violation_type="multiple_faces",
                    frame=display_frame,
                    details=f"Detected {tracker_results['face_count']} faces in frame",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_FACE,
                )
                active_warnings.append("WARNING: MULTIPLE PEOPLE DETECTED")
            else:
                logger.reset_violation("multiple_faces")

            # 3. Head pose deviation
            if tracker_results["head_pose_violation"]:
                logger.report_violation(
                    violation_type="head_pose_deviation",
                    frame=display_frame,
                    details=(
                        f"Head pose offset (Pitch: {tracker_results['pitch_dev']:.1f}, "
                        f"Yaw: {tracker_results['yaw_dev']:.1f})"
                    ),
                    debounce_threshold=Config.DEBOUNCE_FRAMES_POSE,
                )
                active_warnings.append("WARNING: LOOKING AWAY FROM SCREEN")
            else:
                logger.reset_violation("head_pose_deviation")

            # 4. Gaze off-screen
            if tracker_results["gaze_violation"]:
                logger.report_violation(
                    violation_type="eye_gaze_off_screen",
                    frame=display_frame,
                    details=(
                        f"Iris gaze ratio off-screen (L: {tracker_results['gaze_left_ratio']:.2f}, "
                        f"R: {tracker_results['gaze_right_ratio']:.2f})"
                    ),
                    debounce_threshold=Config.DEBOUNCE_FRAMES_POSE,
                )
                active_warnings.append("WARNING: EYES OFF-SCREEN")
            else:
                logger.reset_violation("eye_gaze_off_screen")

            # 5. Distance check
            if tracker_results["student_too_far"]:
                logger.report_violation(
                    violation_type="student_too_far",
                    frame=display_frame,
                    details=f"Face area too small ({tracker_results['face_area_ratio']*100:.1f}%)",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_FACE,
                )
                active_warnings.append("WARNING: STUDENT TOO FAR FROM CAMERA")
            else:
                logger.reset_violation("student_too_far")

            # 6. Posture check
            if tracker_results["posture_violation"]:
                logger.report_violation(
                    violation_type="posture_violation",
                    frame=display_frame,
                    details=tracker_results.get("posture_details", "Abnormal body posture detected"),
                    debounce_threshold=Config.DEBOUNCE_FRAMES_ENV,
                )
                active_warnings.append("WARNING: ABNORMAL BODY POSTURE")
            else:
                logger.reset_violation("posture_violation")

            # 7. Hand Tracking check (optional)
            if Config.ENABLE_HANDS_VIOLATION and tracker_results.get("hands_hidden"):
                logger.report_violation(
                    violation_type="hands_hidden",
                    frame=display_frame,
                    details=tracker_results.get("hands_details", "Hands hidden below desk for > 3s"),
                    debounce_threshold=1,
                )
                active_warnings.append("WARNING: HANDS HIDDEN BELOW DESK")
            else:
                logger.reset_violation("hands_hidden")

            # 7. Lighting checks
            lighting_viol = tracker_results.get("lighting_violation")
            if lighting_viol == "poor_lighting_dark":
                logger.report_violation(
                    violation_type="poor_lighting_dark",
                    frame=display_frame,
                    details=f"Workspace too dark (Mean: {tracker_results['mean_luminance']:.1f})",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_ENV,
                )
                active_warnings.append("WARNING: POOR LIGHTING (TOO DARK)")
                logger.reset_violation("poor_lighting_glare")
            elif lighting_viol == "poor_lighting_glare":
                logger.report_violation(
                    violation_type="poor_lighting_glare",
                    frame=display_frame,
                    details=f"Excessive camera glare (Mean: {tracker_results['mean_luminance']:.1f})",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_ENV,
                )
                active_warnings.append("WARNING: POOR LIGHTING (GLARE)")
                logger.reset_violation("poor_lighting_dark")
            else:
                logger.reset_violation("poor_lighting_dark")
                logger.reset_violation("poor_lighting_glare")

            # 8. YOLO Cell Phone
            if yolo_results.get("phone_detected"):
                logger.report_violation(
                    violation_type="cell_phone",
                    frame=display_frame,
                    details="Mobile phone detected in student workspace",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_YOLO,
                )
                active_warnings.append("WARNING: CELL PHONE DETECTED")
            else:
                logger.reset_violation("cell_phone")

            # 9. YOLO Book
            if yolo_results.get("book_detected"):
                logger.report_violation(
                    violation_type="book",
                    frame=display_frame,
                    details="Prohibited book / study material detected",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_YOLO,
                )
                active_warnings.append("WARNING: BOOK DETECTED")
            else:
                logger.reset_violation("book")

            # 10. YOLO Laptop
            if yolo_results.get("laptop_detected"):
                logger.report_violation(
                    violation_type="laptop",
                    frame=display_frame,
                    details="Prohibited second laptop/screen detected",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_YOLO,
                )
                active_warnings.append("WARNING: SECOND SCREEN/LAPTOP DETECTED")
            else:
                logger.reset_violation("laptop")

            # Render Modern HUD with timer
            draw_hud(
                display_frame,
                tracker_results,
                yolo_results,
                active_warnings,
                fps,
                baseline,
                time_remaining,
                exam_finished,
            )

            cv2.imshow("ACE Exam Proctor Engine", display_frame)

            # Auto-close after completion
            if exam_finished and Config.AUTO_CLOSE_ON_COMPLETE:
                overdue = elapsed_exam - exam_duration
                if overdue >= Config.AUTO_CLOSE_GRACE_SEC:
                    print("\n[ACE Main] 60-Second Exam Session Completed. Closing proctoring window automatically.")
                    break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[ACE Main] Shutdown initiated by user.")
                break
            elif key == ord("r"):
                print("\n[ACE Main] Resetting 60-second exam countdown timer...")
                exam_start_time = time.time()
                logger.reset_all()
            elif key == ord("c"):
                print("\n[ACE Main] Re-calibrating baseline pose...")
                baseline = run_calibration(cap_or_camera=camera, duration_sec=Config.CALIBRATION_DURATION_SEC)
                tracker.set_baseline(baseline)

    except KeyboardInterrupt:
        print("\n[ACE Main] Interrupted by keyboard.")

    finally:
        print("[ACE Main] Cleaning up resources...")
        detector.stop()
        camera.stop()
        tracker.close()
        cv2.destroyAllWindows()
        print("[ACE Main] Proctoring session closed gracefully.")


if __name__ == "__main__":
    main()
