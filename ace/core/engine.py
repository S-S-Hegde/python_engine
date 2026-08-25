"""
ace/core/engine.py - Background Proctoring Engine Orchestrator.
Coordinates camera streaming, AI tracking, object detection, violation debouncing,
60-second exam countdown timer, auto-close termination, and frame encoding for web streaming.
"""

import os
import sys
import time
import json
import threading
from typing import Dict, Any, Optional, List
import cv2
import numpy as np

from ace.config import Config
from ace.core.camera import ThreadedCamera
from ace.core.detector import YOLODetectorWorker
from ace.core.tracker import FaceProctorTracker
from ace.core.logger import ViolationLogger
from ace.utils.calibrate import run_calibration


def draw_hud_overlay(
    frame: np.ndarray,
    tracker_results: dict,
    yolo_results: dict,
    active_warnings: list,
    fps: float,
    time_remaining: int,
    exam_finished: bool,
    auto_close_rem: int = 0,
):
    """Renders an elegant cyber-style HUD overlay on the display frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top Navigation Bar
    cv2.rectangle(overlay, (0, 0), (w, 60), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    is_violation = len(active_warnings) > 0
    if exam_finished:
        if Config.AUTO_CLOSE_ON_COMPLETE:
            status_text = f"EXAM COMPLETED - CLOSING IN {auto_close_rem}s..."
        else:
            status_text = "EXAM SESSION COMPLETED (60s)"
        status_color = (0, 215, 255)  # Gold
    elif is_violation:
        status_text = "STATUS: VIOLATION DETECTED"
        status_color = (0, 0, 255)
    else:
        status_text = "STATUS: EXAM SECURE"
        status_color = (0, 255, 100)

    # Title & Status Badge (Left)
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
        timer_str = f"COMPLETED ({auto_close_rem}s)" if Config.AUTO_CLOSE_ON_COMPLETE else "TIME: 00:00 (FINISHED)"
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

    # Live FPS & Face Count (Top Right)
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

    # YOLO Detection Bounding Boxes
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

    # Bottom Telemetry Bar
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

    # Active Warning Badges on Left Side
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


class ProctorEngine:
    """
    Main proctoring engine orchestrator that coordinates background capture,
    AI models, debouncing, 60-second timer, auto-close termination, and web stream generation.
    """

    def __init__(self):
        self.camera: Optional[ThreadedCamera] = None
        self.tracker: Optional[FaceProctorTracker] = None
        self.detector: Optional[YOLODetectorWorker] = None
        self.logger = ViolationLogger()

        self.exam_duration: int = Config.EXAM_DURATION_SEC
        self.exam_start_time: Optional[float] = None
        self.baseline: Dict[str, Any] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self.should_shutdown: bool = False
        self.auto_close_rem: int = Config.AUTO_CLOSE_GRACE_SEC

        # Latest encoded JPEG frame for MJPEG stream
        self._latest_jpeg: Optional[bytes] = None
        self._latest_display_frame: Optional[np.ndarray] = None
        self._frame_cond = threading.Condition(self._lock)

        # Real-time telemetry snapshot
        self._telemetry: Dict[str, Any] = {
            "fps": 0.0,
            "status": "INITIALIZING",
            "time_remaining": self.exam_duration,
            "exam_duration": self.exam_duration,
            "exam_finished": False,
            "auto_close_rem": self.auto_close_rem,
            "should_shutdown": False,
            "faces_count": 0,
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "pitch_dev": 0.0,
            "yaw_dev": 0.0,
            "gaze_off_screen": False,
            "face_area_ratio": 0.0,
            "student_too_far": False,
            "mean_luminance": 128.0,
            "posture_violation": False,
            "phone_detected": False,
            "book_detected": False,
            "laptop_detected": False,
            "active_warnings": [],
            "timestamp": time.time(),
        }

    def start(self):
        """Initialize all sub-modules and start proctoring loop."""
        if self._running:
            return self

        print("[ACE Engine] Initializing Threaded Camera...")
        self.camera = ThreadedCamera().start()
        time.sleep(0.4)

        # Check or create calibration
        if os.path.exists(Config.CALIBRATION_FILE):
            try:
                with open(Config.CALIBRATION_FILE, "r", encoding="utf-8") as f:
                    self.baseline = json.load(f)
                print(f"[ACE Engine] Loaded existing baseline from {Config.CALIBRATION_FILE}")
            except Exception:
                self.baseline = {}

        if not self.baseline:
            print("[ACE Engine] Running auto-calibration routine...")
            self.baseline = run_calibration(
                cap_or_camera=self.camera,
                duration_sec=Config.CALIBRATION_DURATION_SEC,
                show_ui=False,
            )

        print("[ACE Engine] Initializing MediaPipe Face & Pose Trackers...")
        self.tracker = FaceProctorTracker(baseline=self.baseline)

        print("[ACE Engine] Initializing YOLOv10 Worker Thread...")
        self.detector = YOLODetectorWorker().start()

        # Start 60-second exam session timer
        self.exam_start_time = time.time()
        self.should_shutdown = False
        print(f"[ACE Engine] 60-Second Exam Session Timer Started (Duration: {self.exam_duration}s, AutoClose: {Config.AUTO_CLOSE_ON_COMPLETE}).")

        self._running = True
        self._thread = threading.Thread(
            target=self._processing_loop, name="ACE-EngineLoop", daemon=True
        )
        self._thread.start()
        print("[ACE Engine] Proctoring Engine successfully started.")
        return self

    def reset_exam(self, duration_sec: Optional[int] = None) -> Dict[str, Any]:
        """Reset the 60-second exam session countdown timer."""
        with self._lock:
            if duration_sec is not None:
                self.exam_duration = duration_sec
            self.exam_start_time = time.time()
            self.should_shutdown = False
            self.auto_close_rem = Config.AUTO_CLOSE_GRACE_SEC
            self.logger.reset_all()
        print(f"[ACE Engine] Exam timer reset to {self.exam_duration}s.")
        return {
            "status": "reset",
            "exam_duration": self.exam_duration,
            "time_remaining": self.exam_duration,
        }

    def _processing_loop(self):
        """Main proctoring processing and rendering thread."""
        prev_time = time.time()
        fps = 0.0
        frame_count = 0

        while self._running and self.camera:
            ret, frame = self.camera.get_frame(timeout=0.5)
            if not ret or frame is None:
                continue

            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                frame_count = 0
                prev_time = now

            display_frame = cv2.flip(frame, 1)

            # Calculate 60-second countdown
            if self.exam_start_time is not None:
                elapsed_exam = now - self.exam_start_time
                time_remaining = max(0, int(self.exam_duration - elapsed_exam))
                exam_finished = elapsed_exam >= self.exam_duration
                
                # Auto-close countdown
                if exam_finished and Config.AUTO_CLOSE_ON_COMPLETE:
                    overdue = elapsed_exam - self.exam_duration
                    auto_close_rem = max(0, int(Config.AUTO_CLOSE_GRACE_SEC - overdue))
                    if overdue >= Config.AUTO_CLOSE_GRACE_SEC:
                        self.should_shutdown = True
                else:
                    auto_close_rem = Config.AUTO_CLOSE_GRACE_SEC
            else:
                time_remaining = self.exam_duration
                exam_finished = False
                auto_close_rem = Config.AUTO_CLOSE_GRACE_SEC

            self.auto_close_rem = auto_close_rem

            # Feed YOLO detector asynchronously
            if self.detector:
                self.detector.update_frame(display_frame)

            # Run MediaPipe face, pose, distance, and lighting tracking
            tracker_results = self.tracker.process_frame(display_frame) if self.tracker else {}
            yolo_results = self.detector.get_latest_results() if self.detector else {}

            active_warnings: List[str] = []

            # 1. No Face
            if tracker_results.get("face_status") == "NO_FACE":
                self.logger.report_violation(
                    violation_type="no_face",
                    frame=display_frame,
                    details="Student is not visible in camera frame",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_FACE,
                )
                active_warnings.append("WARNING: NO FACE DETECTED")
            else:
                self.logger.reset_violation("no_face")

            # 2. Multiple Faces
            if tracker_results.get("face_status") == "MULTIPLE_FACES":
                self.logger.report_violation(
                    violation_type="multiple_faces",
                    frame=display_frame,
                    details=f"Detected {tracker_results.get('face_count', 0)} faces in frame",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_FACE,
                )
                active_warnings.append("WARNING: MULTIPLE PEOPLE DETECTED")
            else:
                self.logger.reset_violation("multiple_faces")

            # 3. Head Pose Deviation
            if tracker_results.get("head_pose_violation"):
                self.logger.report_violation(
                    violation_type="head_pose_deviation",
                    frame=display_frame,
                    details=(
                        f"Head pose offset (Pitch dev: {tracker_results.get('pitch_dev', 0.0):.1f}, "
                        f"Yaw dev: {tracker_results.get('yaw_dev', 0.0):.1f})"
                    ),
                    debounce_threshold=Config.DEBOUNCE_FRAMES_POSE,
                )
                active_warnings.append("WARNING: LOOKING AWAY FROM SCREEN")
            else:
                self.logger.reset_violation("head_pose_deviation")

            # 4. Gaze Off-Screen
            if tracker_results.get("gaze_violation"):
                self.logger.report_violation(
                    violation_type="eye_gaze_off_screen",
                    frame=display_frame,
                    details=(
                        f"Iris gaze ratio off-screen (L: {tracker_results.get('gaze_left_ratio', 0.5):.2f}, "
                        f"R: {tracker_results.get('gaze_right_ratio', 0.5):.2f})"
                    ),
                    debounce_threshold=Config.DEBOUNCE_FRAMES_POSE,
                )
                active_warnings.append("WARNING: EYES OFF-SCREEN")
            else:
                self.logger.reset_violation("eye_gaze_off_screen")

            # 5. Distance (Student Too Far)
            if tracker_results.get("student_too_far"):
                self.logger.report_violation(
                    violation_type="student_too_far",
                    frame=display_frame,
                    details=f"Face area too small ({tracker_results.get('face_area_ratio', 0.0)*100:.1f}%)",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_FACE,
                )
                active_warnings.append("WARNING: STUDENT TOO FAR FROM CAMERA")
            else:
                self.logger.reset_violation("student_too_far")

            # 6. Posture Violation
            if tracker_results.get("posture_violation"):
                self.logger.report_violation(
                    violation_type="posture_violation",
                    frame=display_frame,
                    details=tracker_results.get("posture_details", "Abnormal body posture detected"),
                    debounce_threshold=Config.DEBOUNCE_FRAMES_ENV,
                )
                active_warnings.append("WARNING: ABNORMAL BODY POSTURE")
            else:
                self.logger.reset_violation("posture_violation")

            # 7. Hands Hidden Violation (optional)
            if Config.ENABLE_HANDS_VIOLATION and tracker_results.get("hands_hidden"):
                self.logger.report_violation(
                    violation_type="hands_hidden",
                    frame=display_frame,
                    details=tracker_results.get("hands_details", "Hands hidden below desk for > 3s"),
                    debounce_threshold=1,
                )
                active_warnings.append("WARNING: HANDS HIDDEN BELOW DESK")
            else:
                self.logger.reset_violation("hands_hidden")

            # 7. Lighting Checks
            lighting_viol = tracker_results.get("lighting_violation")
            if lighting_viol == "poor_lighting_dark":
                self.logger.report_violation(
                    violation_type="poor_lighting_dark",
                    frame=display_frame,
                    details=f"Workspace too dark (Mean luminance: {tracker_results.get('mean_luminance', 0.0):.1f})",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_ENV,
                )
                active_warnings.append("WARNING: POOR LIGHTING (TOO DARK)")
                self.logger.reset_violation("poor_lighting_glare")
            elif lighting_viol == "poor_lighting_glare":
                self.logger.report_violation(
                    violation_type="poor_lighting_glare",
                    frame=display_frame,
                    details=f"Excessive camera glare (Mean luminance: {tracker_results.get('mean_luminance', 0.0):.1f})",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_ENV,
                )
                active_warnings.append("WARNING: POOR LIGHTING (GLARE)")
                self.logger.reset_violation("poor_lighting_dark")
            else:
                self.logger.reset_violation("poor_lighting_dark")
                self.logger.reset_violation("poor_lighting_glare")

            # 8. YOLO Cell Phone
            if yolo_results.get("phone_detected"):
                self.logger.report_violation(
                    violation_type="cell_phone",
                    frame=display_frame,
                    details="Mobile phone detected in workspace",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_YOLO,
                )
                active_warnings.append("WARNING: CELL PHONE DETECTED")
            else:
                self.logger.reset_violation("cell_phone")

            # 9. YOLO Book
            if yolo_results.get("book_detected"):
                self.logger.report_violation(
                    violation_type="book",
                    frame=display_frame,
                    details="Prohibited book / study material detected",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_YOLO,
                )
                active_warnings.append("WARNING: BOOK DETECTED")
            else:
                self.logger.reset_violation("book")

            # 10. YOLO Laptop
            if yolo_results.get("laptop_detected"):
                self.logger.report_violation(
                    violation_type="laptop",
                    frame=display_frame,
                    details="Prohibited second laptop/computer detected",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_YOLO,
                )
                active_warnings.append("WARNING: SECOND LAPTOP DETECTED")
            else:
                self.logger.reset_violation("laptop")

            # 11. YOLO Secondary Screen / TV / Monitor
            if yolo_results.get("screen_detected"):
                self.logger.report_violation(
                    violation_type="secondary_screen",
                    frame=display_frame,
                    details="Prohibited secondary screen / monitor / TV detected",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_YOLO,
                )
                active_warnings.append("WARNING: SECONDARY SCREEN DETECTED")
            else:
                self.logger.reset_violation("secondary_screen")

            # 12. YOLO Electronic Device / Gadget
            if yolo_results.get("electronic_device_detected"):
                self.logger.report_violation(
                    violation_type="electronic_device",
                    frame=display_frame,
                    details="Prohibited electronic gadget / remote detected",
                    debounce_threshold=Config.DEBOUNCE_FRAMES_YOLO,
                )
                active_warnings.append("WARNING: ELECTRONIC GADGET DETECTED")
            else:
                self.logger.reset_violation("electronic_device")

            # Draw HUD with live timer & auto-close notice
            draw_hud_overlay(
                display_frame,
                tracker_results,
                yolo_results,
                active_warnings,
                fps,
                time_remaining,
                exam_finished,
                auto_close_rem,
            )

            # Encode frame to JPEG
            encode_success, buffer = cv2.imencode(
                ".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )

            if encode_success:
                with self._frame_cond:
                    self._latest_jpeg = buffer.tobytes()
                    self._latest_display_frame = display_frame.copy()
                    self._telemetry = {
                        "fps": round(fps, 1),
                        "status": "COMPLETED" if exam_finished else ("VIOLATION" if active_warnings else "SECURE"),
                        "time_remaining": time_remaining,
                        "exam_duration": self.exam_duration,
                        "exam_finished": exam_finished,
                        "auto_close_rem": auto_close_rem,
                        "should_shutdown": self.should_shutdown,
                        "faces_count": tracker_results.get("face_count", 0),
                        "pitch": round(tracker_results.get("pitch", 0.0), 1),
                        "yaw": round(tracker_results.get("yaw", 0.0), 1),
                        "roll": round(tracker_results.get("roll", 0.0), 1),
                        "pitch_dev": round(tracker_results.get("pitch_dev", 0.0), 1),
                        "yaw_dev": round(tracker_results.get("yaw_dev", 0.0), 1),
                        "gaze_off_screen": tracker_results.get("gaze_violation", False),
                        "face_area_ratio": round(tracker_results.get("face_area_ratio", 0.0), 3),
                        "student_too_far": tracker_results.get("student_too_far", False),
                        "mean_luminance": round(tracker_results.get("mean_luminance", 128.0), 1),
                        "lighting_violation": tracker_results.get("lighting_violation"),
                        "posture_violation": tracker_results.get("posture_violation", False),
                        "hands_hidden": tracker_results.get("hands_hidden", False),
                        "phone_detected": yolo_results.get("phone_detected", False),
                        "book_detected": yolo_results.get("book_detected", False),
                        "laptop_detected": yolo_results.get("laptop_detected", False),
                        "active_warnings": active_warnings,
                        "timestamp": time.time(),
                    }
                    self._frame_cond.notify_all()

    def get_latest_frame(self, timeout: float = 0.5) -> Optional[bytes]:
        """Fetch the latest encoded JPEG frame bytes with condition waiting."""
        with self._frame_cond:
            if self._latest_jpeg is None:
                self._frame_cond.wait(timeout=timeout)
            return self._latest_jpeg

    def get_latest_frame_mat(self) -> Optional[np.ndarray]:
        """Fetch the latest raw display image matrix for evidence capture."""
        with self._lock:
            if self._latest_display_frame is not None:
                return self._latest_display_frame.copy()
            return None

    def get_telemetry(self) -> Dict[str, Any]:
        """Return the latest real-time telemetry snapshot."""
        with self._lock:
            return dict(self._telemetry)

    def recalibrate(self, duration_sec: int = Config.CALIBRATION_DURATION_SEC) -> Dict[str, Any]:
        """Perform on-demand baseline resting pose recalibration."""
        if not self.camera:
            raise RuntimeError("Camera is not active")
        self.baseline = run_calibration(
            cap_or_camera=self.camera,
            duration_sec=duration_sec,
            show_ui=False,
        )
        if self.tracker:
            self.tracker.set_baseline(self.baseline)
        return self.baseline

    def stop(self):
        """Cleanly terminate all proctoring threads and hardware capture."""
        self._running = False
        with self._frame_cond:
            self._frame_cond.notify_all()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self.detector:
            self.detector.stop()
            self.detector = None

        if self.camera:
            self.camera.stop()
            self.camera = None

        if self.tracker:
            self.tracker.close()
            self.tracker = None

        print("[ACE Engine] Engine stopped and all hardware resources released.")
