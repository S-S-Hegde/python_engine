"""
ace/core/tracker.py - Multi-modal Proctoring Analyzer (Face, Pose, Hands, Distance, Lighting).
Integrates MediaPipe FaceLandmarker and PoseLandmarker Tasks with OpenCV luminance,
distance metrics, and hand/wrist visibility tracking.
"""

import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ace.config import Config
from ace.utils.assets import ensure_face_landmarker_model, ensure_pose_landmarker_model


# 3D generic facial reference model points for PnP
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

# Eye landmarks
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
LEFT_IRIS_CENTER = 468

RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_IRIS_CENTER = 473

# Pose Landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16


class FaceProctorTracker:
    """
    Unified analyzer combining MediaPipe FaceLandmarker, PoseLandmarker (posture + hands),
    luminance analysis, and distance verification.
    """

    def __init__(
        self,
        baseline: Optional[Dict[str, float]] = None,
        face_model_path: str = Config.FACE_LANDMARKER_MODEL_PATH,
        pose_model_path: str = Config.POSE_LANDMARKER_MODEL_PATH,
        pitch_thresh: float = Config.HEAD_POSE_PITCH_THRESH,
        yaw_thresh: float = Config.HEAD_POSE_YAW_THRESH,
        roll_thresh: float = Config.HEAD_POSE_ROLL_THRESH,
        min_face_area_ratio: float = Config.MIN_FACE_AREA_RATIO,
        luminance_min: float = Config.LUMINANCE_MIN,
        luminance_max: float = Config.LUMINANCE_MAX,
        gaze_left_thresh: float = Config.GAZE_RATIO_LEFT_THRESH,
        gaze_right_thresh: float = Config.GAZE_RATIO_RIGHT_THRESH,
        hands_hidden_duration_sec: float = Config.HANDS_HIDDEN_DURATION_SEC,
    ):
        self.pitch_thresh = pitch_thresh
        self.yaw_thresh = yaw_thresh
        self.roll_thresh = roll_thresh
        self.min_face_area_ratio = min_face_area_ratio
        self.luminance_min = luminance_min
        self.luminance_max = luminance_max
        self.gaze_left_thresh = gaze_left_thresh
        self.gaze_right_thresh = gaze_right_thresh
        self.hands_hidden_duration_sec = hands_hidden_duration_sec

        self.baseline = baseline or {
            "baseline_pitch": 0.0,
            "baseline_yaw": 0.0,
            "baseline_roll": 0.0,
        }

        # Frame counter and state caching to pace PoseLandmarker
        self._frame_count = 0
        self._last_posture = (False, "", False, "")
        self._hands_hidden_start: Optional[float] = None

        # Ensure model assets exist
        try:
            valid_face_model = ensure_face_landmarker_model(face_model_path)
            face_base_options = python.BaseOptions(model_asset_path=valid_face_model)
            face_options = vision.FaceLandmarkerOptions(
                base_options=face_base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=4,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        except Exception as e:
            print(f"[ACE Tracker] Warning: FaceLandmarker init fallback: {e}")
            self.face_landmarker = None

        try:
            valid_pose_model = ensure_pose_landmarker_model(pose_model_path)
            pose_base_options = python.BaseOptions(model_asset_path=valid_pose_model)
            pose_options = vision.PoseLandmarkerOptions(
                base_options=pose_base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4,
                output_segmentation_masks=False,
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        except Exception as e:
            print(f"[ACE Tracker] Warning: PoseLandmarker init fallback: {e}")
            self.pose_landmarker = None

    def set_baseline(self, baseline: Dict[str, float]):
        """Update resting head pose baseline angles."""
        self.baseline = baseline

    def _estimate_head_pose(
        self, landmarks, img_w: int, img_h: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[np.ndarray]]:
        """Compute standard Tait-Bryan Euler angles (pitch, yaw, roll) via solvePnP."""
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

        if not success:
            return None, None, None, None

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

        return pitch, yaw, roll, rot_vec

    def _estimate_gaze_ratio(self, landmarks, img_w: int, img_h: int) -> Tuple[float, float, bool]:
        """Estimate horizontal gaze direction."""
        try:
            l_outer = np.array([landmarks[LEFT_EYE_OUTER].x * img_w, landmarks[LEFT_EYE_OUTER].y * img_h])
            l_inner = np.array([landmarks[LEFT_EYE_INNER].x * img_w, landmarks[LEFT_EYE_INNER].y * img_h])
            l_iris = np.array([landmarks[LEFT_IRIS_CENTER].x * img_w, landmarks[LEFT_IRIS_CENTER].y * img_h])

            l_eye_width = np.linalg.norm(l_inner - l_outer)
            l_iris_dist = np.linalg.norm(l_iris - l_outer)
            left_ratio = l_iris_dist / max(1.0, l_eye_width)

            r_inner = np.array([landmarks[RIGHT_EYE_INNER].x * img_w, landmarks[RIGHT_EYE_INNER].y * img_h])
            r_outer = np.array([landmarks[RIGHT_EYE_OUTER].x * img_w, landmarks[RIGHT_EYE_OUTER].y * img_h])
            r_iris = np.array([landmarks[RIGHT_IRIS_CENTER].x * img_w, landmarks[RIGHT_IRIS_CENTER].y * img_h])

            r_eye_width = np.linalg.norm(r_outer - r_inner)
            r_iris_dist = np.linalg.norm(r_iris - r_inner)
            right_ratio = r_iris_dist / max(1.0, r_eye_width)

            avg_ratio = (left_ratio + right_ratio) / 2.0
            is_off_screen = (
                avg_ratio < self.gaze_left_thresh or avg_ratio > self.gaze_right_thresh
            )

            return float(left_ratio), float(right_ratio), is_off_screen
        except (IndexError, AttributeError):
            return 0.5, 0.5, False

    def _check_distance(self, landmarks) -> Tuple[float, bool]:
        """
        Calculate face bounding box area relative to frame resolution.
        Returns: (face_area_ratio, is_too_far)
        """
        try:
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            w_norm = max(xs) - min(xs)
            h_norm = max(ys) - min(ys)
            area_ratio = float(w_norm * h_norm)
            is_too_far = area_ratio < self.min_face_area_ratio
            return area_ratio, is_too_far
        except Exception:
            return 0.1, False

    def _check_lighting(self, frame: np.ndarray) -> Tuple[float, Optional[str]]:
        """
        Calculates frame luminance mean value.
        Returns: (mean_luminance, violation_type or None)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_lum = float(gray.mean())

        if mean_lum < self.luminance_min:
            return mean_lum, "poor_lighting_dark"
        elif mean_lum > self.luminance_max:
            return mean_lum, "poor_lighting_glare"
        return mean_lum, None

    def _check_posture_and_hands(self, mp_image: mp.Image) -> Tuple[bool, str, bool, str]:
        """
        Analyzes shoulder landmarks (11 & 12) for posture and wrist landmarks (15 & 16) for hands tracking.
        Returns: (posture_violation, posture_details, hands_violation, hands_details)
        """
        try:
            if not self.pose_landmarker:
                return False, "", False, ""
            pose_result = self.pose_landmarker.detect(mp_image)
            if not pose_result or not pose_result.pose_landmarks:
                # If hands were already hidden, keep timer running
                now = time.time()
                if self._hands_hidden_start is None:
                    self._hands_hidden_start = now
                elapsed = now - self._hands_hidden_start
                hands_viol = elapsed >= self.hands_hidden_duration_sec
                hands_det = f"Hands / body dropped out of view ({elapsed:.1f}s)" if hands_viol else ""
                return False, "", hands_viol, hands_det

            pose_lms = pose_result.pose_landmarks[0]
            l_sh = pose_lms[LEFT_SHOULDER]
            r_sh = pose_lms[RIGHT_SHOULDER]

            # 1. Posture: Shoulders dropped below visible frame
            posture_viol = False
            posture_det = ""
            if l_sh.y > 0.98 or r_sh.y > 0.98:
                posture_viol = True
                posture_det = "Shoulders dropped below frame boundary (slouching/ducking)"
            else:
                sh_dx = l_sh.x - r_sh.x
                sh_dy = l_sh.y - r_sh.y
                sh_dist = float(np.sqrt(sh_dx * sh_dx + sh_dy * sh_dy))
                if sh_dist < 0.10:
                    posture_viol = True
                    posture_det = f"Body turned sideways from screen (Shoulder width: {sh_dist:.2f})"

            # 2. Hand Tracking: Wrists (15 & 16) dropped below bottom boundary (y > 0.95)
            l_wr = pose_lms[LEFT_WRIST]
            r_wr = pose_lms[RIGHT_WRIST]

            # Check if wrists are below frame or hidden
            wrists_hidden = (l_wr.y > 0.95 or l_wr.visibility < 0.3) and (r_wr.y > 0.95 or r_wr.visibility < 0.3)

            now = time.time()
            hands_viol = False
            hands_det = ""

            if Config.ENABLE_HANDS_VIOLATION and wrists_hidden:
                if self._hands_hidden_start is None:
                    self._hands_hidden_start = now
                elapsed = now - self._hands_hidden_start
                if elapsed >= self.hands_hidden_duration_sec:
                    hands_viol = True
                    hands_det = f"Hands hidden below desk for {elapsed:.1f}s (> {self.hands_hidden_duration_sec}s threshold)"
            else:
                self._hands_hidden_start = None

            return posture_viol, posture_det, hands_viol, hands_det
        except Exception:
            return False, "", False, ""

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single image frame and return a comprehensive analysis dictionary.
        """
        h, w = frame.shape[:2]

        # 1. Lighting Check
        mean_lum, lighting_violation = self._check_lighting(frame)

        # Headless Cloud Guard: If Face & Pose landmarkers failed to load, return baseline safely
        if not self.face_landmarker and not self.pose_landmarker:
            return {
                "face_count": 1,
                "face_status": "NORMAL",
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "pitch_dev": 0.0,
                "yaw_dev": 0.0,
                "roll_dev": 0.0,
                "head_pose_violation": False,
                "looking_away": False,
                "gaze_left_ratio": 0.5,
                "gaze_right_ratio": 0.5,
                "gaze_violation": False,
                "face_area_ratio": 0.15,
                "student_too_far": False,
                "mean_luminance": mean_lum,
                "lighting_violation": lighting_violation,
                "posture_violation": False,
                "posture_details": "",
                "hands_hidden": False,
                "hands_details": "",
                "violations": [lighting_violation] if lighting_violation else [],
                "landmarks": None,
            }

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        except Exception:
            return {
                "face_count": 1,
                "face_status": "NORMAL",
                "pitch": 0.0,
                "yaw": 0.0,
                "roll": 0.0,
                "pitch_dev": 0.0,
                "yaw_dev": 0.0,
                "roll_dev": 0.0,
                "head_pose_violation": False,
                "looking_away": False,
                "gaze_left_ratio": 0.5,
                "gaze_right_ratio": 0.5,
                "gaze_violation": False,
                "face_area_ratio": 0.15,
                "student_too_far": False,
                "mean_luminance": mean_lum,
                "lighting_violation": lighting_violation,
                "posture_violation": False,
                "posture_details": "",
                "hands_hidden": False,
                "hands_details": "",
                "violations": [lighting_violation] if lighting_violation else [],
                "landmarks": None,
            }

        # 2. Paced Pose & Hand Tracking (runs every 4th frame)
        self._frame_count += 1
        if self._frame_count % 4 == 0:
            self._last_posture = self._check_posture_and_hands(mp_image)
        posture_violation, posture_details, hands_violation, hands_details = self._last_posture

        # 3. FaceLandmarker Inference
        face_results = None
        try:
            if self.face_landmarker:
                face_results = self.face_landmarker.detect(mp_image)
        except Exception:
            face_results = None

        analysis: Dict[str, Any] = {
            "face_count": 0,
            "face_status": "NO_FACE",
            "pitch": 0.0,
            "yaw": 0.0,
            "roll": 0.0,
            "pitch_dev": 0.0,
            "yaw_dev": 0.0,
            "roll_dev": 0.0,
            "head_pose_violation": False,
            "looking_away": False,
            "gaze_left_ratio": 0.5,
            "gaze_right_ratio": 0.5,
            "gaze_violation": False,
            "face_area_ratio": 0.0,
            "student_too_far": False,
            "mean_luminance": mean_lum,
            "lighting_violation": lighting_violation,
            "posture_violation": posture_violation,
            "posture_details": posture_details,
            "hands_hidden": hands_violation,
            "hands_details": hands_details,
            "violations": [],
            "landmarks": None,
        }

        # Environmental & Posture & Hand Violations
        if lighting_violation:
            analysis["violations"].append(lighting_violation)

        if posture_violation:
            analysis["violations"].append("posture_violation")

        if hands_violation:
            analysis["violations"].append("hands_hidden")

        if not face_results or not face_results.face_landmarks:
            analysis["face_count"] = 0
            analysis["face_status"] = "NO_FACE"
            analysis["violations"].append("no_face")
            return analysis

        face_count = len(face_results.face_landmarks)
        analysis["face_count"] = face_count

        if face_count > 1:
            analysis["face_status"] = "MULTIPLE_FACES"
            analysis["violations"].append("multiple_faces")
        else:
            analysis["face_status"] = "OK"

        # Primary face analysis
        primary_landmarks = face_results.face_landmarks[0]
        analysis["landmarks"] = primary_landmarks

        # Distance Check
        area_ratio, is_too_far = self._check_distance(primary_landmarks)
        analysis["face_area_ratio"] = area_ratio
        if is_too_far:
            analysis["student_too_far"] = True
            analysis["violations"].append("student_too_far")

        # Head Pose Math
        pitch, yaw, roll, _ = self._estimate_head_pose(primary_landmarks, w, h)
        if pitch is not None:
            analysis["pitch"] = pitch
            analysis["yaw"] = yaw
            analysis["roll"] = roll

            b_pitch = self.baseline.get("baseline_pitch", 0.0)
            b_yaw = self.baseline.get("baseline_yaw", 0.0)
            b_roll = self.baseline.get("baseline_roll", 0.0)

            p_dev = abs(pitch - b_pitch)
            y_dev = abs(yaw - b_yaw)
            r_dev = abs(roll - b_roll)

            analysis["pitch_dev"] = p_dev
            analysis["yaw_dev"] = y_dev
            analysis["roll_dev"] = r_dev

            if (
                p_dev > self.pitch_thresh
                or y_dev > self.yaw_thresh
                or r_dev > self.roll_thresh
            ):
                analysis["head_pose_violation"] = True
                analysis["looking_away"] = True
                analysis["violations"].append("head_pose_deviation")

        # Eye Gaze
        g_left, g_right, gaze_off = self._estimate_gaze_ratio(primary_landmarks, w, h)
        analysis["gaze_left_ratio"] = g_left
        analysis["gaze_right_ratio"] = g_right
        if gaze_off:
            analysis["gaze_violation"] = True
            analysis["violations"].append("eye_gaze_off_screen")

        return analysis

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "face_landmarker") and self.face_landmarker is not None:
            self.face_landmarker.close()
        if hasattr(self, "pose_landmarker") and self.pose_landmarker is not None:
            self.pose_landmarker.close()
