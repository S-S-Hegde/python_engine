"""
ace/config.py - Central Configuration Loader for ACE Engine.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env")


class Config:
    # Camera
    CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
    CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", "1280"))
    CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "720"))
    CAMERA_FPS: int = int(os.getenv("CAMERA_FPS", "30"))

    # AI & Object Detection
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.22"))
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov10n.pt")
    DETECTOR_FPS: float = float(os.getenv("DETECTOR_FPS", "10.0"))
    MODELS_DIR: str = str(ROOT_DIR / "models")
    FACE_LANDMARKER_MODEL_PATH: str = str(ROOT_DIR / "models" / "face_landmarker.task")
    POSE_LANDMARKER_MODEL_PATH: str = str(ROOT_DIR / "models" / "pose_landmarker_heavy.task")

    # VLM (Vision-Language Model) Verification Settings
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    VLM_PROVIDER: str = os.getenv("VLM_PROVIDER", "groq")  # 'groq', 'openrouter', 'nvidia', 'openai', 'gemini'
    VLM_MODEL: str = os.getenv("VLM_MODEL", "llama-3.2-11b-vision-instruct")
    VLM_VERIFICATION_ENABLED: bool = os.getenv("VLM_VERIFICATION_ENABLED", "true").lower() in ("true", "1", "yes")

    # Logging & Screenshots
    SCREENSHOT_DIR: str = str(ROOT_DIR / os.getenv("SCREENSHOT_DIR", "logs/screenshots"))
    LOG_FILE: str = str(ROOT_DIR / os.getenv("LOG_FILE", "logs/exam_violations.log"))

    # Debounce / Temporal Filtering (Immediate YOLO response)
    DEBOUNCE_FRAMES_YOLO: int = int(os.getenv("DEBOUNCE_FRAMES_YOLO", "1"))
    DEBOUNCE_FRAMES_POSE: int = int(os.getenv("DEBOUNCE_FRAMES_POSE", "15"))
    DEBOUNCE_FRAMES_FACE: int = int(os.getenv("DEBOUNCE_FRAMES_FACE", "6"))
    DEBOUNCE_FRAMES_ENV: int = int(os.getenv("DEBOUNCE_FRAMES_ENV", "15"))
    VIOLATION_COOLDOWN_SEC: float = float(os.getenv("VIOLATION_COOLDOWN_SEC", "2.0"))

    # Exam Session Timer Settings
    EXAM_DURATION_SEC: int = int(os.getenv("EXAM_DURATION_SEC", "60"))
    AUTO_CLOSE_ON_COMPLETE: bool = os.getenv("AUTO_CLOSE_ON_COMPLETE", "true").lower() in ("true", "1", "yes")
    AUTO_CLOSE_GRACE_SEC: int = int(os.getenv("AUTO_CLOSE_GRACE_SEC", "3"))

    # Calibration & Head Pose Thresholds (Expanded Range)
    CALIBRATION_DURATION_SEC: int = int(os.getenv("CALIBRATION_DURATION_SEC", "5"))
    CALIBRATION_FILE: str = str(ROOT_DIR / "calibration.json")
    HEAD_POSE_PITCH_THRESH: float = float(os.getenv("HEAD_POSE_PITCH_THRESH", "25.0"))
    HEAD_POSE_YAW_THRESH: float = float(os.getenv("HEAD_POSE_YAW_THRESH", "30.0"))
    HEAD_POSE_ROLL_THRESH: float = float(os.getenv("HEAD_POSE_ROLL_THRESH", "22.0"))

    # Hand Tracking Thresholds (Disabled by default to allow mouse/writing/paper work)
    ENABLE_HANDS_VIOLATION: bool = os.getenv("ENABLE_HANDS_VIOLATION", "false").lower() in ("true", "1", "yes")
    HANDS_HIDDEN_DURATION_SEC: float = float(os.getenv("HANDS_HIDDEN_DURATION_SEC", "3.0"))

    # Distance Check (Expanded Distance Range: 0.008 = 0.8% face area)
    MIN_FACE_AREA_RATIO: float = float(os.getenv("MIN_FACE_AREA_RATIO", "0.008"))

    # Lighting Awareness Thresholds (Luminance Mean 0 - 255)
    LUMINANCE_MIN: float = float(os.getenv("LUMINANCE_MIN", "35.0"))
    LUMINANCE_MAX: float = float(os.getenv("LUMINANCE_MAX", "235.0"))

    # Eye Gaze Thresholds (Expanded Range)
    GAZE_RATIO_LEFT_THRESH: float = float(os.getenv("GAZE_RATIO_LEFT_THRESH", "0.25"))
    GAZE_RATIO_RIGHT_THRESH: float = float(os.getenv("GAZE_RATIO_RIGHT_THRESH", "0.75"))

    # Coding Exam Security & OS Lockdown Settings
    ENABLE_PROCESS_AUDIT: bool = os.getenv("ENABLE_PROCESS_AUDIT", "true").lower() in ("true", "1", "yes")
    ENABLE_MULTI_DISPLAY_CHECK: bool = os.getenv("ENABLE_MULTI_DISPLAY_CHECK", "true").lower() in ("true", "1", "yes")
    ENABLE_CLIPBOARD_LOCKDOWN: bool = os.getenv("ENABLE_CLIPBOARD_LOCKDOWN", "true").lower() in ("true", "1", "yes")
    MAX_TAB_SWITCH_STRIKES: int = int(os.getenv("MAX_TAB_SWITCH_STRIKES", "3"))

