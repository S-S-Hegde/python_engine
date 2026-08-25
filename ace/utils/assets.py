"""
ace/utils/assets.py - Asset and model management.
Automatically downloads required task models (face_landmarker.task, pose_landmarker_heavy.task)
from official Google MediaPipe repositories if not present.
"""

import os
import urllib.request
from pathlib import Path

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

POSE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)


def _download_with_progress(url: str, dest_path: Path, model_name: str):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ACE Assets] Downloading MediaPipe {model_name} model to {dest_path}...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100 / total_size)
            print(f"\r[ACE Assets] {model_name} download progress: {percent:.1f}% ({downloaded // 1024} KB / {total_size // 1024} KB)", end="")

    try:
        urllib.request.urlretrieve(url, str(dest_path), reporthook=_progress)
        print(f"\n[ACE Assets] {model_name} downloaded successfully.")
    except Exception as e:
        alt_url = url.replace("/latest/", "/1/")
        print(f"\n[ACE Assets] Retrying {model_name} with mirror: {alt_url}")
        urllib.request.urlretrieve(alt_url, str(dest_path))
        print(f"[ACE Assets] {model_name} downloaded successfully from mirror.")


def ensure_face_landmarker_model(model_path: str) -> str:
    """
    Checks if the MediaPipe face_landmarker.task model exists.
    If not, downloads it from the official Google repository.
    """
    path = Path(model_path)
    if path.exists() and path.stat().st_size > 0:
        return str(path)

    _download_with_progress(FACE_LANDMARKER_URL, path, "Face Landmarker")
    return str(path)


def ensure_pose_landmarker_model(model_path: str) -> str:
    """
    Checks if the MediaPipe pose_landmarker_heavy.task model exists.
    If not, downloads it from the official Google repository.
    """
    path = Path(model_path)
    if path.exists() and path.stat().st_size > 0:
        return str(path)

    _download_with_progress(POSE_LANDMARKER_URL, path, "Pose Landmarker Heavy")
    return str(path)
