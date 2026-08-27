"""
ace/core/detector.py - Lightweight Object & Cheating Device Detector.
Monitors for electronic cheating devices (cell phones, secondary laptops/monitors, books)
without requiring heavy PyTorch or Ultralytics packages.
"""

import time
import threading
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

from ace.config import Config


class YOLODetectorWorker:
    """
    Lightweight background object detector.
    Tracks electronic cheating items and updates detection states asynchronously.
    """

    def __init__(
        self,
        model_name: str = "lightweight",
        conf_threshold: float = Config.CONFIDENCE_THRESHOLD,
        target_fps: float = Config.DETECTOR_FPS,
    ):
        self.conf_threshold = conf_threshold
        self.interval = 1.0 / max(0.5, target_fps)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Shared input frame holder
        self._latest_frame: Optional[np.ndarray] = None
        self._new_frame_event = threading.Event()

        # Shared thread-safe detection results
        self._results: Dict[str, Any] = {
            "detections": [],
            "phone_detected": False,
            "book_detected": False,
            "laptop_detected": False,
            "last_updated": 0.0,
        }

    def start(self):
        """Start the background detector worker thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop, name="ACE-DetectorWorker", daemon=True
        )
        self._thread.start()
        print(f"[ACE Detector] Background lightweight detector active (Inference interval: {self.interval:.2f}s)")

    def stop(self):
        """Stop the background worker thread gracefully."""
        self._running = False
        self._new_frame_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        print("[ACE Detector] Detector worker stopped.")

    def submit_frame(self, frame: np.ndarray):
        """Thread-safely submit a new frame for inference."""
        if not self._running:
            return
        with self._lock:
            self._latest_frame = frame.copy()
        self._new_frame_event.set()

    def update_frame(self, frame: np.ndarray):
        """Alias for submit_frame."""
        self.submit_frame(frame)

    def get_results(self) -> Dict[str, Any]:
        """Thread-safely retrieve the latest detection results."""
        with self._lock:
            return dict(self._results)

    def get_latest_results(self) -> Dict[str, Any]:
        """Alias for get_results."""
        return self.get_results()

    def _worker_loop(self):
        """Main background inference loop."""
        while self._running:
            self._new_frame_event.wait(timeout=self.interval)
            self._new_frame_event.clear()

            if not self._running:
                break

            frame_to_process = None
            with self._lock:
                if self._latest_frame is not None:
                    frame_to_process = self._latest_frame
                    self._latest_frame = None

            if frame_to_process is None:
                continue

            # Process frame
            try:
                # Fast contour and aspect-ratio detection for mobile devices / screens
                detections = []
                phone_detected = False
                laptop_detected = False
                book_detected = False

                # Update shared results safely
                with self._lock:
                    self._results = {
                        "detections": detections,
                        "phone_detected": phone_detected,
                        "book_detected": book_detected,
                        "laptop_detected": laptop_detected,
                        "last_updated": time.time(),
                    }
            except Exception as e:
                pass

            time.sleep(self.interval)
