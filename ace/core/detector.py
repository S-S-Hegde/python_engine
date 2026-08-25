"""
ace/core/detector.py - Background asynchronous YOLOv10 object detector.
Strictly filters for cheating devices (laptop, cell phone, book) with optimized confidence.
"""

import time
import threading
from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO
from ace.config import Config


# COCO Dataset class indices for prohibited cheating objects:
# 62: tv/monitor, 63: laptop, 65: remote, 67: cell phone, 73: book
CHEATING_CLASSES_INDICES = [62, 63, 65, 67, 73]


class YOLODetectorWorker:
    """
    Dedicated background worker thread for YOLOv10 inference.
    Monitors for electronic cheating devices (cell phone, laptop, monitor, remote, book).
    """

    SUSPICIOUS_CLASSES = {
        "cell phone": "cell_phone",
        "laptop": "laptop",
        "tv": "secondary_screen",
        "remote": "electronic_device",
        "book": "book",
    }

    def __init__(
        self,
        model_name: str = Config.YOLO_MODEL,
        conf_threshold: float = Config.CONFIDENCE_THRESHOLD,
        target_fps: float = Config.DETECTOR_FPS,
    ):
        self.model_name = model_name or "yolov10n.pt"
        self.conf_threshold = conf_threshold
        self.interval = 1.0 / max(0.5, target_fps)

        self.model: Optional[YOLO] = None
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

    def start(self) -> "YOLODetectorWorker":
        """Load YOLOv10 model and spawn background detector thread."""
        if self._running:
            return self

        print(f"[ACE Detector] Loading YOLOv10 model '{self.model_name}' (conf: {self.conf_threshold}, classes: {CHEATING_CLASSES_INDICES})...")
        self.model = YOLO(self.model_name)
        self._running = True
        self._thread = threading.Thread(
            target=self._detection_loop, name="ACE-YOLO10Thread", daemon=True
        )
        self._thread.start()
        print(f"[ACE Detector] Background YOLOv10 detector active (Inference interval: {self.interval:.2f}s)")
        return self

    def update_frame(self, frame: np.ndarray):
        """Pass the newest video frame to the detector worker."""
        if not self._running:
            return
        with self._lock:
            self._latest_frame = frame.copy()
        self._new_frame_event.set()

    def _detection_loop(self):
        """Worker loop executing YOLOv10 inference strictly on cheating device classes."""
        while self._running:
            # Wait for a new frame or timeout
            self._new_frame_event.wait(timeout=0.2)
            self._new_frame_event.clear()

            if not self._running:
                break

            current_frame = None
            with self._lock:
                if self._latest_frame is not None:
                    current_frame = self._latest_frame
                    self._latest_frame = None

            if current_frame is None or self.model is None:
                continue

            start_t = time.time()

            try:
                # Run YOLOv10 inference strictly filtering for cheating devices
                results = self.model.predict(
                    current_frame,
                    conf=self.conf_threshold,
                    classes=CHEATING_CLASSES_INDICES,
                    imgsz=640,
                    verbose=False,
                    device="cpu",
                )

                detections_list: List[Dict[str, Any]] = []
                phone_found = False
                book_found = False
                laptop_found = False
                screen_found = False
                device_found = False

                for r in results:
                    for box in r.boxes:
                        cls_idx = int(box.cls[0])
                        class_name = self.model.names.get(cls_idx, "").lower()
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        if class_name == "cell phone":
                            phone_found = True
                        elif class_name == "book":
                            book_found = True
                        elif class_name == "laptop":
                            laptop_found = True
                        elif class_name in ("tv", "monitor"):
                            screen_found = True
                        elif class_name == "remote":
                            device_found = True

                        detections_list.append(
                            {
                                "class_name": class_name,
                                "box": (x1, y1, x2, y2),
                                "confidence": conf,
                            }
                        )

                # Atomically update shared state
                with self._lock:
                    self._results = {
                        "detections": detections_list,
                        "phone_detected": phone_found,
                        "book_detected": book_found,
                        "laptop_detected": laptop_found,
                        "screen_detected": screen_found,
                        "electronic_device_detected": device_found,
                        "last_updated": time.time(),
                    }

            except Exception as e:
                print(f"[ACE Detector] YOLOv10 inference error: {e}")

            # Sleep to match target FPS (minimal sleep for high responsiveness)
            elapsed = time.time() - start_t
            sleep_time = max(0.001, self.interval - elapsed)
            time.sleep(sleep_time)

    def get_latest_results(self) -> Dict[str, Any]:
        """
        Thread-safe getter for current object detection findings.
        Returns instantly without blocking the rendering pipeline.
        """
        with self._lock:
            return dict(self._results)

    def stop(self):
        """Stop background YOLOv10 thread."""
        self._running = False
        self._new_frame_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("[ACE Detector] Background YOLOv10 detector stopped.")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
