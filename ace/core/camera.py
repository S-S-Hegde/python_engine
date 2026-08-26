"""
ace/core/camera.py - Asynchronous, zero-lag camera capture thread with CLAHE low-light enhancement.
Uses queue.Queue(maxsize=1) with stale-frame dropping to ensure real-time video,
and applies automatic contrast/brightness equalization on the LAB L-channel.
"""

import time
import threading
import queue
from typing import Optional, Tuple
import cv2
import numpy as np
from ace.config import Config


class ThreadedCamera:
    """
    Dedicated background thread for OpenCV VideoCapture with CLAHE Low-Light Enhancement.
    Guarantees zero buffer delay by keeping only the most recent frame in memory.
    """

    def __init__(
        self,
        src: int = Config.CAMERA_INDEX,
        width: int = Config.CAMERA_WIDTH,
        height: int = Config.CAMERA_HEIGHT,
        fps: int = Config.CAMERA_FPS,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
    ):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps

        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.fps_actual = 0.0

        # Pre-instantiate CLAHE equalizer for low-latency L-channel enhancement
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_grid_size)

    def start(self) -> "ThreadedCamera":
        """Initialize camera and start background frame reader thread."""
        if self._running:
            return self

        self.is_virtual = False
        try:
            self.cap = cv2.VideoCapture(self.src)
            if not self.cap.isOpened():
                self.cap = None
                self.is_virtual = True
        except Exception:
            self.cap = None
            self.is_virtual = True

        if self.is_virtual:
            print(f"[ACE Camera] Notice: Video device at index {self.src} unavailable (cloud/headless instance). Running in virtual frame mode.")
        else:
            # Set hardware capture properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[ACE Camera] Stream started on device {self.src} ({self.width}x{self.height} @ {self.fps}FPS) [CLAHE Enhanced]")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="ACE-CameraThread", daemon=True)
        self._thread.start()
        return self

    def _enhance_low_light(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhances brightness and contrast in LAB color space using CLAHE on L-channel.
        """
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            cl = self.clahe.apply(l_channel)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception:
            return frame

    def _capture_loop(self):
        """Continuously grab frames, apply CLAHE enhancement, and push to queue."""
        prev_time = time.time()
        frame_counter = 0

        while self._running:
            if self.is_virtual or not self.cap or not self.cap.isOpened():
                # Headless virtual frame generator
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    "ACE Virtual Feed [Headless]",
                    (30, self.height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                time.sleep(1.0 / max(1, self.fps))
            else:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

            frame_counter += 1
            now = time.time()
            if now - prev_time >= 1.0:
                self.fps_actual = frame_counter / (now - prev_time)
                frame_counter = 0
                prev_time = now

            # Apply low-light equalizer
            enhanced_frame = self._enhance_low_light(frame) if not self.is_virtual else frame

            # If queue is full, discard the old frame to maintain zero latency
            if not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._frame_queue.put_nowait(enhanced_frame)
            except queue.Full:
                pass

    def get_frame(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Fetch the newest available enhanced frame.
        :param timeout: Maximum seconds to wait for a frame.
        :return: (True, frame) if available, (False, None) otherwise.
        """
        if not self._running:
            return False, None

        try:
            frame = self._frame_queue.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            return False, None

    def stop(self):
        """Gracefully stop thread and release camera hardware."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None

        # Flush queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

        print("[ACE Camera] Stream stopped and device released.")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
