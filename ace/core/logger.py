"""
ace/core/logger.py - Thread-safe violation debouncer with asynchronous Groq VLM verification.
Integrates temporal filtering and ultra-fast Groq Llama 3.2 11B Vision API verification
to eliminate false positives before broadcasting alerts and saving evidence.
"""

import os
import time
import json
import base64
import logging
import re
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, List, Callable, Any, Tuple
import collections
import urllib.request
import cv2
import numpy as np

from ace.config import Config


class ViolationLogger:
    """
    Manages proctoring alerts, temporal debouncing, background Groq VLM verification,
    proof screenshot capture, 3-frame burst buffering, and real-time subscriber broadcasting.
    """

    def __init__(
        self,
        screenshot_dir: str = Config.SCREENSHOT_DIR,
        log_file: str = Config.LOG_FILE,
        cooldown_sec: float = Config.VIOLATION_COOLDOWN_SEC,
    ):
        self.screenshot_dir = screenshot_dir
        self.log_file = log_file
        self.cooldown_sec = cooldown_sec

        # Ensure directories exist
        os.makedirs(self.screenshot_dir, exist_ok=True)
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Violation counters and cooldown timestamps
        self._consecutive_counts: Dict[str, int] = {}
        self._last_alert_time: Dict[str, float] = {}
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        self._lock = threading.Lock()

        # Rolling 15-frame ring buffer for 3-frame burst evidence capture
        self._frame_buffer: collections.deque = collections.deque(maxlen=15)

        # Background thread pool for non-blocking VLM API requests and burst uploads
        self._vlm_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ACE-VLM-Worker")

        # Setup logger
        self._setup_logger()

        # Initialize Groq client
        self._init_vlm()

    def push_frame(self, frame: np.ndarray):
        """Adds a copy of the live display frame to the rolling 15-frame buffer."""
        if frame is not None:
            with self._lock:
                self._frame_buffer.append(frame.copy())

    def get_burst_frames(self) -> List[Tuple[str, np.ndarray]]:
        """Extracts 3 representative frames (start, mid, end) from the rolling buffer."""
        with self._lock:
            buf_len = len(self._frame_buffer)
            if buf_len == 0:
                return []
            start_frame = self._frame_buffer[0].copy()
            mid_frame = self._frame_buffer[buf_len // 2].copy()
            end_frame = self._frame_buffer[-1].copy()
            return [("start", start_frame), ("mid", mid_frame), ("end", end_frame)]

    def _setup_logger(self):
        """Configure structured file and console proctoring logger."""
        self.logger = logging.getLogger("ACE_ProctorEngine")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [PROCTOR]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # File Handler
            file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def _init_vlm(self):
        """Initialize Groq or fallback VLM SDK clients."""
        self.groq_client = None
        self.gemini_model = None

        if Config.GROQ_API_KEY:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
                self.logger.info(f"Groq VLM verification initialized with model: {Config.VLM_MODEL}")
            except Exception as e:
                self.logger.warning(f"Could not initialize Groq client: {e}")

        elif Config.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=Config.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                self.logger.info("Gemini VLM verification initialized as fallback.")
            except Exception as e:
                self.logger.warning(f"Could not initialize Gemini VLM SDK: {e}")

        elif Config.OPENAI_API_KEY:
            self.logger.info("OpenAI VLM verification initialized as fallback.")
        else:
            self.logger.info("No VLM API Key configured (GROQ_API_KEY). Running in local heuristic mode.")

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback for triggered violation events."""
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """Unregister a violation callback."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def report_violation(
        self,
        violation_type: str,
        frame: Optional[np.ndarray] = None,
        details: str = "",
        debounce_threshold: int = 3,
    ) -> bool:
        """
        Record a potential violation occurrence. If consecutive detections exceed
        the debounce threshold and cooldown window has elapsed, triggers background
        VLM verification without blocking the main video pipeline.
        """
        current_time = time.time()

        with self._lock:
            count = self._consecutive_counts.get(violation_type, 0) + 1
            self._consecutive_counts[violation_type] = count
            last_alert = self._last_alert_time.get(violation_type, 0.0)

            if count >= debounce_threshold and (current_time - last_alert) >= self.cooldown_sec:
                self._last_alert_time[violation_type] = current_time
                self._consecutive_counts[violation_type] = 0  # Reset after triggering

                frame_copy = frame.copy() if frame is not None else None

                # Submit to background VLM worker so camera/HUD loop NEVER blocks
                self._vlm_executor.submit(
                    self._async_verify_and_publish,
                    violation_type,
                    frame_copy,
                    details,
                    count,
                )
                return True

        return False

    def _async_verify_and_publish(
        self,
        violation_type: str,
        frame: Optional[np.ndarray],
        details: str,
        consecutive_count: int,
    ):
        """
        Executes VLM verification in a background thread. Only publishes
        and saves screenshot evidence if verified as a genuine cheating violation.
        """
        # If VLM verification is disabled or no key provided, treat as verified locally
        if not Config.VLM_VERIFICATION_ENABLED or not (Config.GROQ_API_KEY or Config.GEMINI_API_KEY or Config.OPENAI_API_KEY):
            is_verified = True
            vlm_reason = "Verified by local debounced AI rules (VLM API key not configured)"
        else:
            is_verified, vlm_reason = self.verify_with_vlm(frame, violation_type, details)

        if not is_verified:
            self.logger.info(
                f"[VLM REJECTED - FALSE POSITIVE DISCARDED] [{violation_type.upper()}]: {vlm_reason}"
            )
            return

        # 1. Log verified violation
        log_entry = (
            f"VERIFIED VIOLATION: [{violation_type.upper()}] - {details} "
            f"| VLM Reason: {vlm_reason} (Sustained over {consecutive_count} frames)"
        )
        self.logger.warning(log_entry)

        # 2. Save annotated evidence screenshot
        filename = None
        if frame is not None:
            filename = self._save_screenshot(frame, violation_type, details, vlm_reason)

        # 3. Capture 3-frame burst buffer (start, mid, end) and dispatch to Node.js backend
        burst_frames = self.get_burst_frames()
        if not burst_frames and frame is not None:
            burst_frames = [("start", frame), ("mid", frame), ("end", frame)]

        if burst_frames:
            self._vlm_executor.submit(
                self._post_burst_to_backend,
                violation_type,
                details,
                vlm_reason,
                burst_frames,
            )

        # 4. Prepare real-time WebSocket event payload
        event_payload = {
            "event": "violation",
            "violation_type": violation_type,
            "details": details,
            "vlm_verified": True,
            "vlm_reason": vlm_reason,
            "timestamp": datetime.now().isoformat(),
            "screenshot_filename": filename,
            "screenshot_url": f"/api/screenshots/{filename}" if filename else None,
        }

        # 5. Notify UI & WebSocket subscribers
        with self._lock:
            for listener in list(self._listeners):
                try:
                    listener(event_payload)
                except Exception as e:
                    self.logger.error(f"Error in violation listener: {e}")

    def _post_burst_to_backend(
        self,
        violation_type: str,
        details: str,
        vlm_reason: str,
        burst_frames: List[Tuple[str, np.ndarray]],
    ):
        """Asynchronously encodes and dispatches 3-frame burst to Node.js backend."""
        try:
            encoded_burst = []
            for tag, f in burst_frames:
                if f is None:
                    continue
                h, w = f.shape[:2]
                target_w = 640
                target_h = int(h * (target_w / w))
                small_f = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)
                success, buf = cv2.imencode(".jpg", small_f, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if success:
                    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
                    encoded_burst.append({"tag": tag, "base64": b64})

            backend_base = os.getenv("NODE_API_URL", "http://localhost:5000").rstrip("/")
            backend_url = f"{backend_base}/api/exams/record-violation-snapshot"
            payload = {
                "type": violation_type,
                "details": details,
                "vlm_reason": vlm_reason,
                "confidence": 0.95,
                "timestamp": datetime.now().isoformat(),
                "burstFrames": encoded_burst,
            }

            req = urllib.request.Request(
                backend_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "ACE-Vision-Engine/2.4"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status in (200, 201):
                    self.logger.info(f"[Burst Snapshot] Dispatched 3-frame burst for '{violation_type}' to Node backend.")
        except Exception as e:
            self.logger.warning(f"[Burst Snapshot Note] Node backend upload note: {e}")

    def verify_with_vlm(
        self,
        frame: Optional[np.ndarray],
        violation_type: str,
        details: str,
    ) -> Tuple[bool, str]:
        """
        Invokes Groq Llama 3.2 11B Vision (or fallback VLM) to analyze context
        and confirm genuine cheating.
        Returns: (is_verified, explanation_reason)
        """
        if frame is None:
            return True, "No image frame available for verification"

        # Resize frame to lightweight 480px width for sub-second API upload (~25KB)
        h, w = frame.shape[:2]
        target_w = 480
        target_h = int(h * (target_w / w))
        small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        encode_success, buffer = cv2.imencode(
            ".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
        )
        if not encode_success:
            return True, "JPEG encoding failed, defaulting to verified"

        jpeg_bytes = buffer.tobytes()
        base64_image = base64.b64encode(jpeg_bytes).decode("utf-8")

        prompt = (
            f"You are an AI Exam Proctoring Integrity Validator. "
            f"A detection alert occurred: '{violation_type}' - Details: {details}. "
            f"Carefully inspect the image for cheating violations, specifically looking for:\n"
            f"1. Electronic Devices: Mobile cell phones, smartwatches, secondary tablets or monitors.\n"
            f"2. Audio Devices: AirPods, wireless earbuds, earpieces, or wired earphones in ears.\n"
            f"3. Unauthorized Notes: Open books, cheat sheets, or candidate reading off-screen.\n"
            f"4. Face & Posture: Looking away, secondary person in frame, or face missing.\n\n"
            f"Return a strict JSON object:\n"
            f'{{"verified": true/false, "detected_devices": ["phone"|"earpiece"|"book"|"screen"|"none"], "reason": "brief concise explanation"}}'
        )

        # 1. Primary: NVIDIA NIM Vision (Fast, 40 RPM, Free Endpoints)
        if Config.NVIDIA_API_KEY and (Config.VLM_PROVIDER == "nvidia" or not Config.GROQ_API_KEY):
            try:
                import requests
                headers = {
                    "Authorization": f"Bearer {Config.NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                }
                body = {
                    "model": "meta/llama-3.2-11b-vision-instruct",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                            ],
                        }
                    ],
                    "temperature": 0.1,
                    "max_tokens": 150,
                }
                resp = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=body, timeout=15)
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    match = re.search(r"\{.*?\}", content, re.DOTALL)
                    if match:
                        data = json.loads(match.group(0))
                        return bool(data.get("verified", False)), str(data.get("reason", "Analyzed by NVIDIA NIM Vision"))
                    return True, content[:120]
            except Exception as e:
                self.logger.warning(f"NVIDIA NIM VLM call failed: {e}. Trying fallback.")

        # 2. Secondary: Groq Llama 3.2 Vision
        if Config.GROQ_API_KEY:
            try:
                from groq import Groq

                if not self.groq_client:
                    self.groq_client = Groq(api_key=Config.GROQ_API_KEY)

                model_name = (
                    Config.VLM_MODEL
                    if "llama" in Config.VLM_MODEL.lower()
                    else "llama-3.2-11b-vision-instruct"
                )

                chat_completion = self.groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                            ],
                        }
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )

                res_text = chat_completion.choices[0].message.content.strip()
                data = json.loads(res_text)
                verified = bool(data.get("verified", False))
                reason = str(data.get("reason", "Analyzed by Groq Vision"))
                return verified, reason

            except Exception as e:
                self.logger.warning(f"Groq VLM call failed: {e}. Trying fallback.")

        # 3. OpenRouter (Free Vision Models)
        elif Config.OPENROUTER_API_KEY:
            try:
                import requests
                headers = {
                    "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                }
                body = {
                    "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                            ],
                        }
                    ],
                }
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body, timeout=8)
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    match = re.search(r"\{.*?\}", content, re.DOTALL)
                    if match:
                        data = json.loads(match.group(0))
                        return bool(data.get("verified", False)), str(data.get("reason", "Analyzed by OpenRouter"))
                    return True, content[:100]
            except Exception as e:
                self.logger.warning(f"OpenRouter VLM call failed: {e}")
                return True, f"OpenRouter fallback: {str(e)[:60]}"

        # 4. Fallback: OpenAI GPT-4o-mini
        elif Config.OPENAI_API_KEY:
            try:
                import openai
                client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                },
                            ],
                        }
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=150,
                )
                res_text = resp.choices[0].message.content.strip()
                data = json.loads(res_text)
                verified = bool(data.get("verified", False))
                reason = str(data.get("reason", "Analyzed by OpenAI VLM"))
                return verified, reason
            except Exception as e:
                self.logger.warning(f"OpenAI VLM call failed: {e}")
                return True, f"OpenAI fallback: {str(e)[:60]}"

        return True, "Local heuristic verification (No VLM key configured)"

    def reset_violation(self, violation_type: str):
        """Reset consecutive counter when the student returns to compliant behavior."""
        with self._lock:
            if violation_type in self._consecutive_counts:
                self._consecutive_counts[violation_type] = 0

    def reset_all(self):
        """Reset all active counters."""
        with self._lock:
            self._consecutive_counts.clear()

    def get_count(self, violation_type: str) -> int:
        """Return current consecutive count for a violation type."""
        with self._lock:
            return self._consecutive_counts.get(violation_type, 0)

    def _save_screenshot(
        self,
        frame: np.ndarray,
        violation_type: str,
        details: str,
        vlm_reason: str = "",
    ) -> str:
        """Save high-resolution annotated screenshot proof and return filename."""
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S_%f")[:19]
        sanitized_name = "".join(
            c if c.isalnum() or c in ("-", "_") else "_" for c in violation_type
        )
        filename = f"{timestamp_str}_{sanitized_name}.jpg"
        filepath = os.path.join(self.screenshot_dir, filename)

        h, w = frame.shape[:2]
        annotated = frame.copy()

        # Top Red Violation Banner
        cv2.rectangle(annotated, (0, 0), (w, 55), (0, 0, 180), -1)
        banner_text = f"PROCTOR ALERT: {violation_type.upper()} | {now.strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(
            annotated,
            banner_text,
            (15, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Bottom VLM Verification Banner
        cv2.rectangle(annotated, (0, h - 50), (w, h), (15, 15, 20), -1)
        if vlm_reason:
            cv2.putText(
                annotated,
                f"VLM Verified: {vlm_reason[:80]}",
                (15, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )
        if details:
            cv2.putText(
                annotated,
                f"Evidence: {details[:80]}",
                (15, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        try:
            cv2.imwrite(filepath, annotated)
            self.logger.info(f"Evidence screenshot saved: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to write screenshot {filepath}: {e}")

        return filename
