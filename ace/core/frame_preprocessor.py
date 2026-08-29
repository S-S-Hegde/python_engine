"""
ace/core/frame_preprocessor.py - Lightweight OpenCV Preprocessing Pipeline.
Cleans raw webcam frames to remove noise, lighting variations, and static room clutter.

Pipeline Stages:
1. CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB space to normalize backlight/dark conditions.
2. Bilateral Filtering to remove webcam sensor grain while keeping crisp facial/silhouette edges.
3. MOG2 Background Subtraction to isolate the candidate silhouette and mask out room clutter.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2


class FramePreprocessor:
    """
    High-performance, CPU-optimized OpenCV frame preprocessor for proctoring feeds.
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        bilateral_d: int = 7,
        bilateral_sigma_color: float = 50.0,
        bilateral_sigma_space: float = 50.0,
        mog2_history: int = 500,
        mog2_var_threshold: float = 16.0,
        detect_shadows: bool = False,
    ):
        # 1. CLAHE in LAB Color Space
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_grid_size
        )

        # 2. Bilateral Filter Parameters
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space

        # 3. MOG2 Background Subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=mog2_history,
            varThreshold=mog2_var_threshold,
            detectShadows=detect_shadows
        )

        # Morphological kernel for noise cleanup
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def apply_clahe(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Normalizes illumination across the frame without distorting color balance
        by applying CLAHE exclusively to the L (Lightness) channel in LAB color space.
        """
        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to Lightness channel
        l_enhanced = self.clahe.apply(l_channel)
        
        # Merge back and convert to BGR
        enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def apply_bilateral_filter(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Smoothes camera sensor noise and compression grain while preserving sharp edges.
        """
        return cv2.bilateralFilter(
            bgr_frame,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )

    def extract_foreground_mask(self, bgr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies MOG2 background subtraction with morphological cleanup
        to isolate the moving candidate from static room clutter.
        
        Returns:
            clean_mask: Binary foreground mask (255 = foreground candidate, 0 = static background)
            segmented_frame: BGR frame with background blacked out.
        """
        fg_mask = self.bg_subtractor.apply(bgr_frame)

        # Remove salt-and-pepper noise and fill internal holes
        clean_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, self.morph_kernel)

        # Mask original frame
        segmented_frame = cv2.bitwise_and(bgr_frame, bgr_frame, mask=clean_mask)
        return clean_mask, segmented_frame

    def process(self, raw_frame: np.ndarray) -> Dict[str, Any]:
        """
        Executes full preprocessing pipeline on incoming webcam frame.
        
        Returns dictionary containing:
            - preprocessed_frame: Denoised and illumination-normalized BGR frame
            - fg_mask: Cleaned binary foreground mask
            - candidate_segmented: Candidate visual isolated from background
            - candidate_bbox: Bounding box (x, y, w, h) of primary moving subject
            - motion_detected: Boolean indicating active candidate movement
        """
        if raw_frame is None or raw_frame.size == 0:
            return {
                "success": False,
                "error": "EMPTY_FRAME"
            }

        # Step 1: Illumination Normalization (CLAHE)
        equalized = self.apply_clahe(raw_frame)

        # Step 2: Edge-Preserving Grain Smoothing (Bilateral Filter)
        denoised = self.apply_bilateral_filter(equalized)

        # Step 3: Background Subtraction & Foreground Isolation
        fg_mask, segmented = self.extract_foreground_mask(denoised)

        # Find candidate bounding box from foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_bbox = None
        motion_detected = False

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            # Filter trivial noise contours (< 1% of frame area)
            h, w = raw_frame.shape[:2]
            if area > (w * h * 0.01):
                candidate_bbox = cv2.boundingRect(largest_contour)
                motion_detected = True

        return {
            "success": True,
            "preprocessed_frame": denoised,
            "fg_mask": fg_mask,
            "candidate_segmented": segmented,
            "candidate_bbox": candidate_bbox,
            "motion_detected": motion_detected,
        }
