import cv2
import numpy as np
from typing import List, Optional, Tuple

from .yolo_backend import DetectionResult


class MotionDetector:
    def __init__(
        self,
        method: str = "mog2",
        min_area: int = 500,
        threshold: float = 25.0,
        blur_ksize: int = 21,
        dilate_iters: int = 2,
        history: int = 500,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
    ):
        self.min_area = min_area
        self.threshold = threshold
        self.blur_ksize = blur_ksize
        self.dilate_iters = dilate_iters
        self.method = method.lower()
        self._initialize(history, var_threshold, detect_shadows)
        self._prev_gray = None

    def _initialize(self, history: int, var_threshold: float, detect_shadows: bool):
        if self.method == "mog2":
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history, varThreshold=var_threshold, detectShadows=detect_shadows,
            )
        elif self.method == "knn":
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=history, dist2Threshold=var_threshold, detectShadows=detect_shadows,
            )
        elif self.method == "frame_diff":
            self.subtractor = None
        else:
            raise ValueError(f"Unsupported motion detection method: {self.method}. Use 'mog2', 'knn', or 'frame_diff'.")

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        if self.method == "frame_diff":
            return self._detect_frame_diff(frame)
        return self._detect_subtractor(frame)

    def _detect_subtractor(self, frame: np.ndarray) -> List[DetectionResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        fg_mask = self.subtractor.apply(blurred)
        return self._extract_motion_regions(fg_mask, frame.shape[:2])

    def _detect_frame_diff(self, frame: np.ndarray) -> List[DetectionResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        if self._prev_gray is None:
            self._prev_gray = blurred
            return []
        diff = cv2.absdiff(self._prev_gray, blurred)
        _, fg_mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        self._prev_gray = blurred
        return self._extract_motion_regions(fg_mask, frame.shape[:2])

    def _extract_motion_regions(self, fg_mask: np.ndarray, frame_shape: Tuple[int, int]) -> List[DetectionResult]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=self.dilate_iters)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            results.append(DetectionResult(
                bbox=(x, y, x + w, y + h),
                confidence=min(1.0, area / 5000.0),
                class_id=0,
                class_name="motion",
            ))
        return results

    def reset(self):
        if self.subtractor is not None:
            self.subtractor.clear()
        self._prev_gray = None

    def get_info(self) -> dict:
        return {
            "method": self.method,
            "min_area": self.min_area,
            "threshold": self.threshold,
        }
