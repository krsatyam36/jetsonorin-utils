import os
import time
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .yolo_backend import YOLOModelType, YOLOBackend, DetectionResult, create_yolo_backend
from .face_detector import FaceDetector
from .motion_detector import MotionDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DetectionEngine")

COLORS = {
    "yolo": (0, 255, 0),
    "face": (255, 0, 0),
    "motion": (0, 0, 255),
    "fps": (255, 255, 255),
    "label_bg": (0, 0, 0),
}


class DetectionEngine:
    def __init__(
        self,
        yolo_model: str = "yolo11n.pt",
        yolo_model_type: YOLOModelType = YOLOModelType.ULTRALYTICS,
        yolo_device: str = "cuda:0",
        enable_yolo: bool = True,
        enable_face: bool = True,
        enable_motion: bool = True,
        face_method: str = "haar",
        motion_method: str = "mog2",
        confidence_threshold: float = 0.5,
        model_dir: str = "models",
        motion_min_area: int = 500,
    ):
        self.confidence_threshold = confidence_threshold
        self.model_dir = model_dir
        self.enable_yolo = enable_yolo
        self.enable_face = enable_face
        self.enable_motion = enable_motion

        self.yolo_backend: Optional[YOLOBackend] = None
        self.face_detector: Optional[FaceDetector] = None
        self.motion_detector: Optional[MotionDetector] = None

        if self.enable_yolo:
            self._init_yolo(yolo_model, yolo_model_type, yolo_device)
        if self.enable_face:
            self._init_face(face_method)
        if self.enable_motion:
            self._init_motion(motion_method, motion_min_area)

        self._fps = 0.0
        self._prev_time = time.time()
        self._frame_count = 0
        self._detection_counts = {}

    def _init_yolo(self, model: str, model_type: YOLOModelType, device: str):
        model_path = self._resolve_model_path(model, model_type)
        try:
            self.yolo_backend = create_yolo_backend(
                model_type=model_type,
                model_path=model_path,
                model_dir=self.model_dir,
                device=device,
            )
            logger.info("YOLO backend initialized: %s (%s)", model_path, model_type.value)
        except Exception as e:
            logger.warning("YOLO backend failed to initialize: %s. YOLO detection disabled.", e)
            self.enable_yolo = False

    def _resolve_model_path(self, model: str, model_type: YOLOModelType) -> str:
        if model_type == YOLOModelType.ULTRALYTICS:
            return model
        if os.path.isabs(model):
            return model
        return os.path.join(self.model_dir, model)

    def _init_face(self, method: str):
        try:
            self.face_detector = FaceDetector(method=method, model_dir=self.model_dir)
            logger.info("Face detector initialized: %s", method)
        except Exception as e:
            logger.warning("Face detector failed to initialize: %s. Face detection disabled.", e)
            self.enable_face = False

    def _init_motion(self, method: str, min_area: int):
        try:
            self.motion_detector = MotionDetector(method=method, min_area=min_area)
            logger.info("Motion detector initialized: %s", method)
        except Exception as e:
            logger.warning("Motion detector failed to initialize: %s. Motion detection disabled.", e)
            self.enable_motion = False

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        self._update_fps()
        all_detections: Dict[str, List[DetectionResult]] = {}
        annotated = frame.copy()

        if self.enable_yolo and self.yolo_backend is not None:
            try:
                yolo_results = self.yolo_backend.detect(frame, self.confidence_threshold)
                all_detections["yolo"] = yolo_results
                annotated = self._draw_detections(annotated, yolo_results, COLORS["yolo"])
            except Exception as e:
                logger.error("YOLO detection error: %s", e)

        if self.enable_face and self.face_detector is not None:
            try:
                face_results = self.face_detector.detect(frame, self.confidence_threshold)
                all_detections["face"] = face_results
                annotated = self._draw_detections(annotated, face_results, COLORS["face"])
            except Exception as e:
                logger.error("Face detection error: %s", e)

        if self.enable_motion and self.motion_detector is not None:
            try:
                motion_results = self.motion_detector.detect(frame)
                all_detections["motion"] = motion_results
                annotated = self._draw_detections(annotated, motion_results, COLORS["motion"])
            except Exception as e:
                logger.error("Motion detection error: %s", e)

        annotated = self._draw_stats(annotated, all_detections)
        self._detection_counts = {k: len(v) for k, v in all_detections.items()}
        return annotated, all_detections

    def _draw_detections(
        self, frame: np.ndarray, detections: List[DetectionResult], color: Tuple[int, int, int],
    ) -> np.ndarray:
        for det in detections:
            c = self._class_color(det.class_id, color)
            cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), c, 2)
            text = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (det.x1, det.y1 - th - 6), (det.x1 + tw + 4, det.y1), c, -1)
            cv2.putText(
                frame, text, (det.x1 + 2, det.y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return frame

    @staticmethod
    def _class_color(class_id: int, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
        palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 255, 0), (255, 128, 0), (128, 0, 255), (255, 0, 128),
            (0, 128, 255), (0, 255, 128), (128, 128, 255), (255, 128, 128), (128, 255, 128),
        ]
        return palette[class_id % len(palette)] if class_id >= 0 else fallback

    def _draw_stats(self, frame: np.ndarray, all_detections: Dict[str, List[DetectionResult]]) -> np.ndarray:
        total = sum(len(dets) for dets in all_detections.values())

        stats = [
            f"FPS: {self._fps:.1f}",
            f"Detections: {total}",
        ]
        for det_type, dets in all_detections.items():
            stats.append(f"  {det_type}: {len(dets)}")

        for i, text in enumerate(stats):
            y = 30 + i * 22
            cv2.putText(
                frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["fps"], 2, cv2.LINE_AA,
            )
        return frame

    def _update_fps(self):
        self._frame_count += 1
        now = time.time()
        if now - self._prev_time >= 1.0:
            self._fps = self._frame_count / (now - self._prev_time)
            self._frame_count = 0
            self._prev_time = now

    @property
    def fps(self) -> float:
        return self._fps

    def get_status(self) -> Dict:
        return {
            "fps": self._fps,
            "yolo_enabled": self.enable_yolo,
            "face_enabled": self.enable_face,
            "motion_enabled": self.enable_motion,
            "yolo_info": self.yolo_backend.get_model_info() if self.yolo_backend else None,
            "face_info": self.face_detector.get_info() if self.face_detector else None,
            "motion_info": self.motion_detector.get_info() if self.motion_detector else None,
            "detection_counts": self._detection_counts,
        }

    def set_confidence_threshold(self, threshold: float):
        self.confidence_threshold = max(0.0, min(1.0, threshold))

    def toggle_yolo(self, enabled: bool):
        self.enable_yolo = enabled

    def toggle_face(self, enabled: bool):
        self.enable_face = enabled

    def toggle_motion(self, enabled: bool):
        self.enable_motion = enabled
