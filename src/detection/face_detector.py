import os
from typing import List, Optional

import cv2
import numpy as np

from .yolo_backend import DetectionResult


class FaceDetector:
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

    def __init__(self, method: str = "haar", model_dir: Optional[str] = None):
        self.method = method.lower()
        self.model_dir = model_dir or self.MODELS_DIR
        self.net = None
        self.haar_cascade = None
        self._initialize()

    def _initialize(self):
        if self.method == "dnn":
            self._init_dnn()
        elif self.method == "haar":
            self._init_haar()
        else:
            raise ValueError(f"Unsupported face detection method: {self.method}. Use 'haar' or 'dnn'.")

    def _init_haar(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade not found at {cascade_path}. Check OpenCV installation.")
        self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        if self.haar_cascade.empty():
            raise RuntimeError("Failed to load Haar cascade classifier.")

    def _init_dnn(self):
        prototxt_path = os.path.join(self.model_dir, "deploy.prototxt")
        caffemodel_path = os.path.join(self.model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
            self._download_dnn_models(prototxt_path, caffemodel_path)
        self.net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    def _download_dnn_models(self, prototxt_path: str, caffemodel_path: str):
        os.makedirs(self.model_dir, exist_ok=True)
        import urllib.request
        prototxt_url = (
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        )
        caffemodel_url = (
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
            "res10_300x300_ssd_iter_140000.caffemodel"
        )
        if not os.path.exists(prototxt_path):
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
        if not os.path.exists(caffemodel_path):
            urllib.request.urlretrieve(caffemodel_url, caffemodel_path)

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[DetectionResult]:
        if self.method == "dnn":
            return self._detect_dnn(frame, confidence_threshold)
        return self._detect_haar(frame)

    def _detect_haar(self, frame: np.ndarray) -> List[DetectionResult]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
        )
        results = []
        for x, y, w, h in faces:
            results.append(DetectionResult(
                bbox=(x, y, x + w, y + h),
                confidence=1.0,
                class_id=0,
                class_name="face",
            ))
        return results

    def _detect_dnn(self, frame: np.ndarray, confidence_threshold: float) -> List[DetectionResult]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            results.append(DetectionResult(
                bbox=(max(0, x1), max(0, y1), min(w, x2), min(h, y2)),
                confidence=float(confidence),
                class_id=0,
                class_name="face",
            ))
        return results

    def get_info(self) -> dict:
        return {"method": self.method}
