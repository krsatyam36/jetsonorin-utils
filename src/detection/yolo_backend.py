import os
import cv2
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


class YOLOModelType(Enum):
    ULTRALYTICS = "ultralytics"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


class DetectionResult:
    __slots__ = ("bbox", "confidence", "class_id", "class_name")

    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float, class_id: int, class_name: str):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    def __repr__(self) -> str:
        return f"DetectionResult(class={self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2)

    def __repr__(self):
        return f"{self.class_name} [{self.confidence:.2f}] ({self.bbox})"


class YOLOBackend(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, confidence_threshold: float) -> List[DetectionResult]:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        pass


class UltralyticsBackend(YOLOBackend):
    def __init__(self, model_path: str, model_dir: str = "models", device: str = "cuda:0"):
        self.model_path = model_path
        self.model_dir = model_dir
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            if "cuda" in self.device:
                self.model.to(self.device)
        except ImportError:
            raise ImportError(
                "ultralytics package is required for Ultralytics YOLO backend.\n"
                "Install with: pip install ultralytics"
            )

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[DetectionResult]:
        results = self.model(frame, conf=confidence_threshold, device=self.device, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                detections.append(DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                ))
        return detections

    def get_model_info(self) -> Dict:
        return {
            "type": "ultralytics",
            "model_path": self.model_path,
            "device": self.device,
        }


class ONNXBackend(YOLOBackend):
    def __init__(self, model_path: str, model_dir: str = "models", class_names: Optional[List[str]] = None):
        self.model_path = model_path
        self.class_names = class_names or COCO_CLASSES
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self._load_model()

    def _load_model(self):
        try:
            import onnxruntime as ort
            providers = [
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ]
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_name = self.session.get_outputs()[0].name
        except ImportError:
            raise ImportError(
                "onnxruntime package is required for ONNX backend.\n"
                "Install with: pip install onnxruntime-gpu"
            )

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[DetectionResult]:
        input_tensor = self._preprocess(frame)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return self._postprocess(outputs[0], frame.shape, confidence_threshold)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if len(self.input_shape) == 4:
            _, _, h, w = self.input_shape
        else:
            h, w = self.input_shape[1], self.input_shape[2]
        resized = cv2.resize(frame, (w, h))
        blob = cv2.dnn.blobFromImage(resized, 1 / 255.0, (w, h), swapRB=True, crop=False)
        return blob.astype(np.float32)

    def _postprocess(self, output: np.ndarray, original_shape: Tuple[int, ...], confidence_threshold: float) -> List[DetectionResult]:
        if output.ndim == 3:
            output = output[0]
        if output.shape[0] == 84:
            output = output.T
        orig_h, orig_w = original_shape[:2]
        detections = []
        for detection in output:
            scores = detection[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < confidence_threshold:
                continue
            cx, cy, bw, bh = detection[:4]
            x1 = int((cx - bw / 2) * orig_w)
            y1 = int((cy - bh / 2) * orig_h)
            x2 = int((cx + bw / 2) * orig_w)
            y2 = int((cy + bh / 2) * orig_h)
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            detections.append(DetectionResult(
                bbox=(max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2)),
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
            ))
        return detections

    def get_model_info(self) -> Dict:
        return {
            "type": "onnx",
            "model_path": self.model_path,
            "input_shape": self.input_shape,
            "providers": self.session.get_providers() if self.session else [],
        }


class TensorRTBackend(YOLOBackend):
    def __init__(self, model_path: str, model_dir: str = "models", class_names: Optional[List[str]] = None):
        self.model_path = model_path
        self.class_names = class_names or COCO_CLASSES
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self._load_model()

    def _load_model(self):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            with open(self.model_path, "rb") as f:
                runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
                self.engine = runtime.deserialize_cuda_engine(f.read())
                self.context = self.engine.create_execution_context()
                for i in range(self.engine.num_bindings):
                    shape = self.engine.get_binding_shape(i)
                    size = abs(trt.volume(shape))
                    dtype = trt.nptype(self.engine.get_binding_dtype(i))
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    self.allocations.append(device_mem)
                    binding = {
                        "host": host_mem,
                        "device": device_mem,
                        "shape": shape,
                        "dtype": dtype,
                        "name": self.engine.get_binding_name(i),
                    }
                    if self.engine.binding_is_input(i):
                        self.inputs.append(binding)
                    else:
                        self.outputs.append(binding)
        except ImportError:
            raise ImportError(
                "TensorRT and PyCUDA are required for TensorRT backend.\n"
                "On Jetson: they are pre-installed in the JetPack environment."
            )

    def detect(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[DetectionResult]:
        import pycuda.driver as cuda
        input_data = self._preprocess(frame)
        np.copyto(self.inputs[0]["host"], input_data.ravel())
        cuda.memcpy_htod(self.inputs[0]["device"], self.inputs[0]["host"])
        self.context.execute_v2(self.allocations)
        cuda.memcpy_dtoh(self.outputs[0]["host"], self.outputs[0]["device"])
        output_data = self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        return self._postprocess(output_data, frame.shape, confidence_threshold)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        input_shape = self.inputs[0]["shape"]
        _, _, h, w = input_shape
        resized = cv2.resize(frame, (w, h))
        blob = cv2.dnn.blobFromImage(resized, 1 / 255.0, (w, h), swapRB=True, crop=False)
        return blob.astype(np.float32)

    def _postprocess(self, output: np.ndarray, original_shape: Tuple[int, ...], confidence_threshold: float) -> List[DetectionResult]:
        if output.ndim == 3:
            output = output[0]
        if output.shape[0] == 84:
            output = output.T
        orig_h, orig_w = original_shape[:2]
        detections = []
        for detection in output:
            scores = detection[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < confidence_threshold:
                continue
            cx, cy, bw, bh = detection[:4]
            x1 = int((cx - bw / 2) * orig_w)
            y1 = int((cy - bh / 2) * orig_h)
            x2 = int((cx + bw / 2) * orig_w)
            y2 = int((cy + bh / 2) * orig_h)
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            detections.append(DetectionResult(
                bbox=(max(0, x1), max(0, y1), min(orig_w, x2), min(orig_h, y2)),
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
            ))
        return detections

    def get_model_info(self) -> Dict:
        return {
            "type": "tensorrt",
            "model_path": self.model_path,
            "input_bindings": [i["name"] for i in self.inputs],
            "output_bindings": [o["name"] for o in self.outputs],
        }


def create_yolo_backend(model_type: YOLOModelType, model_path: str, model_dir: str = "models", device: str = "cuda:0") -> YOLOBackend:
    backends = {
        YOLOModelType.ULTRALYTICS: UltralyticsBackend,
        YOLOModelType.ONNX: ONNXBackend,
        YOLOModelType.TENSORRT: TensorRTBackend,
    }
    backend_cls = backends.get(model_type)
    if backend_cls is None:
        raise ValueError(f"Unsupported YOLO model type: {model_type}. Choose from {list(backends.keys())}")
    return backend_cls(model_path=model_path, model_dir=model_dir, device=device)
