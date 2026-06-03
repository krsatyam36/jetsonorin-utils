# Jetson Orin Utilities 🚀

![Jetson](https://img.shields.io/badge/NVIDIA-Jetson-76B900?style=for-the-badge&logo=nvidia)
![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=YOLO&logoColor=black)

A growing collection of production-grade tools, scripts, and utilities for NVIDIA Jetson devices. This repository is the central hub for Jetson-based drone and computer vision development.

## ✨ Features

- **Live MJPEG Video Streaming (`src/stream.py`)**: Flask-based web server that streams hardware-accelerated video from any V4L2 camera to any device on the local network.
- **Multi-Backend YOLO Object Detection**: Supports Ultralytics (native PyTorch, YOLOv5–v11), ONNX (cross-platform), and TensorRT (Jetson-optimized) inference backends with a unified API.
- **Face Detection**: Haar cascade (lightweight, CPU) and OpenCV DNN (accurate, GPU) with auto-download of model files.
- **Motion Detection**: MOG2, KNN background subtraction, and frame-differencing with configurable sensitivity.
- **Unified Detection Pipeline**: All three detectors run on every frame and results are fused into a single annotated output with distinct visual encoding.
- **Runtime Toggle API**: Enable/disable individual detectors on the fly via HTTP endpoints without restarting.
- **JSON Status Endpoint**: Full system telemetry including FPS, per-detector state, and backend metadata.

## 🗺️ Roadmap

- **AI Inference**: Object tracking, segmentation, and pose estimation.
- **Hardware Control**: GPIO and PWM scripts for peripheral control.
- **Flight Telemetry**: MAVLink communication with Pixhawk flight controllers.
- **System Monitors**: Temperature, CPU/GPU usage, and power mode utilities.

## 🚀 Getting Started

### Prerequisites

```bash
sudo apt update
sudo apt install python3-flask python3-opencv v4l-utils -y
pip install -r requirements.txt
```

### Quick Start

```bash
python3 src/stream.py
```

Open a browser on any device on the same network and navigate to `http://<JETSON_IP>:5000`.

### Configuration via Environment Variables

All detection settings are configurable through environment variables, no code changes needed:

| Variable | Default | Description |
|---|---|---|
| `YOLO_MODEL` | `yolo11n.pt` | Model name/path (auto-downloaded for Ultralytics) |
| `YOLO_MODEL_TYPE` | `ultralytics` | Backend: `ultralytics`, `onnx`, or `tensorrt` |
| `YOLO_DEVICE` | `cuda:0` | Inference device (CPU or CUDA GPU) |
| `ENABLE_YOLO` | `1` | Enable/disable YOLO detection |
| `ENABLE_FACE` | `1` | Enable/disable face detection |
| `ENABLE_MOTION` | `1` | Enable/disable motion detection |
| `FACE_METHOD` | `haar` | Face detection: `haar` or `dnn` |
| `MOTION_METHOD` | `mog2` | Motion detection: `mog2`, `knn`, or `frame_diff` |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence for detections (0.0–1.0) |
| `MOTION_MIN_AREA` | `500` | Minimum contour area (px) for motion regions |

Example:
```bash
YOLO_MODEL=yolo11s.pt ENABLE_FACE=0 python3 src/stream.py
```

## 📡 API Endpoints

| Endpoint | Description |
|---|---|
| `/` | Web page with live detection stream |
| `/video_feed` | Raw MJPEG video stream |
| `/status` | JSON status of all detectors and FPS |
| `/toggle/yolo` | Toggle YOLO detection on/off |
| `/toggle/face` | Toggle face detection on/off |
| `/toggle/motion` | Toggle motion detection on/off |

## 🏗️ Architecture

```
jetsonorin-utils/
├── .gitignore
├── README.md
├── requirements.txt
├── models/                      # Downloaded model weights (gitignored)
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── deploy.prototxt
└── src/
    ├── stream.py                # Flask server + detection integration
    └── detection/
        ├── __init__.py          # Public API exports
        ├── detector.py          # DetectionEngine orchestrator
        ├── yolo_backend.py      # YOLO backends (Ultralytics/ONNX/TensorRT)
        ├── face_detector.py     # Face detection (Haar/DNN)
        └── motion_detector.py   # Motion detection (MOG2/KNN/FrameDiff)
```

### Detection Pipeline

Each frame captured from the camera passes through this pipeline:

```
Camera Frame
    │
    ├──► YOLO Detection (green boxes)
    │     Supports Ultralytics / ONNX / TensorRT models
    │     User-selectable via YOLO_MODEL_TYPE env var
    │
    ├──► Face Detection (blue boxes)
    │     Haar cascade (CPU) or DNN SSD (GPU)
    │
    ├──► Motion Detection (red boxes)
    │     MOG2 / KNN / Frame differencing
    │
    └──► Annotated Frame + Detection Results
          FPS overlay + per-detector counts
          Returned to stream.py for MJPEG encoding
```

All detectors run on the **original unmodified frame** to avoid annotation artifacts. Bounding boxes from all detectors are drawn together on a single output frame with color-coded labels.

## 🤖 Robotics Applications

This architecture is designed for autonomous robot development:

- **Object Avoidance**: YOLO detects obstacles (people, vehicles, furniture).
- **Person Tracking**: Face detection + motion tracking for follow-me behavior.
- **Surveillance**: Motion detection triggers recording or alerts.
- **Drone Navigation**: Frame-differencing motion detection for optical flow.
- **Multi-Sensor Fusion**: All detection results are available as structured data for decision-making algorithms.

## 🔧 Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run with custom YOLO model
YOLO_MODEL=yolo8n.pt python3 src/stream.py

# Run with ONNX model
YOLO_MODEL_TYPE=onnx YOLO_MODEL=model.onnx python3 src/stream.py

# Run with only motion detection
ENABLE_YOLO=0 ENABLE_FACE=0 python3 src/stream.py
```

## 📄 License

MIT
