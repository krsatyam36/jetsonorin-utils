# Jetson Orin Detection Stream

![Jetson](https://img.shields.io/badge/NVIDIA-Jetson-76B900?style=for-the-badge&logo=nvidia)
![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=YOLO&logoColor=black)

Production-grade live video streaming server with runtime-togglable face, motion, and YOLO object detection — purpose-built for NVIDIA Jetson Orin.

## Features

- **Live MJPEG Stream** — Low-latency video from any V4L2 camera, viewable in any browser.
- **Interactive CLI Setup** — Choose target FPS (60/30/15), model (YOLOv8/v11), and resolution (480p/720p/1080p) before launch.
- **Three Detectors, Toggle Live** — Face (Haar/DNN), Motion (MOG2/KNN/FrameDiff), YOLO (v8/v11). All default OFF, toggle via keyboard or API.
- **Web UI with Keyboard Shortcuts** — `F` face, `M` motion, `H` human, `S` snapshot, `+`/`-` confidence, `A` all, `T` dark/light theme, `1`/`2`/`3` FPS.
- **Dark/Light Theme** — Toggle via `T` key or theme button.
- **Live Detection Counts** — Per-detector counts shown in status bar, updated every second.
- **REST API** — Status, info, health, ping, uptime, detections, snapshot, toggle, confidence, FPS control.
- **Production Hardening** — CORS, security headers, rate limiting, optional Basic Auth, request logging, JSON error handlers, graceful shutdown.
- **Docker Support** — Multi-stage Dockerfile + docker-compose for Jetson with GPU passthrough.
- **CI/CD** — GitHub Actions for ruff lint + pytest.

## Quick Start

```bash
pip install -r requirements.txt
python3 src/stream.py
```

Choose FPS → model → resolution interactively, then open `http://<JETSON_IP>:5000`.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `F` | Toggle face detection |
| `M` | Toggle motion detection |
| `H` | Toggle human (YOLO) detection |
| `S` | Download snapshot |
| `+` / `-` | Raise / lower confidence threshold |
| `A` | Toggle all detectors |
| `T` | Toggle dark/light theme |
| `1` / `2` / `3` | Set target FPS to 60 / 30 / 15 |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Web UI |
| `/video_feed` | MJPEG stream |
| `/status` | Full system status (FPS, detectors, counts) |
| `/info` | Version, uptime, config |
| `/health` | Health check (exempt from rate limit + auth) |
| `/ready` | Readiness probe (returns 503 until engine initializes) |
| `/ping` | Plain-text `pong` |
| `/uptime` | Server uptime in seconds |
| `/detections` | Current per-detector detection counts |
| `/snapshot` | Latest frame as JPEG download |
| `/toggle/<detector>` | Toggle `face`, `motion`, `human`, or `all` |
| `/confidence/<up\|down>` | Adjust confidence threshold |
| `/fps/<15\|30\|60>` | Set target FPS |

## Available Models

| Model | Best For |
|-------|----------|
| YOLO11n (Nano) | 60 FPS — fastest |
| YOLO11s (Small) | 30 FPS |
| YOLO11m (Medium) | 15 FPS |
| YOLO11l (Large) | Lower FPS, higher accuracy |
| YOLO11x (X-Large) | Maximum accuracy |
| YOLOv8n (Nano) | 60 FPS — lighter than YOLO11n |
| YOLOv8s (Small) | 30 FPS |
| YOLOv8m (Medium) | 15 FPS |
| YOLOv8l (Large) | Lower FPS |
| YOLOv8x (X-Large) | Most accurate (+ legacy compatibility) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Server port |
| `YOLO_DEVICE` | `cuda:0` | Inference device |
| `FACE_METHOD` | `haar` | `haar` or `dnn` |
| `MOTION_METHOD` | `mog2` | `mog2`, `knn`, or `frame_diff` |
| `AUTH_ENABLED` | `false` | Enable Basic Auth |
| `AUTH_USERNAME` | `admin` | Basic Auth username |
| `AUTH_PASSWORD` | `changeme` | Basic Auth password |

## Docker

```bash
docker compose up --build
```

Requires NVIDIA Container Toolkit (`nvidia-ctk`) on the host.

## Development

```bash
make install    # pip install -r requirements.txt
make run        # python3 src/stream.py
make lint       # ruff check
make test       # pytest -v
make format     # ruff format
make clean      # pyclean + coverage erase
```

## Architecture

```
src/
├── stream.py                # Flask app: routes, middleware, CLI setup, HTML/JS UI
└── detection/
    ├── __init__.py           # Public API exports
    ├── detector.py           # DetectionEngine orchestrator
    ├── yolo_backend.py       # YOLO backends (Ultralytics/ONNX/TensorRT)
    ├── face_detector.py      # Face detection (Haar/DNN)
    └── motion_detector.py    # Motion detection (MOG2/KNN/FrameDiff)
```

All backends are initialized at startup but start **disabled**. Toggling via keyboard or API enables inference without restarting the server. Motion detection runs at 1/3 frame rate to preserve FPS.

## License

MIT
