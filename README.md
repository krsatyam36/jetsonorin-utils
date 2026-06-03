# Jetson Orin Detection Stream

![Jetson](https://img.shields.io/badge/NVIDIA-Jetson-76B900?style=for-the-badge&logo=nvidia)
![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-00FFFF?style=for-the-badge&logo=YOLO&logoColor=black)

Production-grade live video streaming with runtime-togglable face, motion, and YOLO object detection — built for NVIDIA Jetson Orin.

```
pip install -r requirements.txt
python3 src/stream.py
# Open http://<JETSON_IP>:5000
```

---

## StreamServer

```python
+run(host, port)                                 # Start Flask server
+GET /                                           # Web UI (HTML + JS)
+GET /video_feed                                 # MJPEG stream (multipart/x-mixed-replace)
+GET /snapshot                                   # Latest frame as JPEG download
```

```
         +---------+
         | Browser |◄── MJPEG ── /video_feed
         +---------+        │
              │             ├── /snapshot
              ▼             └── /status (JSON)
         +------------+
         | StreamServer|──► DetectionEngine.process_frame()
         +------------+
```

Status, info, health, ping, uptime, detections, toggle, confidence, fps, ready

## DetectionEngine

```python
+process_frame(frame) : (np.ndarray, dict)      # Run all enabled detectors, return annotated frame
+get_status() : dict                             # FPS, enabled flags, detection counts
+set_confidence_threshold(threshold) : void        # Clamp 0.0–1.0
+toggle_yolo(enabled) : void
+toggle_face(enabled) : void
+toggle_motion(enabled) : void
+fps : float                                      # Property: current FPS
```

```
                    +------------------+
                    | DetectionEngine  |
                    +------------------+
                    | enable_yolo      |
                    | enable_face      |
                    | enable_motion    |
                    +--------+---------+
                             │
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
         +----------+  +----------+  +-------------+
         | YOLO     |  | Face     |  | Motion      │
         | Backend  |  | Detector |  | Detector    │
         +----------+  +----------+  +-------------+
         | detect() |  | detect() |  | detect()    |
         +----------+  +----------+  +-------------+
```

All backends initialized at startup, default **OFF**. Toggling enables inference without restart.

## YOLOBackend

```python
+detect(frame, confidence_threshold) : List[DetectionResult]
+get_model_info() : dict
```

**UltralyticsBackend** — Native PyTorch (YOLOv5–v11)
```python
+__init__(model_path, model_dir, device)
```

**ONNXBackend** — Cross-platform GPU inference
```python
+__init__(model_path, model_dir, class_names)
```

**TensorRTBackend** — Jetson-optimized inference
```python
+__init__(model_path, model_dir, class_names)
```

### DetectionResult

```python
+bbox : Tuple[int, int, int, int]                # (x1, y1, x2, y2)
+confidence : float
+class_id : int
+class_name : str
+x1, y1, x2, y2 : int                            # Properties
+width, height : int
+area : int
+center : Tuple[int, int]
```

## FaceDetector

```python
+detect(frame, confidence_threshold) : List[DetectionResult]
+get_info() : dict
```

| Method | Backend |
|--------|---------|
| `haar` | Haar cascade (CPU, lightweight) |
| `dnn`  | OpenCV SSD (GPU, accurate) |

Models auto-downloaded on first use.

## MotionDetector

```python
+detect(frame) : List[DetectionResult]            # Returns motion regions as DetectionResult list
+reset() : void                                   # Clear background model
+get_info() : dict
```

| Method | Description |
|--------|-------------|
| `mog2` | Adaptive Gaussian mixture (OpenCV) |
| `knn`  | K-nearest neighbours background subtraction |
| `frame_diff` | Frame differencing (cheapest, no background model) |

Motion detection runs every **3rd frame** to preserve FPS.

---

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

## Models

| YOLO11 | YOLOv8 | Target FPS |
|--------|--------|------------|
| 11n (Nano) | 8n (Nano) | 60 |
| 11s (Small) | 8s (Small) | 30 |
| 11m (Medium) | 8m (Medium) | 15 |
| 11l (Large) | 8l (Large) | 10–25 |
| 11x (X-Large) | 8x (X-Large) | 5–15 |

## Development

```bash
make install    # pip install -r requirements.txt
make run        # python3 src/stream.py
make lint       # ruff check
make test       # pytest -v
make format     # ruff format
make clean      # pyclean + coverage erase
docker compose up --build  # Docker (requires nvidia-ctk)
```

## License

MIT
