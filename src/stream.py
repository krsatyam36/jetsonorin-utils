import os
import cv2
from flask import Flask, Response, jsonify

from detection import DetectionEngine, YOLOModelType

app = Flask(__name__)

engine = DetectionEngine(
    yolo_model=os.getenv("YOLO_MODEL", "yolo11n.pt"),
    yolo_model_type=YOLOModelType(os.getenv("YOLO_MODEL_TYPE", "ultralytics")),
    yolo_device=os.getenv("YOLO_DEVICE", "cuda:0"),
    enable_yolo=os.getenv("ENABLE_YOLO", "1") == "1",
    enable_face=os.getenv("ENABLE_FACE", "1") == "1",
    enable_motion=os.getenv("ENABLE_MOTION", "1") == "1",
    face_method=os.getenv("FACE_METHOD", "haar"),
    motion_method=os.getenv("MOTION_METHOD", "mog2"),
    confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
    motion_min_area=int(os.getenv("MOTION_MIN_AREA", "500")),
)


def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        annotated, detections = engine.process_frame(frame)

        ret, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


@app.route("/")
def index():
    status = engine.get_status()
    detection_status = " | ".join([
        f"YOLO: {'ON' if status['yolo_enabled'] else 'OFF'}",
        f"Face: {'ON' if status['face_enabled'] else 'OFF'}",
        f"Motion: {'ON' if status['motion_enabled'] else 'OFF'}",
    ])
    return f"""
    <html>
        <head>
            <title>Jetson Detection Stream</title>
            <style>
                body {{ background-color: #222; color: white; text-align: center; font-family: sans-serif; }}
                img {{ max-width: 100%; height: auto; border: 3px solid #444; border-radius: 8px; }}
                .status {{ color: #0f0; font-size: 14px; margin: 10px; }}
            </style>
        </head>
        <body>
            <h1>Jetson Detection Stream</h1>
            <p class="status">{detection_status}</p>
            <img src="/video_feed">
            <p><a href="/status" style="color: #88f;">View JSON Status</a></p>
        </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify(engine.get_status())


@app.route("/toggle/<detector>")
def toggle(detector: str):
    status_data = engine.get_status()
    if detector == "yolo":
        engine.toggle_yolo(not status_data["yolo_enabled"])
    elif detector == "face":
        engine.toggle_face(not status_data["face_enabled"])
    elif detector == "motion":
        engine.toggle_motion(not status_data["motion_enabled"])
    else:
        return jsonify({"error": f"Unknown detector: {detector}"}), 400
    return jsonify(engine.get_status())


if __name__ == "__main__":
    status = engine.get_status()
    print("Detection Engine initialized:")
    print(f"  YOLO:  {'ON' if status['yolo_enabled'] else 'OFF'} ({status.get('yolo_info', {}).get('model_path', 'N/A')})")
    print(f"  Face:  {'ON' if status['face_enabled'] else 'OFF'} ({status.get('face_info', {}).get('method', 'N/A')})")
    print(f"  Motion: {'ON' if status['motion_enabled'] else 'OFF'} ({status.get('motion_info', {}).get('method', 'N/A')})")
    print("Starting server... Access it in your browser at http://<JETSON_IP>:5000")
    print("  /video_feed - MJPEG video stream")
    print("  /status     - JSON detection status")
    print("  /toggle/yolo, /toggle/face, /toggle/motion - toggle detectors on/off")
    app.run(host="0.0.0.0", port=5000, debug=False)
