import os
import signal
import subprocess
import sys
import time
import uuid
from collections import defaultdict

import cv2
from flask import Flask, Response, jsonify, request

from detection import DetectionEngine, YOLOModelType

app = Flask(__name__)

engine = None
target_fps = 30
cfg = {}
_latest_jpeg = None
_start_time = time.time()
_shutdown = False
VERSION = "0.1.0"

# ── Rate Limiter ─────────────────────────────────────────────────────────────

class RateLimiter:
    def __init__(self, max_requests: int = 100, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self._clients: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        self._clients[key] = [t for t in self._clients[key] if now - t < self.window]
        if len(self._clients[key]) >= self.max_requests:
            return False
        self._clients[key].append(now)
        return True


rate_limiter = RateLimiter()

import base64

AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
AUTH_USER = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASS = os.getenv("AUTH_PASSWORD", "changeme")


def handle_signal(signum, frame):
    global _shutdown
    if _shutdown:
        sys.exit(1)
    _shutdown = True
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


# ── Port killer ──────────────────────────────────────────────────────────────

def kill_port(port: int = 5000, max_retries: int = 5):
    import socket
    for attempt in range(max_retries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout:
                pids = [int(pid) for pid in result.stdout.strip().split()]
                sig = signal.SIGKILL if attempt >= max_retries - 2 else signal.SIGTERM
                for pid in pids:
                    os.kill(pid, sig)
                print(f"Killed {len(pids)} process(es) on port {port} with {sig.name}")
        except Exception as e:
            print(f"Failed to free port {port}: {e}", file=sys.stderr)
        time.sleep(0.5)


# ── Interactive Setup ────────────────────────────────────────────────────────

MODEL_CATALOG = [
    {"id": "yolo11n.pt", "label": "YOLO11n (Nano) — fastest",     "fps_estimate": (60, 200)},
    {"id": "yolo11s.pt", "label": "YOLO11s (Small)",               "fps_estimate": (30,  80)},
    {"id": "yolo11m.pt", "label": "YOLO11m (Medium)",              "fps_estimate": (15,  40)},
    {"id": "yolo11l.pt", "label": "YOLO11l (Large)",               "fps_estimate": (10,  25)},
    {"id": "yolo11x.pt", "label": "YOLO11x (X-Large) — most accurate", "fps_estimate": (5, 15)},
]

RES_PRESETS = [
    (640,  480,  "640×480   (SD)"),
    (1280, 720,  "1280×720  (HD)"),
    (1920, 1080, "1920×1080 (Full HD)"),
]

FPS_TARGETS = [
    (60, "60 FPS — Smooth (best for fast motion)"),
    (30, "30 FPS — Standard live stream"),
    (15, "15 FPS — High accuracy, lower bandwidth"),
]


def _pick(options, prompt, default=0):
    for i, (_, desc) in enumerate(options):
        print(f"  {i+1}) {desc}")
    idx = input(prompt).strip()
    if idx.isdigit() and 1 <= int(idx) <= len(options):
        return options[int(idx) - 1]
    return options[default]


def interactive_setup():
    global target_fps
    print("=" * 60)
    print("  JETSON DETECTION STREAM — Interactive Setup")
    print("=" * 60)

    print("\n[1] Target frame rate")
    fps_val, _ = _pick(FPS_TARGETS, f"  Choose [1-{len(FPS_TARGETS)}] (default: 2): ", default=1)
    target_fps = fps_val

    best_match = MODEL_CATALOG[0]
    for m in MODEL_CATALOG:
        low, high = m["fps_estimate"]
        if low <= target_fps <= high:
            best_match = m
            break
    if target_fps < best_match["fps_estimate"][0]:
        best_match = MODEL_CATALOG[-1]

    print(f"\n[2] Detection model  (recommended for {target_fps} FPS: {best_match['label']})")
    print("  Available models:")
    for i, m in enumerate(MODEL_CATALOG):
        tag = "  ← recommended" if m["id"] == best_match["id"] else ""
        print(f"  {i+1}) {m['label']}{tag}")
    idx = input(f"  Choose [1-{len(MODEL_CATALOG)}] (default: auto): ").strip()
    if idx.isdigit() and 1 <= int(idx) <= len(MODEL_CATALOG):
        chosen_model = MODEL_CATALOG[int(idx) - 1]["id"]
    else:
        chosen_model = best_match["id"]
    print(f"  → {chosen_model}")

    print(f"\n[3] Capture resolution")
    res = _pick(RES_PRESETS, f"  Choose [1-{len(RES_PRESETS)}] (default: 2): ", default=1)

    print(f"\n[4] Detectors (all OFF by default — toggle live with keyboard)")
    print("     F = Face detection")
    print("     M = Motion detection")
    print("     H = Human (YOLO) detection")
    print()

    return {
        "model": chosen_model,
        "width": res[0],
        "height": res[1],
    }


# ── FPS limiter ──────────────────────────────────────────────────────────────

class FPSLimiter:
    def __init__(self):
        self._last = time.perf_counter()

    def wait(self):
        global target_fps
        period = 1.0 / target_fps if target_fps > 0 else 0
        if period <= 0:
            return
        elapsed = time.perf_counter() - self._last
        remaining = period - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last = time.perf_counter()


# ── Video Feed ───────────────────────────────────────────────────────────────

def generate_frames():
    global engine, cfg
    width, height = cfg["width"], cfg["height"]
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    limiter = FPSLimiter()

    while True:
        success, frame = cap.read()
        if not success:
            break

        limiter.wait()
        annotated, detections = engine.process_frame(frame)

        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, now_str, (annotated.shape[1] - 220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        global _latest_jpeg
        _latest_jpeg = frame_bytes

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


# ── Flask Routes ─────────────────────────────────────────────────────────────

HTML_PAGE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Jetson Detection Stream</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif;
    text-align: center; min-height: 100vh; display: flex; flex-direction: column;
  }
  h1 { font-size: 1.4rem; margin: 12px 0 4px; letter-spacing: 1px; }
  .status-bar {
    display: flex; justify-content: center; gap: 18px; flex-wrap: wrap;
    margin: 6px 0 10px; font-size: 0.9rem;
  }
  .status-bar span { padding: 3px 12px; border-radius: 4px; background: #222; }
  .on  { color: #4f4; border-left: 3px solid #4f4; }
  .off { color: #888; border-left: 3px solid #555; }
  .container { flex: 1; display: flex; align-items: center; justify-content: center; padding: 0 12px; }
  img {
    max-width: 100%; max-height: 85vh; border-radius: 8px;
    box-shadow: 0 0 24px rgba(0,0,0,0.6); cursor: pointer;
  }
  .help {
    margin: 10px 0 14px; font-size: 0.8rem; color: #888;
    letter-spacing: 0.5px;
  }
  .help kbd {
    background: #333; color: #ddd; padding: 1px 8px; border-radius: 3px;
    font-family: monospace; border: 1px solid #555; margin: 0 2px;
  }
  .help .active { color: #4f4; }
  body.light { background: #eef; color: #222; }
  body.light .status-bar span { background: #ddd; }
  body.light .help { color: #666; }
  body.light .help kbd { background: #ccc; color: #333; border-color: #999; }
  body.light .on { color: #070; }
  #theme-btn {
    position: fixed; top: 10px; left: 14px; cursor: pointer;
    background: rgba(0,0,0,0.65); border: none; color: #eee;
    padding: 4px 12px; border-radius: 4px; font-size: 1rem;
  }
  body.light #theme-btn { background: rgba(0,0,0,0.15); color: #333; }
  #fps-badge {
    position: fixed; top: 10px; right: 14px;
    background: rgba(0,0,0,0.65); padding: 4px 12px; border-radius: 4px;
    font-size: 0.85rem; font-family: monospace;
  }
</style>
</head>
<body>

<button id="theme-btn" onclick="document.body.classList.toggle('light');this.textContent=document.body.classList.contains('light')?'☀':'🌙'">🌙</button>
<h1>Jetson Detection Stream</h1>
<p style="font-size:0.8rem;color:#888;margin-bottom:4px" id="info-line">{model} @ {width}x{height}</p>

<div class="status-bar" id="status-bar">
  <span id="s-face" class="off">Face [F]</span>
  <span id="s-motion" class="off">Motion [M]</span>
  <span id="s-human" class="off">Human [H]</span>
</div>

<div class="container">
  <img id="stream" src="/video_feed" alt="Live stream">
</div>

<div class="help">
  Click the video then press &nbsp;
  <kbd>F</kbd> face &nbsp; <kbd>M</kbd> motion &nbsp; <kbd>H</kbd> human &nbsp;
  <kbd>S</kbd> snap &nbsp; <kbd>+</kbd><kbd>-</kbd> conf &nbsp;
  <kbd>A</kbd> all &nbsp; <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> fps &nbsp;
  <kbd>T</kbd> theme
</div>

<div id="uptime-badge" style="position:fixed;bottom:10px;left:14px;background:rgba(0,0,0,0.65);padding:4px 12px;border-radius:4px;font-size:0.8rem;font-family:monospace;">--</div>
<div id="fps-badge">-- FPS</div>
<div id="conf-badge" style="position:fixed;top:34px;right:14px;background:rgba(0,0,0,0.65);padding:4px 12px;border-radius:4px;font-size:0.85rem;font-family:monospace;">Conf: 0.50</div>

<script>
const BASE = '';
const badges = {
  face:  document.getElementById('s-face'),
  motion: document.getElementById('s-motion'),
  human: document.getElementById('s-human'),
};
const fpsEl = document.getElementById('fps-badge');
const uptimeEl = document.getElementById('uptime-badge');

async function setFps(val) {
  await fetch(BASE + '/fps/' + val);
  fpsEl.textContent = val + ' FPS (target)';
}

async function toggle(detector) {
  const r = await fetch(BASE + '/toggle/' + detector);
  const data = await r.json();
  updateUI(data);
}

function updateUI(data) {
  const labels = { face: 'Face', motion: 'Motion', human: 'Human' };
  const props = { face: 'face_enabled', motion: 'motion_enabled', human: 'yolo_enabled' };
  const countKeys = { face: 'face', motion: 'motion', human: 'yolo' };
  for (const [key, label] of Object.entries(labels)) {
    const el = badges[key];
    const on = data[props[key]];
    const cnt = data.detection_counts ? data.detection_counts[countKeys[key]] : 0;
    el.className = on ? 'on' : 'off';
    el.textContent = label + ' [' + key.toUpperCase() + ']' + (on && cnt > 0 ? ' ' + cnt : '');
  }
  if (data.fps != null) fpsEl.textContent = data.fps.toFixed(1) + ' FPS';
}

document.addEventListener('keydown', function(e) {
  const key = e.key.toLowerCase();
  if (key === 'f') { e.preventDefault(); toggle('face'); }
  else if (key === 'm') { e.preventDefault(); toggle('motion'); }
  else if (key === 'h') { e.preventDefault(); toggle('human'); }
  else if (key === 's') {
    e.preventDefault();
    const a = document.createElement('a');
    a.href = '/snapshot';
    a.download = 'snapshot_' + Date.now() + '.jpg';
    a.click();
  }
  else if (key === '=' || key === '+') { e.preventDefault(); adjustConf('up'); }
  else if (key === '-') { e.preventDefault(); adjustConf('down'); }
  else if (key === 'a') { e.preventDefault(); toggle('all'); }
  else if (key === '1') { e.preventDefault(); setFps(60); }
  else if (key === '2') { e.preventDefault(); setFps(30); }
  else if (key === '3') { e.preventDefault(); setFps(15); }
  else if (key === 't') {
    e.preventDefault();
    document.body.classList.toggle('light');
    document.getElementById('theme-btn').textContent =
      document.body.classList.contains('light') ? '☀' : '🌙';
  }
});

async function adjustConf(dir) {
  const r = await fetch(BASE + '/confidence/' + dir);
  const d = await r.json();
  document.getElementById('conf-badge').textContent = 'Conf: ' + d.confidence.toFixed(2);
}

setInterval(async () => {
  try {
    const r = await fetch(BASE + '/status');
    const data = await r.json();
    updateUI(data);
  } catch {}
  try {
    const r = await fetch(BASE + '/uptime');
    const d = await r.json();
    uptimeEl.textContent = 'Up ' + d.uptime + 's';
  } catch {}
}, 1000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return HTML_PAGE.format(model=cfg["model"], width=cfg["width"], height=cfg["height"]), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot")
def snapshot():
    if _latest_jpeg is None:
        return "No frame yet", 503
    return Response(
        _latest_jpeg,
        mimetype="image/jpeg",
        headers={"Content-Disposition": "inline; filename=snapshot.jpg"},
    )


@app.route("/info")
def info():
    return jsonify({
        "service": "jetson-stream",
        "version": VERSION,
        "uptime": round(time.time() - _start_time, 1),
        "target_fps": target_fps,
        "resolution": f"{cfg.get('width', 0)}x{cfg.get('height', 0)}",
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "jetson-stream"})


@app.route("/ready")
def ready():
    if engine is not None:
        return jsonify({"status": "ok", "uptime": round(time.time() - _start_time, 1)})
    return jsonify({"status": "not_ready"}), 503


@app.route("/uptime")
def uptime():
    return jsonify({"uptime": round(time.time() - _start_time, 1)})


@app.route("/detections")
def detections():
    s = engine.get_status()
    return jsonify(s.get("detection_counts", {}))


@app.route("/status")
def status():
    data = engine.get_status()
    data["version"] = VERSION
    return jsonify(data)


@app.route("/confidence/<direction>")
def confidence(direction: str):
    thresh = engine.confidence_threshold
    if direction == "up":
        engine.set_confidence_threshold(thresh + 0.05)
    elif direction == "down":
        engine.set_confidence_threshold(thresh - 0.05)
    return jsonify({"confidence": engine.confidence_threshold})


@app.route("/fps/<int:fps_val>")
def set_fps(fps_val: int):
    global target_fps
    if fps_val in (15, 30, 60):
        target_fps = fps_val
    return jsonify({"target_fps": target_fps})


@app.route("/toggle/<detector>")
def toggle(detector: str):
    s = engine.get_status()
    if detector == "face":
        engine.toggle_face(not s["face_enabled"])
    elif detector == "motion":
        engine.toggle_motion(not s["motion_enabled"])
    elif detector == "human":
        engine.toggle_yolo(not s["yolo_enabled"])
    elif detector == "all":
        any_on = s["face_enabled"] or s["motion_enabled"] or s["yolo_enabled"]
        engine.toggle_face(not any_on)
        engine.toggle_motion(not any_on)
        engine.toggle_yolo(not any_on)
    else:
        return jsonify({"error": f"Unknown detector: {detector}"}), 400
    return jsonify(engine.get_status())


# ── Request Logging ──────────────────────────────────────────────────────────

import logging

request_logger = logging.getLogger("access")


@app.before_request
def log_request_start():
    request._start_time = time.time()
    request._request_id = request.headers.get("X-Request-Id", str(uuid.uuid4())[:8])

    if request.path.startswith("/video_feed"):
        return
    client_ip = request.remote_addr or "unknown"
    if not rate_limiter.is_allowed(client_ip):
        return jsonify({"error": "rate_limit", "message": "Too many requests"}), 429

    if AUTH_ENABLED and request.path not in ("/health",):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Basic "):
            return jsonify({"error": "unauthorized"}), 401
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
            user, pwd = decoded.split(":", 1)
            if user != AUTH_USER or pwd != AUTH_PASS:
                return jsonify({"error": "unauthorized"}), 401
        except Exception:
            return jsonify({"error": "unauthorized"}), 401


@app.after_request
def log_request(response):
    from flask import request
    if hasattr(request, "_start_time"):
        dur = (time.time() - request._start_time) * 1000
        request_logger.info("%s %s %s %.0fms", request.method, request.path, response.status_code, dur)
    return response


# ── CORS & Security ──────────────────────────────────────────────────────────

@app.after_request
def add_security_headers(response):
    rid = getattr(request, "_request_id", None)
    if rid:
        response.headers["X-Request-Id"] = rid
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Server-Timing"] = f"uptime;dur={time.time() - _start_time:.0f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Vary"] = "Authorization, Origin"
    if engine:
        counts = engine._detection_counts
        if counts:
            total = sum(counts.values())
            response.headers["X-Detection-Count"] = str(total)
    if request.path.startswith("/video_feed"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    else:
        response.headers["Cache-Control"] = "no-store"
    return response


# ── Error Handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "not_found", "message": str(e)}), 404


@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "bad_request", "message": str(e)}), 400


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "internal_error", "message": "Internal server error"}), 500


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global cfg, engine, target_fps
    cfg = interactive_setup()
    kill_port(5000)

    port = int(os.getenv("PORT", "5000"))

    engine = DetectionEngine(
        yolo_model=cfg["model"],
        yolo_model_type=YOLOModelType.ULTRALYTICS,
        yolo_device=os.getenv("YOLO_DEVICE", "cuda:0"),
        enable_yolo=False,
        enable_face=False,
        enable_motion=False,
        face_method=os.getenv("FACE_METHOD", "haar"),
        motion_method=os.getenv("MOTION_METHOD", "mog2"),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
        motion_min_area=int(os.getenv("MOTION_MIN_AREA", "500")),
    )

    w, h = cfg["width"], cfg["height"]
    print(f"\n  Stream:  http://<JETSON_IP>:{port}")
    print(f"  Model:   {cfg['model']}")
    print(f"  Target:  {target_fps} FPS @ {w}x{h}")
    print(f"  Keys:    F=face  M=motion  H=human  S=snapshot  +/-=conf  A=all  T=theme  1/2/3=fps")
    print(f"  Status:  http://<JETSON_IP>:{port}/status\n")

    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
