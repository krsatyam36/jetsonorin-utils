import json
import sys
sys.path.insert(0, "src")


def test_health_endpoint():
    from stream import app
    with app.test_client() as c:
        r = c.get("/health")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert data["status"] == "ok"


def test_ready_endpoint():
    from stream import app
    with app.test_client() as c:
        r = c.get("/ready")
        assert r.status_code in (200, 503)


def test_status_endpoint():
    from stream import app
    with app.test_client() as c:
        r = c.get("/status")
        assert r.status_code == 200


def test_uptime_endpoint():
    from stream import app
    with app.test_client() as c:
        r = c.get("/uptime")
        assert r.status_code == 200
        data = json.loads(r.data)
        assert "uptime" in data


def test_404_returns_json():
    from stream import app
    with app.test_client() as c:
        r = c.get("/nonexistent")
        assert r.status_code == 404
        data = json.loads(r.data)
        assert "error" in data


def test_security_headers():
    from stream import app
    with app.test_client() as c:
        r = c.get("/health")
        assert r.headers.get("X-Content-Type-Options") == "nosniff"
        assert r.headers.get("X-Frame-Options") == "DENY"
        assert "Server-Timing" in r.headers


def test_cors_headers():
    from stream import app
    with app.test_client() as c:
        r = c.get("/health")
        assert r.headers.get("Access-Control-Allow-Origin") == "*"
