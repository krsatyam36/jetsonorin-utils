import cv2
from flask import Flask, Response

app = Flask(__name__)

def generate_frames():
    # Open the camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # Set to MJPG for high performance
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
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
        else:
            # Encode the frame as a JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in the format required for HTTP MJPEG streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    cap.release()

@app.route('/')
def index():
    # A simple HTML page to display the video
    return '''
    <html>
        <head>
            <title>Jetson Drone Cam</title>
            <style>
                body { background-color: #222; color: white; text-align: center; font-family: sans-serif; }
                img { max-width: 100%; height: auto; border: 3px solid #444; border-radius: 8px; }
            </style>
        </head>
        <body>
            <h1>Arducam Live Feed</h1>
            <img src="/video_feed">
        </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    # This route provides the actual continuous video stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # host='0.0.0.0' exposes the server to the entire local network
    print("Starting server... Access it in your browser!")
    app.run(host='0.0.0.0', port=5000, debug=False)
