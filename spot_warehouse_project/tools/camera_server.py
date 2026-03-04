#!/usr/bin/env python3
"""
Standalone MJPEG camera server for the Spot dashboard.

Run in a terminal with ROS2 sourced (env_isaaclab):
    python3 /home/yeseul/IsaacRobotics/tools/camera_server.py

Requires: rclpy, numpy, Pillow
    pip install Pillow numpy
"""

import io
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

TOPICS = {
    "/scar/camera/frontleft/image": "frontleft",
    "/scar/camera/frontright/image": "frontright",
}
PORT = 8001

_frames: dict[str, bytes] = {}
_lock = threading.Lock()


# ─── Image conversion ────────────────────────────────────────────────────────

def ros_image_to_jpeg(msg) -> bytes | None:
    try:
        enc = msg.encoding.lower()
        h, w = msg.height, msg.width
        raw = np.frombuffer(msg.data, dtype=np.uint8)

        if enc == "rgb8":
            arr = raw.reshape(h, w, 3)
        elif enc == "bgr8":
            arr = raw.reshape(h, w, 3)[:, :, ::-1]
        elif enc == "mono8":
            arr = raw.reshape(h, w)
        elif enc == "rgba8":
            arr = raw.reshape(h, w, 4)[:, :, :3]
        elif enc == "bgra8":
            arr = raw.reshape(h, w, 4)[:, :, 2::-1]
        else:
            arr = raw.reshape(h, w, -1)

        from PIL import Image as PILImage
        pil = PILImage.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=80)
        return buf.getvalue()
    except Exception as e:
        print(f"[Camera] encode error ({msg.encoding}): {e}")
        return None


def make_callback(key: str):
    def callback(msg):
        jpeg = ros_image_to_jpeg(msg)
        if jpeg:
            with _lock:
                _frames[key] = jpeg
    return callback


# ─── ROS2 subscriber thread ───────────────────────────────────────────────────

def spin_ros():
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import Image

    rclpy.init()
    node = Node("dashboard_camera_node")

    for topic, key in TOPICS.items():
        node.create_subscription(Image, topic, make_callback(key), 10)
        print(f"[Camera] Subscribed to {topic}")

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    print("[Camera] Spinning...")
    executor.spin()


# ─── MJPEG HTTP server ────────────────────────────────────────────────────────

class CameraHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # silence access logs

    def do_GET(self):
        # Serve GET /api/camera/{key}
        parts = self.path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "api" and parts[1] == "camera":
            key = parts[2]
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        frame = _frames.get(key)
                    if frame:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + frame
                            + b"\r\n"
                        )
                        self.wfile.flush()
                    time.sleep(0.033)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    ros_thread = threading.Thread(target=spin_ros, daemon=True)
    ros_thread.start()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), CameraHandler)
    print(f"[CameraServer] MJPEG stream at http://localhost:{PORT}/api/camera/{{frontleft|frontright}}")
    server.serve_forever()
