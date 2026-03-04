"""
ROS2 camera subscriber → JPEG frame cache.
Subscribes to sensor_msgs/Image topics and keeps the latest JPEG frame per topic.
"""

import io
import threading
import numpy as np

TOPICS = [
    "/scar/camera/frontleft/image",
    "/scar/camera/frontright/image",
]

# Friendly short names for URL routing
TOPIC_KEYS = {
    "/scar/camera/frontleft/image":  "frontleft",
    "/scar/camera/frontright/image": "frontright",
}


class CameraBridge:
    def __init__(self):
        self._frames: dict[str, bytes] = {}   # key → latest JPEG bytes
        self._lock = threading.Lock()
        self._node = None

    # ─────────────────────────── image conversion ──────────────────────

    def _ros_image_to_jpeg(self, msg) -> bytes | None:
        """Convert sensor_msgs/Image to JPEG bytes without cv_bridge."""
        try:
            enc = msg.encoding.lower()
            h, w = msg.height, msg.width
            raw = np.frombuffer(msg.data, dtype=np.uint8)

            if enc in ("rgb8",):
                arr = raw.reshape(h, w, 3)
            elif enc in ("bgr8",):
                arr = raw.reshape(h, w, 3)[:, :, ::-1]   # BGR → RGB
            elif enc in ("mono8",):
                arr = raw.reshape(h, w)
            elif enc in ("rgba8",):
                arr = raw.reshape(h, w, 4)[:, :, :3]     # drop alpha
            elif enc in ("bgra8",):
                arr = raw.reshape(h, w, 4)[:, :, 2::-1]  # BGRA → RGB
            else:
                # Fallback: try to treat as RGB
                arr = raw.reshape(h, w, -1)

            from PIL import Image as PILImage
            pil = PILImage.fromarray(arr)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=80)
            return buf.getvalue()
        except Exception as e:
            print(f"[Camera] encode error ({msg.encoding}): {e}")
            return None

    # ─────────────────────────── rclpy node ───────────────────────────

    def _make_callback(self, key: str):
        def callback(msg):
            jpeg = self._ros_image_to_jpeg(msg)
            if jpeg:
                with self._lock:
                    self._frames[key] = jpeg
        return callback

    def start(self):
        """Initialize rclpy and spin in a background daemon thread."""
        def _spin():
            import rclpy
            from rclpy.node import Node
            from rclpy.executors import SingleThreadedExecutor

            try:
                if not rclpy.ok():
                    rclpy.init()
            except Exception:
                try:
                    rclpy.init()
                except Exception as e:
                    print(f"[Camera] rclpy.init() failed: {e}")
                    return

            try:
                node = Node("dashboard_camera_node")
                self._node = node

                from sensor_msgs.msg import Image
                for topic, key in TOPIC_KEYS.items():
                    node.create_subscription(
                        Image, topic, self._make_callback(key), 10
                    )
                    print(f"[Camera] Subscribed to {topic}")

                executor = SingleThreadedExecutor()
                executor.add_node(node)
                print("[Camera] Executor spinning...")
                executor.spin()
            except Exception as e:
                import traceback
                print(f"[Camera] spin error: {e}")
                traceback.print_exc()

        t = threading.Thread(target=_spin, daemon=True)
        t.start()

    # ─────────────────────────── public API ───────────────────────────

    def get_frame(self, key: str) -> bytes | None:
        with self._lock:
            return self._frames.get(key)

    def available_keys(self) -> list[str]:
        with self._lock:
            return list(self._frames.keys())
