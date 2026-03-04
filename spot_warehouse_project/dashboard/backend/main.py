import asyncio
import io
import threading
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .state_bridge import StateManager
from .recorder import HDF5Recorder
from .camera_bridge import CameraBridge

app = FastAPI(title="Spot Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Populated by start_dashboard_server()
_state_manager: Optional[StateManager] = None
_recorder: Optional[HDF5Recorder] = None
_camera: Optional[CameraBridge] = None
_recording_task_running = False


# ─────────────────────────── REST endpoints ────────────────────────────

@app.get("/api/status")
def get_status():
    return {
        "recording": _recorder.is_recording if _recorder else False,
        "episode_count": _recorder.episode_count if _recorder else 0,
        "recording_duration": _recorder.get_recording_duration() if _recorder else 0.0,
        "file_size_bytes": _recorder.get_file_size() if _recorder else 0,
        "connected_to_sim": _state_manager is not None and _state_manager.get_latest_state() is not None,
    }


@app.post("/api/record/start")
def record_start():
    if _recorder is None:
        return JSONResponse({"error": "recorder not ready"}, status_code=503)
    if _recorder.is_recording:
        return JSONResponse({"error": "already recording"}, status_code=400)
    ep_idx = _recorder.start_episode()
    return {"episode": ep_idx, "status": "recording"}


@app.post("/api/record/stop")
def record_stop():
    if _recorder is None:
        return JSONResponse({"error": "recorder not ready"}, status_code=503)
    if not _recorder.is_recording:
        return JSONResponse({"error": "not recording"}, status_code=400)
    meta = _recorder.stop_episode()
    return {"status": "stopped", "episode": meta}


@app.get("/api/episodes")
def get_episodes():
    if _recorder is None:
        return []
    return _recorder.get_episodes()


# ─────────────────────────── WebSocket ────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            state = _state_manager.get_latest_state() if _state_manager else None
            if state:
                # Send a slimmed-down version for display (drop full obs to save bandwidth)
                payload = {
                    "timestamp": state["timestamp"],
                    "command": state["command"],
                    "pose": state["pose"],
                    "joint_pos": state["joint_pos"],
                    "joint_vel": state["joint_vel"],
                    "recording": _recorder.is_recording if _recorder else False,
                    "episode_count": _recorder.episode_count if _recorder else 0,
                    "recording_duration": _recorder.get_recording_duration() if _recorder else 0.0,
                }
                await websocket.send_json(payload)
            await asyncio.sleep(0.1)  # 10 Hz
    except (WebSocketDisconnect, Exception):
        pass


# ─────────────────────────── MJPEG camera streams ─────────────────────

async def _mjpeg_generator(key: str):
    """Async generator: yields multipart JPEG frames at ~30 Hz."""
    while True:
        frame = _camera.get_frame(key) if _camera else None
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        await asyncio.sleep(0.033)   # ~30 Hz


@app.get("/api/camera/{key}")
async def camera_stream(key: str):
    if _camera is None:
        return JSONResponse({"error": "camera bridge not started"}, status_code=503)
    return StreamingResponse(
        _mjpeg_generator(key),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/camera/{key}/snapshot")
async def camera_snapshot(key: str):
    """Single JPEG frame (for polling fallback)."""
    if _camera is None:
        return JSONResponse({"error": "camera bridge not started"}, status_code=503)
    frame = _camera.get_frame(key)
    if frame is None:
        return JSONResponse({"error": f"no frame yet for '{key}'"}, status_code=404)
    return StreamingResponse(io.BytesIO(frame), media_type="image/jpeg")


# ─────────────────────────── Background recorder task ─────────────────

def _recorder_drain_loop(state_manager: StateManager, recorder: HDF5Recorder):
    """Drains the state queue and feeds steps to recorder. Runs in FastAPI thread."""
    while True:
        states = state_manager.drain_queue()
        if recorder.is_recording:
            for s in states:
                recorder.add_step(s)
        time.sleep(0.05)  # 20 Hz drain rate


# ─────────────────────────── Server launcher ───────────────────────────

def start_dashboard_server(
    state_manager: StateManager,
    recorder: HDF5Recorder,
    port: int = 8000,
    enable_camera: bool = True,
) -> None:
    """Launch FastAPI server in a daemon background thread."""
    global _state_manager, _recorder, _camera
    _state_manager = state_manager
    _recorder = recorder

    # Start HDF5 recorder session
    path = recorder.start_session()
    print(f"[Dashboard] HDF5 session: {path}")

    # Start ROS2 camera bridge
    if enable_camera:
        _camera = CameraBridge()
        _camera.start()
        print(f"[Dashboard] Camera bridge started")

    # Start background drain loop
    drain_thread = threading.Thread(
        target=_recorder_drain_loop,
        args=(state_manager, recorder),
        daemon=True,
    )
    drain_thread.start()

    # Start uvicorn in daemon thread
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    server_thread = threading.Thread(
        target=server.run,
        daemon=True,
    )
    server_thread.start()

    print(f"[Dashboard] API server running at http://localhost:{port}")
    print(f"[Dashboard] Open http://localhost:5173 (dev) or http://localhost:{port} in browser")
