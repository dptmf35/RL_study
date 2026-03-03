import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


class HDF5Recorder:
    """Episode-based HDF5 recorder for imitation learning data collection."""

    def __init__(self, save_dir: str = "data/recordings"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._file: Optional[h5py.File] = None
        self._file_path: Optional[Path] = None
        self._episode_count: int = 0
        self._recording: bool = False
        self._episode_start_time: Optional[float] = None
        self._episodes_meta: list = []

        # Per-step buffer (accumulated during recording, flushed on stop)
        self._buffer: dict = {}

    def start_session(self) -> str:
        """Create a new HDF5 file for this session."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._file_path = self.save_dir / f"session_{ts}.h5"
        self._file = h5py.File(self._file_path, "w")
        self._file.attrs["created_at"] = ts
        self._file.attrs["robot_type"] = "SpotArm"
        self._file.attrs["obs_dim"] = 69
        self._file.attrs["action_dim"] = 19
        self._episode_count = 0
        self._episodes_meta = []
        return str(self._file_path)

    def start_episode(self) -> int:
        """Begin buffering a new episode. Returns episode index."""
        if self._file is None:
            self.start_session()
        self._recording = True
        self._episode_start_time = time.time()
        self._buffer = {
            "timestamps": [],
            "obs": [],
            "actions": [],
            "commands": [],
            "poses": [],
        }
        return self._episode_count

    def add_step(self, state: dict) -> None:
        """Buffer a single timestep. Only valid while recording."""
        if not self._recording:
            return
        self._buffer["timestamps"].append(state["timestamp"])
        self._buffer["obs"].append(state["obs"])
        self._buffer["actions"].append(state["action"])
        self._buffer["commands"].append(state["command"])
        self._buffer["poses"].append(state["pose"])

    def stop_episode(self) -> dict:
        """Flush buffer to HDF5 and return episode metadata."""
        if not self._recording or self._file is None:
            return {}

        self._recording = False
        end_time = time.time()
        duration = end_time - self._episode_start_time

        n_steps = len(self._buffer["timestamps"])
        ep_name = f"episode_{self._episode_count:04d}"

        if n_steps > 0:
            grp = self._file.create_group(ep_name)
            grp.create_dataset("timestamps", data=np.array(self._buffer["timestamps"], dtype=np.float64))
            grp.create_dataset("obs",        data=np.array(self._buffer["obs"],        dtype=np.float32))
            grp.create_dataset("actions",    data=np.array(self._buffer["actions"],    dtype=np.float32))
            grp.create_dataset("commands",   data=np.array(self._buffer["commands"],   dtype=np.float32))
            grp.create_dataset("poses",      data=np.array(self._buffer["poses"],      dtype=np.float32))
            grp.attrs["start_time"] = self._episode_start_time
            grp.attrs["end_time"] = end_time
            grp.attrs["duration_sec"] = duration
            grp.attrs["n_steps"] = n_steps
            self._file.flush()

        meta = {
            "id": self._episode_count,
            "name": ep_name,
            "n_steps": n_steps,
            "duration_sec": round(duration, 2),
            "start_time": datetime.fromtimestamp(self._episode_start_time).strftime("%H:%M:%S"),
        }
        self._episodes_meta.append(meta)
        self._episode_count += 1
        self._buffer = {}
        return meta

    def close(self) -> None:
        """Safely close the HDF5 file (flushes in-progress episode first)."""
        if self._recording:
            self.stop_episode()
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def episode_count(self) -> int:
        return self._episode_count

    def get_episodes(self) -> list:
        return list(self._episodes_meta)

    def get_file_size(self) -> int:
        """Return HDF5 file size in bytes, or 0 if no file."""
        if self._file_path and self._file_path.exists():
            return self._file_path.stat().st_size
        return 0

    def get_recording_duration(self) -> float:
        """Return elapsed seconds for current episode, or 0."""
        if self._recording and self._episode_start_time:
            return time.time() - self._episode_start_time
        return 0.0
