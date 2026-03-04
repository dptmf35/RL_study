from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import sys
import time
import numpy as np
import os
import carb
from pathlib import Path
import omni.appwindow

from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.utils.extensions import enable_extension
enable_extension('isaacsim.ros2.bridge')

# Make dashboard importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from spot_policy import SpotArmFlatTerrainPolicy
from dashboard.backend.state_bridge import StateManager
from dashboard.backend.recorder import HDF5Recorder
from dashboard.backend.main import start_dashboard_server


class SpotRunner(object):
    def __init__(self, physics_dt, render_dt) -> None:
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        # Spawn warehouse
        prim = define_prim("/World/Warehouse", "Xform")
        asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
        prim.GetReferences().AddReference(asset_path)

        policy_path = os.path.join(BASE_DIR, "policies/spot_arm/models", "spot_arm_policy.pt")
        policy_params_path = os.path.join(BASE_DIR, "policies/spot_arm/params", "env.yaml")
        usd_path = os.path.join(BASE_DIR, "assets", "spot_arm.usd")

        self._spot = SpotArmFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            usd_path=usd_path,
            policy_path=policy_path,
            policy_params_path=policy_params_path,
            position=np.array([1, 0, 0.8]),
        )

        self._base_command = np.zeros(3)
        self._input_keyboard_mapping = {
            "NUMPAD_8": [1.0, 0.0, 0.0], "UP":    [1.0, 0.0, 0.0],
            "NUMPAD_2": [-1.0, 0.0, 0.0], "DOWN": [-1.0, 0.0, 0.0],
            "NUMPAD_6": [0.0, -1.0, 0.0], "RIGHT": [0.0, -1.0, 0.0],
            "NUMPAD_4": [0.0, 1.0, 0.0],  "LEFT":  [0.0, 1.0, 0.0],
            "NUMPAD_7": [0.0, 0.0, 1.0],  "N": [0.0, 0.0, 1.0],
            "NUMPAD_9": [0.0, 0.0, -1.0], "M": [0.0, 0.0, -1.0],
        }

        self.needs_reset = False
        self.first_step = True

        # Dashboard infrastructure
        self._state_manager = StateManager()
        self._recorder = HDF5Recorder(save_dir=str(BASE_DIR / "data" / "recordings"))

    def setup(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._sub_keyboard_event
        )
        self._world.add_physics_callback("spot_forward", callback_fn=self.on_physics_step)

        # Start dashboard server in background
        start_dashboard_server(self._state_manager, self._recorder, port=8000, enable_camera=False)

    def on_physics_step(self, step_size) -> None:
        if self.first_step:
            self._spot.initialize()
            self.first_step = False
        elif self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
        else:
            self._spot.forward(step_size, self._base_command)
            # Push state to dashboard at policy frequency (every decimation steps)
            if self._spot._policy_counter % self._spot._decimation == 0:
                self._push_state()

    def _push_state(self) -> None:
        try:
            pos, quat = self._spot.robot.get_world_pose()
            joint_pos = self._spot.robot.get_joint_positions()
            joint_vel = self._spot.robot.get_joint_velocities()
            obs = self._spot._compute_observation(self._base_command)
            state = {
                "timestamp": time.time(),
                "obs": obs.tolist(),
                "action": self._spot.action.tolist(),
                "command": self._base_command.tolist(),
                "pose": np.concatenate([pos, quat]).tolist(),
                "joint_pos": joint_pos.tolist(),
                "joint_vel": joint_vel.tolist(),
            }
            self._state_manager.push_state(state)
        except Exception as e:
            carb.log_warn(f"[Dashboard] State push failed: {e}")

    def run(self) -> None:
        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True

    def shutdown(self) -> None:
        self._recorder.close()

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command += np.array(self._input_keyboard_mapping[event.input.name])
                print(f"[CMD] key={event.input.name:10s}  vx={self._base_command[0]:+.1f}  vy={self._base_command[1]:+.1f}  wz={self._base_command[2]:+.1f}")
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._input_keyboard_mapping:
                self._base_command -= np.array(self._input_keyboard_mapping[event.input.name])
                print(f"[CMD] release={event.input.name:10s} vx={self._base_command[0]:+.1f}  vy={self._base_command[1]:+.1f}  wz={self._base_command[2]:+.1f}")
        return True


def main():
    physics_dt = 1 / 200.0
    render_dt = 1 / 60.0

    runner = SpotRunner(physics_dt=physics_dt, render_dt=render_dt)
    simulation_app.update()
    runner._world.reset()
    simulation_app.update()
    runner.setup()
    simulation_app.update()

    print("[Teleop] Controls: Arrow keys / Numpad = fwd/back/left/right | N/M = rotate")
    runner.run()
    runner.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
