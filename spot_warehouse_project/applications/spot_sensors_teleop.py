from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import os
from pathlib import Path

import omni.appwindow

from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.utils.extensions import enable_extension
enable_extension('isaacsim.ros2.bridge')

from spot_policy import SpotFlatTerrainPolicy


BASE_DIR = Path(__file__).resolve().parent.parent


class SpotSensorsTeleopRunner:
    def __init__(self, physics_dt, render_dt):
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        # Spawn warehouse
        prim = define_prim("/World/Warehouse", "Xform")
        asset_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
        prim.GetReferences().AddReference(asset_path)

        policy_path = os.path.join(BASE_DIR, "policies/spot_flat/models", "policy.pt")
        params_path = os.path.join(BASE_DIR, "policies/spot_flat/params", "env.yaml")
        usd_path    = os.path.join(BASE_DIR, "assets", "spot_sensors.usd")

        self._spot = SpotFlatTerrainPolicy(
            prim_path="/World/Spot",
            name="Spot",
            usd_path=usd_path,
            policy_path=policy_path,
            policy_params_path=params_path,
            position=np.array([1, 0, 0.8]),
        )

        self._base_command = np.zeros(3)
        self._speed = 1.0          # current speed multiplier (m/s)
        self._speed_step = 0.5
        self._speed_min  = 0.5
        self._speed_max  = 3.0
        self._held_keys  = set()   # currently pressed direction keys

        # Unit direction vectors — scaled by self._speed at runtime
        self._dir_mapping = {
            "NUMPAD_8": np.array([ 1.0,  0.0,  0.0]), "UP":    np.array([ 1.0,  0.0,  0.0]),
            "NUMPAD_2": np.array([-1.0,  0.0,  0.0]), "DOWN":  np.array([-1.0,  0.0,  0.0]),
            "NUMPAD_6": np.array([ 0.0, -1.0,  0.0]), "RIGHT": np.array([ 0.0, -1.0,  0.0]),
            "NUMPAD_4": np.array([ 0.0,  1.0,  0.0]), "LEFT":  np.array([ 0.0,  1.0,  0.0]),
            "NUMPAD_7": np.array([ 0.0,  0.0,  1.0]), "N":     np.array([ 0.0,  0.0,  1.0]),
            "NUMPAD_9": np.array([ 0.0,  0.0, -1.0]), "M":     np.array([ 0.0,  0.0, -1.0]),
        }
        self.needs_reset = False
        self.first_step = True

    def setup(self) -> None:
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._sub_keyboard_event
        )
        self._world.add_physics_callback("spot_forward", callback_fn=self._on_physics_step)

    def _on_physics_step(self, step_size) -> None:
        if self.first_step:
            self._spot.initialize()
            self.first_step = False
        elif self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True
        else:
            self._spot.forward(step_size, self._base_command)

    def _recompute_command(self):
        cmd = np.zeros(3)
        for key in self._held_keys:
            cmd += self._dir_mapping[key] * self._speed
        # Clamp to policy training range
        cmd[0] = np.clip(cmd[0], -2.0, 3.0)   # vx
        cmd[1] = np.clip(cmd[1], -1.5, 1.5)   # vy
        cmd[2] = np.clip(cmd[2], -2.0, 2.0)   # wz
        self._base_command = cmd

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:
        key = event.input.name if hasattr(event.input, 'name') else event.input
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key in self._dir_mapping:
                self._held_keys.add(key)
                self._recompute_command()
                print(f"[CMD] vx={self._base_command[0]:+.1f}  vy={self._base_command[1]:+.1f}  wz={self._base_command[2]:+.1f}  speed={self._speed:.1f}")
            elif key in ("EQUAL", "NUMPAD_ADD"):
                self._speed = min(self._speed + self._speed_step, self._speed_max)
                self._recompute_command()
                print(f"[Speed] {self._speed:.1f} m/s")
            elif key in ("MINUS", "NUMPAD_SUBTRACT"):
                self._speed = max(self._speed - self._speed_step, self._speed_min)
                self._recompute_command()
                print(f"[Speed] {self._speed:.1f} m/s")
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key in self._dir_mapping:
                self._held_keys.discard(key)
                self._recompute_command()
        return True

    def run(self) -> None:
        while simulation_app.is_running():
            self._world.step(render=True)
            if self._world.is_stopped():
                self.needs_reset = True


def main():
    physics_dt = 1 / 200.0
    render_dt  = 1 / 60.0

    runner = SpotSensorsTeleopRunner(physics_dt=physics_dt, render_dt=render_dt)
    simulation_app.update()
    runner._world.reset()
    simulation_app.update()
    runner.setup()
    simulation_app.update()

    print("[Teleop] Controls: Arrow keys / Numpad = fwd/back/left/right | N/M = rotate CW/CCW")
    print("[Teleop] Speed:    = / NUMPAD+ = speed up  |  - / NUMPAD- = speed down  (0.5 ~ 3.0 m/s, default 1.0)")
    print("[Teleop] Click the Isaac Sim viewport first to give it keyboard focus.")
    runner.run()
    simulation_app.close()


if __name__ == "__main__":
    main()
