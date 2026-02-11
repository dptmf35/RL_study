# AGENTS Guide for RL_study

This repository contains MuJoCo-based reinforcement learning environments and training scripts for the UR5e arm and Robotiq gripper. Treat this guide as the operational handbook for agentic contributors.

## Quick Facts
- Primary language: Python 3.12 (virtualenv stored under `rl-study/`).
- Core libraries: Gymnasium, Stable-Baselines3 (PPO/SAC), PyTorch, MuJoCo 3.x.
- Entry points live in `scripts/`; custom Gymnasium environments live in `envs/`.
- Simulation assets (XML, meshes) live in `assets/`; keep them under version control unless explicitly ignored.
- Generated artefacts (`logs/`, `models/`, `videos/`, `tensorboard_logs/`) are gitignored—do not commit outputs.

## Repository Layout (edit in these areas; leave vendored dirs alone)
- `envs/` – Gymnasium environments (`UR5ePickPlaceEnv`, `UR5eRobotiqPickPlaceEnv`, `UR5eRobotiqGoalEnv`).
- `scripts/` – CLI tools for training, evaluation, video export, physics sanity checks, teleoperation.
- `configs/default.yaml` – Canonical hyperparameter set shared by PPO scripts.
- `assets/` – MuJoCo XML scenes and meshes; modifying these affects simulation geometry.
- `requirements.txt` – Python dependency constraints for lean installs.
- `README.md`, `README_ROBOTIQ.md`, `QUICKSTART_ROBOTIQ.md` – Human-facing walkthroughs; sync these with code if behavior changes.
- `.gitignore` – Avoid re-adding generated models/logs; respect the exclusions when creating new artefacts.
- `Gymnasium-Robotics/`, `Pybullet_demo/`, `rl-study/`, `mujoco_menagerie/`, `mujoco/`, `ur5_reinforcement_learning_grasp_object/` – External assets or venv snapshots; do **not** edit.

## Environment Setup (first run)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use .\venv\Scripts\activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```
- The project currently uses torch>=2.0; GPU builds require matching CUDA wheels.
- MuJoCo 3.x is installed via pip (`mujoco>=3.0.0`), but you must supply the native binaries if running locally.
- For headless servers export rendering backend: `export MUJOCO_GL=egl` before running scripts.
- TensorBoard is optional but recommended: `pip install tensorboard` (already in requirements).

## Training & Simulation Commands
- Baseline PPO (UR5e gripper):
  ```bash
  python scripts/train_ppo.py --total-timesteps 2000000 --n-envs 8
  ```
- Robotiq PPO variant (shorter horizon defaults baked in):
  ```bash
  python scripts/train_ppo_robotiq.py --task-mode pick_place --n-envs 8
  ```
- SAC + HER Robotiq experiment:
  ```bash
  python scripts/train_sac_her_robotiq.py --task pick
  ```
- Monitor TensorBoard while training: `tensorboard --logdir logs/` (uses VecNormalize stats under `models/<run>/`).
- Video capture for trained policies:
  ```bash
  python scripts/record_video.py --model-path models/<run>/best/best_model.zip --n-episodes 3
  ```
- Teleoperation sandbox: `python scripts/teleop_ur5e_robotiq.py --render` (requires display or EGL streaming).

## Evaluation & Diagnostics
- Evaluate deterministic policy with rendering:
  ```bash
  python scripts/evaluate.py --model-path models/<run>/final_model.zip --n-episodes 10 --render
  ```
- Run random policy smoke test (no model needed):
  ```bash
  python scripts/evaluate.py --random --n-episodes 3 --slow-motion
  ```
- Quick physics sanity check for Robotiq environment:
  ```bash
  python scripts/test_env_robotiq.py --task reach --render
  ```
- Heuristic pick-phase regression: `python scripts/test_pick_task.py` (opens viewer, waits for ESC).
- Low-level MuJoCo validation: `python test_robotiq_env.py` (viewer loop, prints joint metadata).
- Legacy environment regression: `python test_improved_env.py` (prints reward components while stepping randomly).
- Workspace debugging utilities: `python scripts/check_workspace.py` to inspect reachability limits.

## Running a Single Targeted Test
- There is no pytest suite; smoke tests are standalone scripts.
- To exercise one behavior, run the relevant script directly (for example, `python scripts/test_cube_physics.py` to stress contacts).
- When adding new tests prefer self-contained CLI scripts that print PASS/FAIL like existing examples.
- Wire new scripts into documentation after verifying they exit cleanly (no hanging viewers, handle `KeyboardInterrupt`).

## Runtime Tips & Environment Variables
- Always activate the virtualenv before invoking scripts to avoid pulling global MuJoCo builds.
- Set `MUJOCO_GL=egl` on headless machines; use `MUJOCO_GL=osmesa` only if EGL unavailable.
- For reproducibility pass `--seed` and keep `np.random.default_rng` seeds consistent when sampling.
- GPU selection: set `CUDA_VISIBLE_DEVICES` externally rather than editing scripts.
- Remote rendering through VNC requires `--render` plus EGL viewport forwarding.

## Data & Checkpoint Management
- Checkpoints save under `models/<run_name>/`; each run stores `best/` and `final_model/` plus `vec_normalize.pkl`.
- Logs for TensorBoard live under `logs/<run_name>/`; prune before committing to avoid large diffs.
- Videos default to `videos/`; adjust via `--output-dir` on `record_video.py`.
- Keep large binaries (trained zips, MP4s) out of git; share via artifact storage if needed.
- When debugging, use dedicated run names to avoid overwriting baseline experiments.

## Code Style & Conventions
- Follow PEP 8 spacing and 120-character soft limit; existing files occasionally exceed 100 chars but stay readable.
- Imports are grouped as: standard library, third-party, then local modules; each block separated by a blank line.
- Use `pathlib.Path` for filesystem work (see training scripts) and avoid hardcoding absolute paths.
- Type hints are encouraged for function signatures (`Tuple[np.ndarray, Dict[str, Any]]` patterns in `envs/`).
- Public modules/classes start with descriptive docstrings; keep them updated when behavior changes.
- Prefer explicit enums/strings for mode switches (`task_mode` values: `reach|pick|pick_place`).
- Keep CLI entry points under `if __name__ == "__main__":` with `argparse` for discoverability.
- Use f-strings for user-facing logs and progress updates; avoid `.format()` unless needed for localization.
- Logging: rely on `print()` plus TensorBoard scalars (`TensorboardCallback`)—do not introduce heavy logging frameworks.
- Error handling: wrap risky IO in `try/except` and re-raise with context; tests use assertions for invariants.
- Environments must return `(obs, info)` from `reset()` and `(obs, reward, terminated, truncated, info)` from `step()` to stay Gymnasium-compliant.
- Maintain deterministic randomization by reading from `self.np_random` within Gymnasium envs.
- Keep MuJoCo handles (`self.model`, `self.data`) as instance attributes; do not recreate them inside `step()` loops.
- Reward shaping uses staged dictionaries; extend `_compute_reward` by adding new keys rather than rewriting logic wholesale.
- Scaling constants (`joint_action_scale`, thresholds) should live on the class and be documented inline.
- Use numpy vector math; avoid Python loops when computing distances or concatenating observations.
- Device management: pass `--device` flag through CLI; prefer `"auto"` so SB3 selects CUDA when available.
- When adding callbacks inherit from `BaseCallback` and use `self.logger.record` to log scalars.
- For new VecNormalize checkpoints, mirror existing save/load patterns (`env.save`, `VecNormalize.load`).
- Keep heuristics and debug utilities in `scripts/` instead of embedding them inside env classes.
- Do not introduce global state; pass configuration via function arguments or YAML.
- Respect `.gitignore`; never commit generated MuJoCo logs, large meshes, or virtualenv directories.

## Patterns to Mirror in New Code
- Training scripts insert project root into `sys.path`—repeat this pattern for new CLIs needing local imports.
- Wrap environments with `Monitor` and `VecNormalize` when using vectorized environments.
- For evaluation, attempt to load `vec_normalize.pkl` from sibling directories before warning users (see `evaluate.py`).
- CLI defaults favor dense rewards and long horizons; expose overrides via flags instead of editing constants.
- Reward debugging prints live under `info['reward_components']`; reuse this structure for new diagnostics.
- When modifying MuJoCo XML make matching updates to spawn ranges in Python to prevent unreachable goals.
- Keep teleop/heuristic scripts interactive but bounded—ensure they catch exceptions and close viewers gracefully.
- Use `np.concatenate([...]).astype(np.float32)` for observation assembly to maintain dtype consistency.
- For new assets, place them in `assets/` and update `.gitignore` only if they should not be versioned.
- Document new capabilities in both README and this guide to keep humans and agents aligned.

## External Tooling & IDE Rules
- No Cursor or Copilot rules are present; follow this document alongside README guidance.
- Pre-commit hooks are not configured—run formatters manually if you introduce them.
- If you add linters (ruff, pylint), document invocation commands here before relying on them in CI.

## Contribution Workflow Tips
- Prefer incremental PRs—training scripts are long; isolate changes to targeted sections.
- Verify MuJoCo loads locally before posting PRs (headless runs can fail silently without EGL).
- Capture screenshots/videos externally rather than embedding binaries into the repo.
- When sharing checkpoints use relative paths in docs (`models/<run>/best/best_model.zip`).
- Coordinate large asset updates (meshes/XML) with maintainers; these changes affect physics stability.

## Simulation Gotchas
- Ensure `assets/ur5e_pick_place.xml` and Python spawn ranges stay in sync; mismatches yield unreachable targets.
- Reset functions expect cube quaternions `[1, 0, 0, 0]`; keep quaternion normalized when introducing rotations.
- Gripper openness thresholds (`0.002` open, `0.010` closed) drive reward logic—adjust both code and comments together.
- Reward dictionaries power TensorBoard diagnostics; add new keys sparingly and document them in `info['reward_components']`.
- Viewer loops block execution; wrap demos in `try/finally` to close MuJoCo viewers on exceptions.
- VecNormalize stats live beside models; copying a checkpoint without the `vec_normalize.pkl` causes performance drops.
- High parallel env counts need matching `--n-envs` when loading checkpoints to keep saving cadence consistent.
- Large video renders require adequate disk space; prune `videos/` after exporting artifacts.
- Headless EGL sessions need `sudo apt-get install libglfw3 libglfw3-dev` on fresh Ubuntu images.

Stay curious, keep the robot upright, and update this guide whenever workflows shift.
