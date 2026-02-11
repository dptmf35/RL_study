"""
Evaluation and Visualization Script for Go2 Locomotion Policy.

Loads a trained policy and visualizes it in MuJoCo's interactive viewer.
Supports both flat and terrain environments.

Usage:
    python eval.py --model checkpoints/<run>/best_model.zip
    python eval.py --model checkpoints/<run>/best_model.zip --terrain
    python eval.py --model checkpoints/<run>/best_model.zip --no-viewer
    python eval.py --model checkpoints/<run>/best_model.zip --record video.mp4
"""

import os
import sys
import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(__file__))
from envs.go2_env import Go2Env
from envs.go2_terrain_env import Go2TerrainEnv


def load_model_and_vecnorm(model_path):
    """Load trained PPO model and VecNormalize stats."""
    model = PPO.load(model_path, device="cpu")

    vecnorm_path = model_path.replace(".zip", "_vecnormalize.pkl")
    if not os.path.exists(vecnorm_path):
        model_dir = os.path.dirname(model_path)
        for f in os.listdir(model_dir):
            if f.endswith("_vecnormalize.pkl"):
                vecnorm_path = os.path.join(model_dir, f)
                break

    return model, vecnorm_path


def make_env(use_terrain=False, difficulty=0.5, **kwargs):
    """Create environment based on mode."""
    if use_terrain:
        return Go2TerrainEnv(difficulty=difficulty, **kwargs)
    return Go2Env(**kwargs)


def evaluate_headless(model_path, n_episodes=10, use_terrain=False, difficulty=0.5):
    """Run evaluation without visualization, print statistics."""
    model, vecnorm_path = load_model_and_vecnorm(model_path)

    env = DummyVecEnv([lambda: make_env(use_terrain, difficulty)])
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize from: {vecnorm_path}")

    episode_rewards = []
    episode_lengths = []
    forward_distances = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        start_pos = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1

            if start_pos is None and "base_pos" in info[0]:
                start_pos = info[0]["base_pos"].copy()

        final_pos = info[0].get("base_pos", np.zeros(3))
        fwd_dist = final_pos[0] - (start_pos[0] if start_pos is not None else 0)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        forward_distances.append(fwd_dist)
        print(f"  Episode {ep+1}/{n_episodes}: reward={total_reward:.2f}, "
              f"steps={steps}, forward={fwd_dist:.2f}m")

    env.close()

    print(f"\n=== Evaluation Results ({n_episodes} episodes) ===")
    print(f"  Mode         : {'terrain' if use_terrain else 'flat'}")
    print(f"  Mean reward  : {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"  Mean length  : {np.mean(episode_lengths):.1f} +/- {np.std(episode_lengths):.1f}")
    print(f"  Mean forward : {np.mean(forward_distances):.2f}m +/- {np.std(forward_distances):.2f}m")


def visualize_interactive(model_path, cmd_vx=0.5, cmd_vy=0.0, cmd_yaw=0.0,
                          use_terrain=False, difficulty=0.5):
    """Visualize policy in MuJoCo interactive viewer."""
    model_ppo, vecnorm_path = load_model_and_vecnorm(model_path)

    # VecNormalize wrapper for observation normalization
    vec_env = DummyVecEnv([lambda: make_env(use_terrain, difficulty)])
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded VecNormalize from: {vecnorm_path}")

    # Raw environment for viewer
    env = make_env(use_terrain, difficulty)
    obs, info = env.reset()
    env._command = np.array([cmd_vx, cmd_vy, cmd_yaw])

    mode = "terrain" if use_terrain else "flat"
    print(f"\n=== Interactive Visualization ({mode}) ===")
    print(f"  Command: vx={cmd_vx:.2f}, vy={cmd_vy:.2f}, yaw={cmd_yaw:.2f}")
    print(f"  Close the viewer window to exit.")
    print()

    def normalize_obs(obs):
        if isinstance(vec_env, VecNormalize):
            return vec_env.normalize_obs(obs.reshape(1, -1)).flatten()
        return obs

    total_reward = 0.0
    step_count = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            obs_norm = normalize_obs(obs)
            action, _ = model_ppo.predict(obs_norm.reshape(1, -1), deterministic=True)
            action = action.flatten()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            if terminated or truncated:
                print(f"  Episode done: steps={step_count}, reward={total_reward:.2f}, "
                      f"forward={info['base_pos'][0]:.2f}m")
                obs, info = env.reset()
                env._command = np.array([cmd_vx, cmd_vy, cmd_yaw])
                total_reward = 0.0
                step_count = 0

            viewer.sync()

            elapsed = time.time() - step_start
            sleep_time = env.control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()
    vec_env.close()


def record_video(model_path, output_path, n_steps=500, cmd_vx=0.5,
                 use_terrain=False, difficulty=0.5):
    """Record a video of the policy running."""
    try:
        import mediapy as media
    except ImportError:
        print("Install mediapy for video recording: pip install mediapy")
        return

    model_ppo, vecnorm_path = load_model_and_vecnorm(model_path)

    vec_env = DummyVecEnv([lambda: make_env(use_terrain, difficulty)])
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    env = make_env(use_terrain, difficulty, render_mode="rgb_array")
    obs, info = env.reset()
    env._command = np.array([cmd_vx, 0.0, 0.0])

    frames = []
    renderer = mujoco.Renderer(env.model, height=480, width=640)

    for _ in range(n_steps):
        if isinstance(vec_env, VecNormalize):
            obs_norm = vec_env.normalize_obs(obs.reshape(1, -1)).flatten()
        else:
            obs_norm = obs

        action, _ = model_ppo.predict(obs_norm.reshape(1, -1), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action.flatten())

        renderer.update_scene(env.data)
        frames.append(renderer.render())

        if terminated or truncated:
            obs, info = env.reset()
            env._command = np.array([cmd_vx, 0.0, 0.0])

    renderer.close()
    env.close()
    vec_env.close()

    media.write_video(output_path, frames, fps=50)
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Go2 Locomotion Policy")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.zip)")
    parser.add_argument("--no-viewer", action="store_true", help="Run headless evaluation only")
    parser.add_argument("--record", type=str, default=None, help="Record video to file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--cmd_vx", type=float, default=0.5, help="Forward velocity command")
    parser.add_argument("--cmd_vy", type=float, default=0.0, help="Lateral velocity command")
    parser.add_argument("--cmd_yaw", type=float, default=0.0, help="Yaw rate command")
    parser.add_argument("--terrain", action="store_true", help="Use terrain environment")
    parser.add_argument("--difficulty", type=float, default=0.5, help="Terrain difficulty (0-1)")
    args = parser.parse_args()

    if args.record:
        record_video(args.model, args.record, cmd_vx=args.cmd_vx,
                     use_terrain=args.terrain, difficulty=args.difficulty)
    elif args.no_viewer:
        evaluate_headless(args.model, n_episodes=args.episodes,
                         use_terrain=args.terrain, difficulty=args.difficulty)
    else:
        visualize_interactive(args.model, cmd_vx=args.cmd_vx,
                            cmd_vy=args.cmd_vy, cmd_yaw=args.cmd_yaw,
                            use_terrain=args.terrain, difficulty=args.difficulty)
