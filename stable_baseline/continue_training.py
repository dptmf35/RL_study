"""
Continue training from existing model with improved reward
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv

# Import custom environment (with updated rewards)
import my_pick_cube_env


def main():
    print("=" * 60)
    print("Continuing PPO training with improved rewards")
    print("=" * 60)

    # Training environment
    print("\n[1/3] Creating training environment...")
    train_env = gym.make(
        "MyPickCube-v0",
        num_envs=32,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode=None,
        sim_backend="auto",
    )
    max_episode_steps = gym_utils.find_max_episode_steps_value(train_env)
    train_env = ManiSkillSB3VectorEnv(train_env)
    print(f"   ✓ Training envs: 32 parallel")

    # Evaluation environment
    print("\n[2/3] Creating evaluation environment...")
    eval_env = gym.make(
        "MyPickCube-v0",
        num_envs=8,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        sim_backend="auto",
    )
    eval_env = RecordEpisode(
        eval_env,
        output_dir="eval_videos_v2",
        save_video=True,
        trajectory_name="eval_v2",
        max_steps_per_video=max_episode_steps,
        save_trajectory=False,
    )
    eval_env = ManiSkillSB3VectorEnv(eval_env)
    print(f"   ✓ Eval envs: 8 parallel")

    # Load existing model
    print("\n[3/3] Loading existing model and continuing training...")
    model = PPO.load("ppo_pickcube_final", env=train_env)
    print(f"   ✓ Loaded: ppo_pickcube_final.zip")
    print(f"   - Now training with UPDATED rewards (grasp & lift emphasized)")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="./checkpoints_v2/",
        name_prefix="ppo_pickcube_v2",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model_v2/",
        log_path="./eval_logs_v2/",
        eval_freq=25_000,
        n_eval_episodes=8,
        deterministic=True,
    )

    # Continue training
    print("\n" + "=" * 60)
    print("Training additional 500,000 steps...")
    print("With improved reward structure:")
    print("  - Reaching: 2.0 (was 1.0)")
    print("  - Grasping: 5.0 (was 2.0) ← 큐브 잡기 강조!")
    print("  - Lifting: 5.0 (was 3.0) ← 들어올리기 강조!")
    print("  - Success: 10.0 (was 5.0)")
    print("  - Goal height: 0.2m (was 0.3m) ← 더 쉽게!")
    print("=" * 60)
    
    model.learn(
        total_timesteps=500_000,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=False,  # Continue from previous timesteps
    )

    # Save final model
    model.save("ppo_pickcube_v2_final")
    print("\n" + "=" * 60)
    print("Additional training complete!")
    print(f"Model saved: ppo_pickcube_v2_final.zip")
    print("=" * 60)

    # Cleanup
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
