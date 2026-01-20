"""
PPO training script for MyPickCube-v0 custom environment
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv

# Import custom environment to register it
import my_pick_cube_env


def main():
    print("=" * 60)
    print("Training PPO on MyPickCube-v0 (PANDA robot, state obs)")
    print("=" * 60)

    # Training environment
    print("\n[1/4] Creating training environment...")
    train_env = gym.make(
        "MyPickCube-v0",
        num_envs=32,  # parallel environments
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        reward_mode="dense",  # Use dense reward! ⭐
        render_mode=None,  # no rendering during training
        sim_backend="auto",  # auto select GPU/CPU
    )
    max_episode_steps = gym_utils.find_max_episode_steps_value(train_env)
    train_env = ManiSkillSB3VectorEnv(train_env)
    print(f"   - Training envs: 32 parallel")
    print(f"   - Max episode steps: {max_episode_steps}")
    print(f"   - Obs space: {train_env.observation_space}")
    print(f"   - Action space: {train_env.action_space}")

    # Evaluation environment (fewer envs, with video recording)
    print("\n[2/4] Creating evaluation environment...")
    eval_env = gym.make(
        "MyPickCube-v0",
        num_envs=8,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        reward_mode="dense",  # Use dense reward! ⭐
        render_mode="rgb_array",  # enable rendering for eval videos
        sim_backend="auto",
    )
    eval_env = RecordEpisode(
        eval_env,
        output_dir="eval_videos",
        save_video=True,
        trajectory_name="eval",
        max_steps_per_video=max_episode_steps,
        save_trajectory=False,
    )
    eval_env = ManiSkillSB3VectorEnv(eval_env)
    print(f"   - Eval envs: 8 parallel")
    print(f"   - Videos will be saved to: eval_videos/")

    # Create PPO model
    print("\n[3/4] Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,# 3e-4 (기본), 1e-4 (안정), 1e-3 (빠름)
        n_steps=256,  # steps per env per update
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
    )
    print(f"   - Policy: MlpPolicy")
    print(f"   - Learning rate: 3e-4")
    print(f"   - Steps per update: {256 * 32} (256 steps × 32 envs)")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,  # save every 50k steps
        save_path="./checkpoints/",
        name_prefix="ppo_pickcube",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=25_000,  # evaluate every 25k steps
        n_eval_episodes=8,
        deterministic=True,
    )

    # Train
    print("\n[4/4] Starting training...")
    print("   - Total timesteps: 500,000")
    print("   - Checkpoints saved every 50k steps to: ./checkpoints/")
    print("   - Best model saved to: ./best_model/")
    print("   - Tensorboard logs: ./tensorboard_logs/")
    print("=" * 60)
    
    model.learn(
        total_timesteps=750_000,
        callback=[checkpoint_callback, eval_callback],
    )

    # Save final model
    model.save("ppo_pickcube_final")
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved: ppo_pickcube_final.zip")
    print("=" * 60)

    # Cleanup
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
