"""
Evaluate trained PPO model on MyPickCube-v0
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv

# Import custom environment
import my_pick_cube_env


def evaluate_model(model_path: str, num_episodes: int = 10, render: bool = True):
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to saved model (without .zip)
        num_episodes: Number of episodes to run
        render: Whether to save videos
    """
    print("=" * 60)
    print(f"Evaluating model: {model_path}")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading model...")
    model = PPO.load(model_path)
    print(f"   - Model loaded from: {model_path}.zip")

    # Create evaluation environment
    print("\n[2/3] Creating environment...")
    env_kwargs = dict(
        num_envs=4,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        sim_backend="auto",
    )
    
    if render:
        env_kwargs["render_mode"] = "rgb_array"
    
    eval_env = gym.make("MyPickCube-v0", **env_kwargs)
    max_episode_steps = gym_utils.find_max_episode_steps_value(eval_env)
    
    if render:
        eval_env = RecordEpisode(
            eval_env,
            output_dir="eval_videos_test",
            save_video=True,
            trajectory_name="test",
            max_steps_per_video=max_episode_steps,
            save_trajectory=False,
        )
    
    eval_env = ManiSkillSB3VectorEnv(eval_env)
    print(f"   - Envs: {env_kwargs['num_envs']} parallel")
    print(f"   - Max steps per episode: {max_episode_steps}")
    if render:
        print(f"   - Videos: eval_videos_test/")

    # Run evaluation
    print(f"\n[3/3] Running {num_episodes} episodes...")
    obs = eval_env.reset()
    
    episode_rewards = []
    episode_successes = []
    current_episode_rewards = np.zeros(env_kwargs['num_envs'])
    episodes_completed = 0
    
    while episodes_completed < num_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        
        current_episode_rewards += rewards
        
        # Check for done episodes
        for i, done in enumerate(dones):
            if done:
                episode_rewards.append(current_episode_rewards[i])
                current_episode_rewards[i] = 0
                
                # Get success from info
                if "final_info" in infos:
                    success = infos["final_info"][i].get("success", False)
                    episode_successes.append(success)
                
                episodes_completed += 1
                if episodes_completed >= num_episodes:
                    break

    eval_env.close()

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    
    if episode_successes:
        success_rate = np.mean(episode_successes) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="ppo_pickcube_final",
        help="Path to model (without .zip extension)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable video recording",
    )
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render,
    )
