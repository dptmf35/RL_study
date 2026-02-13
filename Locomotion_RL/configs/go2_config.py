"""Training configuration for Go2 locomotion."""


TRAIN_CONFIG = {
    # Environment
    "env": {
        "control_dt": 0.02,        # 50 Hz control
        "sim_dt": 0.005,           # 200 Hz physics
        "max_episode_steps": 1000, # 20s episodes
        "kp": 40.0,
        "kd": 1.0,
        "action_scale": 0.25,
        "cmd_vx_range": (0.0, 1.0),
        "cmd_vy_range": (-0.3, 0.3),
        "cmd_yaw_range": (-0.5, 0.5),
        # Reward scales
        "rew_forward_vel": 1.0,
        "rew_lateral_vel": 0.5,
        "rew_yaw_rate": 0.5,
        "rew_torque": -1e-5,
        "rew_joint_acc": -2.5e-7,
        "rew_action_rate": -0.01,
        "rew_orientation": -1.0,
        "rew_base_height": -10.0,
        "rew_feet_air_time": 1.0,
        "rew_alive": 0.5,
        # Termination
        "max_body_tilt": 0.8,
        "min_base_height": 0.15,
    },

    # Terrain-specific env config (extra keys for Go2TerrainEnv)
    "terrain_env": {
        "difficulty": 0.5,
        "terrain_seed": None,
    },

    # Reward overrides for terrain training (stronger forward incentive)
    "terrain_reward_overrides": {
        "rew_forward_vel": 2.0,     # 1.0 → 2.0 (incentivize walking over standing)
        "rew_feet_air_time": 2.0,   # 1.0 → 2.0 (encourage gait cycles)
        "rew_alive": 0.1,           # 0.5 → 0.1 (less reward for standing still)
    },

    # PPO hyperparameters
    "ppo": {
        "learning_rate": 1e-4,          # 3e-4 → 1e-4 (more stable updates)
        "n_steps": 4096,                # 2048 → 4096 (better gradient estimates)
        "batch_size": 128,              # 64 → 128 (smoother updates)
        "n_epochs": 5,                  # 10 → 5 (prevent over-fitting per batch)
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.005,             # 0.01 → 0.005 (prevent std explosion)
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "net_arch": [256, 256, 128],
            "log_std_init": -1.0,       # lower initial std (~0.37 vs default 1.0)
        },
    },

    # Training
    "total_timesteps": 5_000_000,
    "n_envs": 4,
    "eval_freq": 50_000,
    "n_eval_episodes": 5,
    "save_freq": 100_000,
    "log_dir": "logs",
    "model_dir": "checkpoints",
}
