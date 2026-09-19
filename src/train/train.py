from pathlib import Path
from typing import Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecEnv, VecMonitor
import supersuit as ss

from core.SumoCurriculumWrapper import CurriculumCallback
from core.env import SumoEnv

# Global Configuration

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = (BASE_DIR / ".." / ".." / "maps" / "krzyzak" / "krzyzak.sumocfg").resolve()
RESULTS_DIR = (BASE_DIR / ".." / ".." / "results").resolve()
TENSORBOARD_LOG_DIR = "../ppo_sumo_tensorboard/"

TOTAL_TIMESTEPS = 2_000_000
CHECKPOINT_FREQ = 50_000
PPO_N_STEPS = 2048
PPO_BATCH_SIZE = 512

# Architectural Components

def prepare_infrastructure() -> None:
    """Initializes the directory structure required for saving results and logs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def setup_environment() -> Tuple[VecEnv, SumoEnv]:
    """
        Initializes and vectorizes the SUMO environment, preparing it for
        assimilation by Stable-Baselines3 algorithms.

        Returns:
            VecEnv: Optimized environment after applying SuperSuit wrappers.
            SumoEnv: Raw environment (direct reference for Curriculum Learning).
        """
    raw_env = SumoEnv(str(CONFIG_PATH), gui=False)

    # SuperSuit transformation cascade for Parameter Sharing
    env = ss.pettingzoo_env_to_vec_env_v1(raw_env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

    # Monitoring layer
    monitor_path = str(RESULTS_DIR / "monitor.csv")
    env = VecMonitor(env, filename=monitor_path)

    return env, raw_env

def build_callbacks(raw_env: SumoEnv) -> CallbackList:
    """Constructs an integrated chain of training callbacks."""
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(RESULTS_DIR),
        name_prefix="ppo_sumo_model"
    )

    curriculum_callback = CurriculumCallback(
        total_timesteps=TOTAL_TIMESTEPS,
        raw_env=raw_env,
        verbose=1
    )

    return CallbackList([checkpoint_callback, curriculum_callback])

def initialize_agent(env: VecEnv) -> PPO:
    """Instantiates and parameterizes the Proximal Policy Optimization model."""
    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE
    )

# Learning module

def train() -> None:
    """
    Main function controlling the reinforcement learning process.
    Integrates the environment, policy, and supervisory callback components.
    """
    prepare_infrastructure()

    print("[SYSTEM] Initializing SUMO simulation instance and vectorizing space...")
    env, raw_env = setup_environment()

    print("[SYSTEM] Registering asynchronous events (Curriculum & Checkpoints)...")
    callbacks = build_callbacks(raw_env)

    print("[SYSTEM] Instantiating PPO agent neural network...")
    model = initialize_agent(env)

    print(f"[SYSTEM] Starting policy optimization process ({TOTAL_TIMESTEPS} steps)...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    except KeyboardInterrupt:
        print("\n[SYSTEM] Process forcefully interrupted by the operator (KeyboardInterrupt).")
    finally:
        print("[SYSTEM] Finalization: Saving model and gracefully terminating SUMO processes...")
        final_model_path = str(RESULTS_DIR / "model_krzyzak_final")
        model.save(final_model_path)
        env.close()
        print(f"[SYSTEM] Operation complete. Model secured at: {final_model_path}")


if __name__ == "__main__":
    train()