from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env import SumoEnv
import os

def train():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "maps", "krzyzak", "krzyzak.sumocfg"))

    # 1. Init of environment
    env = SumoEnv(CONFIG_PATH, gui=False)

    RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))

    #auto save after 20k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=RESULTS_DIR,
        name_prefix="ppo_sumo_model"
    )

    # 2. chose "brain" (PPO)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_sumo_tensorboard/")

    # 3. Start
    print("Start training...")
    try:
        model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("Training interrupted by user")


    model.save("model_krzyzak_v3")
    print("Model saved!")

if __name__ == "__main__":
    train()