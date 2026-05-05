from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from env import SumoEnv
import os

def make_env(config_path, rank, seed=0):
    def _init():
        env = SumoEnv(config_path, gui=False, rank=rank)
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    set_random_seed(seed)
    return _init

def train():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "maps", "krzyzak", "krzyzak.sumocfg"))
    RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))

    num_cpu = 8 #how many instances do we get here

    # 1. Init of environment
    #env = SumoEnv(CONFIG_PATH, gui=False)
    env = SubprocVecEnv([make_env(CONFIG_PATH, i) for i in range(num_cpu)])


    #auto save after 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // num_cpu,
        save_path=RESULTS_DIR,
        name_prefix="ppo_sumo_model"
    )

    # 2. brain
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_sumo_tensorboard/", n_steps=2048, batch_size= 64)

    # 3. Start
    print("Start training...")
    try:
        model.learn(total_timesteps=2000000, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("Training interrupted by user")


    model.save("model_krzyzak_v4")
    print("Model saved!")

if __name__ == "__main__":
    train()