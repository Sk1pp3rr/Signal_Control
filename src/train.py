import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import supersuit as ss

from SumoCurriculumWrapper import CurriculumCallback
from env import SumoEnv


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

    # jeżeli środowisko na wyniki nie istnieje to jest stworzone
    os.makedirs(RESULTS_DIR, exist_ok=True)

    raw_env = SumoEnv(CONFIG_PATH, gui=False)
    env = ss.pettingzoo_env_to_vec_env_v1(raw_env)

    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

    env = VecMonitor(env, filename=os.path.join(RESULTS_DIR, "monitor.csv"))

    total_timesteps = 2_000_000

    # 1. Init of environment
    #env = SumoEnv(CONFIG_PATH, gui=False)
    #env = SubprocVecEnv([make_env(CONFIG_PATH, i) for i in range(num_cpu)]    env

    #auto save after 50k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=RESULTS_DIR,
        name_prefix="ppo_sumo_model"
    )

    curriculum_callback = CurriculumCallback(
        total_timesteps=total_timesteps,
        raw_env=raw_env,
        verbose = 1
    )

    callbacks = CallbackList([checkpoint_callback, curriculum_callback])



    # 2. brain
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_sumo_tensorboard/", n_steps=2048, batch_size=512)

    # 3. Start
    print("Start training...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        print("Training interrupted by user")


    model.save("model_krzyzak_v4")
    print("Model saved!")

if __name__ == "__main__":
    train()