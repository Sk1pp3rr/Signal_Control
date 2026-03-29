from stable_baselines3 import PPO
from env import SumoEnv
import os

def train():
    # Ścieżka do mapy
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "maps", "krzyzak", "krzyzak.sumocfg"))

    # 1. Inicjalizacja środowiska
    env = SumoEnv(CONFIG_PATH, gui=False)

    # 2. chose "brain" (PPO)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_sumo_tensorboard/")

    # 3. Start
    print("Start training...")
    model.learn(total_timesteps=200000)

    # 4. Zapisanie modelu
    model.save("model_krzyzak_v1")
    print("Model saved!")

if __name__ == "__main__":
    train()