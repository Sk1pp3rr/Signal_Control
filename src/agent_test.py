import os
from stable_baselines3 import PPO
from env import SumoEnv


def test_trained_agent():
    # 1. Paths to files
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #MODEL_PATH = os.path.dirname(os.path.join(BASE_DIR, "..","results","ppo_sumo_model_100000_steps.zip"))
    CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "maps", "krzyzak", "krzyzak.sumocfg"))
    MODEL_PATH = os.path.join(BASE_DIR, "ppo_sumo_model_120000_steps.zip")

    # 2. Init from GUI
    # gui=True, to watch the agent
    env = SumoEnv(CONFIG_PATH, gui=True)

    # 3. Loading trained model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: no model found in: {MODEL_PATH}")
        return

    model = PPO.load(MODEL_PATH)
    print("Loaded successfully!")

    # 4. Test loop
    obs, info = env.reset()
    total_reward = 0

    print("test start...")

    try:
        for i in range(500):
            # Prediction of the best possible action
            action, _states = model.predict(obs, deterministic=True)

            # performing action in SUMO
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if i % 50 == 0:
                print(f"Step {i}: Cumulated reward= {total_reward:.2f}")

            if terminated or truncated:
                print("The END.")
                break

    except Exception as e:
        print(f"Error during test: {e}")

    finally:
        print(f"Final result: {total_reward:.2f}")
        input("press ENTER to continue...")
        env.close()


if __name__ == "__main__":
    test_trained_agent()