import os
from env import SumoEnv

def run_sanity_test():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "maps", "krzyzak", "krzyzak.sumocfg"))

    #starting with gui
    env = SumoEnv(CONFIG_PATH,gui = True)
    try:
        obs, info = env.reset()
        print(f"first observation: {obs}")

        for i in range(500):
            action = env.action_space.sample() #random choice from 0 to 1
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: {action}, reward: {reward}, terminated: {terminated}, observation: {obs}")

        print("succes!")

    except Exception as e:
        print(f"failed! error: {e}")

    finally:
        input("Press Enter...")
        env.close()

if __name__ == "__main__":
    run_sanity_test()
