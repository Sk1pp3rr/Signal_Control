import os
import time
from stable_baselines3 import PPO
import supersuit as ss
from core.env import SumoEnv
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "../..", "maps", "krzyzak", "krzyzak.sumocfg"))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../..", "results", "model_krzyzak_final.zip"))

TEST_PHASE = 3
TEST_STEPS = 500


def prepare_wrapped_env(target_phase=None):
    raw_env = SumoEnv(CONFIG_PATH, gui=True)
    if target_phase is not None:
        raw_env.set_target_phase(target_phase)

    env = ss.pettingzoo_env_to_vec_env_v1(raw_env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
    return env


def test_trained_agent():
    print("\n" + "=" * 50)
    print("Our Agent")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH) and not os.path.exists(MODEL_PATH + ".zip"):
        print(f"Error: no model found in: {MODEL_PATH}")
        return 0

    model = PPO.load(MODEL_PATH)
    print("Loaded successfully!")

    env = prepare_wrapped_env(TEST_PHASE)
    obs = env.reset()
    total_reward = 0

    print("start testing our agent...")

    try:
        for i in range(TEST_STEPS):
            action, _states = model.predict(obs, deterministic=True)

            obs, rewards, dones, infos = env.step(action)
            total_reward += sum(rewards)

            if i > 0 and i % 50 == 0:
                print(f"Step {i:03d} | total reward = {total_reward:.2f}")

    except Exception as e:
        print(f"Error during test: {e}")

    finally:
        print(f"Total score of our agent: {total_reward:.2f}")
        env.close()

    return total_reward


def test_random_agent():
    print("\n" + "=" * 50)
    print("Random Agent")
    print("=" * 50)

    env = prepare_wrapped_env(TEST_PHASE)
    obs = env.reset()
    total_reward = 0

    print("start testing random agent...")

    try:
        for i in range(TEST_STEPS):
            action = [env.action_space.sample() for _ in range(env.num_envs)]

            obs, rewards, dones, infos = env.step(action)
            total_reward += sum(rewards)

            if i > 0 and i % 50 == 0:
                print(f"Step {i:03d} | total reward = {total_reward:.2f}")

    except Exception as e:
        print(f"Error during test: {e}")

    finally:
        print(f"Total score of random agent: {total_reward:.2f}")
        env.close()

    return total_reward


def test_fixed_time_agent():
    print("\n" + "=" * 50)
    print("fixed time agent")
    print("=" * 50)

    env = prepare_wrapped_env(TEST_PHASE)
    obs = env.reset()
    total_reward = 0

    STEPS_PER_PHASE = 8

    print("start testing fixed time agent...")

    try:
        for step in range(TEST_STEPS):
            current_action = (step // STEPS_PER_PHASE) % 2
            action = [current_action for _ in range(env.num_envs)]

            obs, rewards, dones, infos = env.step(action)
            total_reward += sum(rewards)

            if step > 0 and step % 50 == 0:
                print(f"Step {step:03d} | total reward = {total_reward:.2f}")

    except Exception as e:
        print(f"Error during test: {e}")

    finally:
        print(f"Total score of fixed-time agent: {total_reward:.2f}")
        env.close()

    return total_reward


if __name__ == "__main__":
    score_random = test_random_agent()
    time.sleep(2)

    score_fixed = test_fixed_time_agent()
    time.sleep(2)

    score_ai = test_trained_agent()

    print("The Results: (Random vs Fixed vs AI)")
    print(f"Random Agent:   {score_random:.2f} pkt")
    print(f"Fixed time Agent:     {score_fixed:.2f} pkt")
    print(f"Our Agent :) :         {score_ai:.2f} pkt")
    if score_ai>score_fixed and score_ai>score_random:
        print("Our agent is the best!")
    else:
        print("Our agent is not the best :(")

    kategorie=['Our_Agent', 'Fixed_Agent', 'Random_Agent']
    wartosci=[score_ai, score_fixed, score_random]

    plt.bar(kategorie, wartosci)

    plt.title("Performance Comparison of Traffic Light Control Agents")
    plt.xlabel("Agents")
    plt.ylabel("Performance")
    plt.savefig("Agents_score.png", dpi=300, bbox_inches='tight')
    plt.show()



