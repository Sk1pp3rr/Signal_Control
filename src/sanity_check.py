import os
import traceback
from env import SumoEnv


def run_sanity_test():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "maps", "krzyzak", "krzyzak.sumocfg"))

    # starting with gui
    env = SumoEnv(CONFIG_PATH, gui=True)
    try:
        obs, info = env.reset(options={"target_phase": 2})
        to_midnight = env.current_step * 5
        hour = (to_midnight // 3600) % 24
        minute = (to_midnight % 3600) // 60
        print(f"First observation keys (Agents): {list(obs.keys())}")
        print(f"Virtual time of simulation beggining: {hour}:{minute} (Faza: {env.events.get_time_phase(env.current_step)})")

        for i in range(500):
            # Multi-Agent Action Sampling
            actions = {agent: env.action_space[agent].sample() for agent in env.agents}

            # Simulation step
            obs, reward, terminated, truncated, info = env.step(actions)

            #Check if all junctions finished
            is_terminated = all(terminated.values())
            is_truncated = all(truncated.values())

            if is_terminated or is_truncated:
                print(f"\nSimulation ended in step: {i + 1} (Terminated: {is_terminated}, Truncated: {is_truncated})")
                break

            to_midnight = env.current_step * 5
            hour = (to_midnight // 3600) % 24
            minute = (to_midnight % 3600) // 60
            print(f"Step {i + 1} | Action: {actions} | Rewards: {reward}, time: {hour}:{minute}")

        print("\nEnvironment work correctly!")

    except Exception as e:
        print("\n" + "=" * 40)
        print("ERROR TRACEBACK:")
        print("=" * 40)
        traceback.print_exc()
        print("=" * 40)

    finally:
        input("Press enter to close GUI SUMO...")
        env.close()


if __name__ == "__main__":
    run_sanity_test()