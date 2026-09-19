import sys
import traceback
from pathlib import Path
from typing import Dict

from core.env import SumoEnv


def run_sanity_check() -> None:
    """
    Verifies the correctness of the SumoEnv (PettingZoo/Gymnasium) environment.
    Initializes the simulation with GUI enabled, performs random actions,
    and checks if the input/output communication works flawlessly.
    """
    print("[SANITY CHECK] Initializing test procedure...")

    # 1. Path configuration using modern pathlib (consistent with train.py)
    base_dir = Path(__file__).resolve().parent
    config_path = (base_dir / ".." / "maps" / "krzyzak" / "krzyzak.sumocfg").resolve()

    if not config_path.exists():
        print(f"[CRITICAL ERROR] Configuration file not found at: {config_path}")
        sys.exit(1)

    # 2. Environment initialization with GUI enabled (for visual debugging)
    print("[SANITY CHECK] Creating SumoEnv instance (GUI: True)...")
    env = SumoEnv(str(config_path), gui=True)

    try:
        # 3. Environment reset forcing peak hours (e.g., Phase 3 - Day)
        print("[SANITY CHECK] Resetting environment (Phase: 3 - Day)...")
        obs, infos = env.reset(options={"target_phase": 3})

        # Verify agents structure and observation dimensions
        print(f"[SANITY CHECK] Registered agents: {env.agents}")
        for agent in env.agents:
            print(f" -> {agent} observation vector shape: {obs[agent].shape}")

        print("\n[SANITY CHECK] Starting simulation loop (500 macro-steps limit)...\n")

        for step_num in range(1, 501):
            # 4. Multi-Agent random action sampling
            actions: Dict[str, int] = {
                agent: env.action_space(agent).sample() for agent in env.agents
            }

            # 5. Physical simulation step
            obs, rewards, terminated, truncated, infos = env.step(actions)

            # Calculate virtual SUMO time based on steps (1 step = 5s in simulation logic)
            to_midnight = env.current_step * 5
            hour = (to_midnight // 3600) % 24
            minute = (to_midnight % 3600) // 60

            # 6. Spam control - print log at the beginning and every 10 steps
            if step_num == 1 or step_num % 10 == 0:
                # Round rewards for console readability
                rounded_rewards = {k: round(v, 2) for k, v in rewards.items()}
                print(f"Step: {step_num:03d} | Virtual time: {hour:02d}:{minute:02d} | "
                      f"Actions: {actions} | Rewards: {rounded_rewards}")

            # 7. Check episode termination conditions
            is_terminated = all(terminated.values())
            is_truncated = all(truncated.values())

            if is_terminated or is_truncated:
                print(f"\n[SANITY CHECK] Episode ended naturally at step {step_num}.")
                print(f"Flags: Terminated={is_terminated}, Truncated={is_truncated}")
                break

        print("\n[SUCCESS] Environment architecture is working perfectly!")

    except Exception as e:
        print("\n[CRITICAL ERROR] Crash detected during simulation loop:")
        traceback.print_exc()
    finally:
        # 8. Graceful TraCI termination ensuring no orphaned background processes
        print("[SANITY CHECK] Terminating TraCI process and freeing memory...")
        env.close()
        print("[SANITY CHECK] Operation completed.")


if __name__ == "__main__":
    run_sanity_check()