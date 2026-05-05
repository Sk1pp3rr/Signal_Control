import os
import random
import matplotlib.pyplot as plt
import traci
from stable_baselines3 import PPO

from env import SumoEnv
from baseline_fixed import run_fixed_baseline
from other_metods.baseline_actuated import run_actuated_baseline

MAX_SIM_TIME = 3600
SEED = 12345


def run_ppo_agent(config_path, model_path, max_sim_time=3600):
    print("--- START AGENT PPO ---")
    random.seed(SEED)

    env = SumoEnv(config_path, gui=False, rank=12)
    model = PPO.load(model_path)

    metrics_log = {
        'queue_history': [],
        'halting_history': [],
        'priority_wait_dict': {}
    }

    obs, info = env.reset(seed=SEED)
    conn = traci.getConnection(env.sumo.label)

    try:
        while conn.simulation.getTime() < max_sim_time:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            j_metrics = env.sumo.get_junction_metrics()
            metrics_log['queue_history'].append(j_metrics['max_jam_length'])
            metrics_log['halting_history'].append(j_metrics['total_halting'])

            passed = env.sumo.get_gps_status()
            for v in passed:
                if v['type'] in ["ambulance", "city_bus"]:
                    veh_id = v['id']
                    current_wait = v.get('wait', 0)
                    metrics_log['priority_wait_dict'][veh_id] = max(
                        metrics_log['priority_wait_dict'].get(veh_id, 0), current_wait)

            if terminated or truncated:
                break
    finally:
        env.close()

    print("--- END AGENT PPO ---")
    avg_queue = sum(metrics_log['queue_history']) / max(1, len(metrics_log['queue_history']))
    avg_halting = sum(metrics_log['halting_history']) / max(1, len(metrics_log['halting_history']))
    total_priority_wait = sum(metrics_log['priority_wait_dict'].values())

    final_metrics = {
        'queue_length': avg_queue,
        'total_halting': avg_halting,
        'priority_wait_time': total_priority_wait
    }

    return final_metrics


def plot_comparisons(fixed_metrics, actuated_metrics, ppo_metrics):
    labels = ['Fixed Time', 'Actuated', 'PPO Agent']

    queues = [fixed_metrics['queue_length'], actuated_metrics['queue_length'], ppo_metrics['queue_length']]
    halting = [fixed_metrics['total_halting'], actuated_metrics['total_halting'], ppo_metrics['total_halting']]

    # Zmiana na 1 wiersz, 2 kolumny
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Comparison of traffic lights models (Sim time: {MAX_SIM_TIME}s)', fontsize=16)

    colors = ['#ff9999', '#66b3ff', '#99ff99']

    axes[0].bar(labels, queues, color=colors)
    axes[0].set_title('Average Queue Length (m)\n[Less = Better]')

    axes[1].bar(labels, halting, color=colors)
    axes[1].set_title('Average Halting Vehicles\n[Less = Better]')

    plt.tight_layout()
    plt.savefig('time_comparison_of_models.png', dpi=500)
    plt.show()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "maps", "krzyzak", "krzyzak.sumocfg"))
    MODEL_PATH = os.path.join(BASE_DIR, "..","..","models", "model_krzyzak_v4.zip")

    res_fixed = run_fixed_baseline(CONFIG_PATH, max_sim_time=MAX_SIM_TIME, seed=SEED)
    res_actuated = run_actuated_baseline(CONFIG_PATH, max_sim_time=MAX_SIM_TIME, seed=SEED)
    res_ppo = run_ppo_agent(CONFIG_PATH, MODEL_PATH, max_sim_time=MAX_SIM_TIME)

    plot_comparisons(res_fixed, res_actuated, res_ppo)