import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE_PATH = "krzyzak_v4.csv"
OUTPUT_IMAGE = "wykresy_treningu.png"


def plot_tensorboard_data():
    print(f"Wczytywanie danych z {CSV_FILE_PATH}...")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku {CSV_FILE_PATH}!")
        return

    metrics_to_plot = [
        ('rollout/ep_rew_mean', 'Średnia Nagroda (Reward Mean)', '#1f77b4'),
        ('rollout/ep_len_mean', 'Średnia Długość Epizodu', '#ff7f0e'),
        ('train/entropy_loss', 'Entropia (Entropy Loss)', '#2ca02c'),
        ('train/value_loss', 'Błąd Wartości (Value Loss)', '#d62728'),
        ('train/policy_gradient_loss', 'Błąd Polityki (Policy Gradient Loss)', '#9467bd'),
        ('train/clip_fraction', 'Odsetek odciętych aktualizacji (Clip Fraction)', '#8c564b')
    ]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))
    fig.suptitle('Podsumowanie Treningu Modelu PPO (SUMO)', fontsize=18, fontweight='bold', y=0.96)

    for i, (column, title, color) in enumerate(metrics_to_plot):
        ax = axes[i // 2, i % 2]

        if column in df.columns:
            data = df[['step', column]].dropna()

            steps_in_k = data['step'] / 1000

            ax.plot(steps_in_k, data[column], color=color, linewidth=2)

            ax.set_title(title, fontsize=14, pad=10)
            ax.set_xlabel('Kroki Treningowe [tysiące]', fontsize=11)
            ax.set_ylabel('Wartość', fontsize=11)

            ax.grid(True, linestyle='--', alpha=0.6)

            ax.fill_between(steps_in_k, data[column], min(data[column]), color=color, alpha=0.1)
        else:
            ax.text(0.5, 0.5, f'Brak danych dla:\n{column}',
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_axis_off()

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"Gotowe! Wykresy zapisano jako wysokiej jakości obraz: {OUTPUT_IMAGE}")

    plt.show()


if __name__ == "__main__":
    plot_tensorboard_data()