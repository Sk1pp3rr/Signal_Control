import random

from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps, raw_env, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.current_phase = -1
        self.raw_env = raw_env

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.total_timesteps

        if progress < 0.20:
            new_phase = 0  # night
        elif progress < 0.45:
            new_phase = 3  #Day
        elif progress < 0.60:
            new_phase = 4 # afternoon peak
        elif progress < 0.75:
            new_phase = 2 # morning peak
        else:
            new_phase = None  # random

        if new_phase != self.current_phase:
            self.current_phase = new_phase
            if self.verbose > 0:
                print(f"\n[CURRICULUM] Phase: {new_phase} (Progress: {progress:.1%})")

            self.raw_env.set_target_phase(new_phase)

        return True