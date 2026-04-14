import pandas as pd
from tbparse import SummaryReader

log_dir = "ppo_sumo_tensorboard/"
reader = SummaryReader(log_dir, pivot=True)
df = reader.scalars

df.to_csv(f"{log_dir}summary.csv", index=False)
print("succes")