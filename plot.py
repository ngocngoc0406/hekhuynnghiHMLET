import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train_log.csv")

plt.figure(figsize=(8,5))
plt.plot(df["epoch"], df["loss"], marker='o', linewidth=2)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.show()
