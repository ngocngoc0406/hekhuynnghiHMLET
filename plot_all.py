import pandas as pd
import matplotlib.pyplot as plt

# Load data
loss = pd.read_csv("train_log.csv")
precision = pd.read_csv("precision.csv")
recall = pd.read_csv("recall.csv")
ndcg = pd.read_csv("ndcg.csv")

# Tạo figure tổng hợp
plt.figure(figsize=(16,12))

# -------------------------
# 1. Loss curve
# -------------------------
plt.subplot(2,2,1)
plt.plot(loss["epoch"], loss["loss"], marker='o', color='blue')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# -------------------------
# 2. Precision@K
# -------------------------
plt.subplot(2,2,2)
for k in ["precision10","precision20","precision30","precision40","precision50"]:
    plt.plot(precision["epoch"], precision[k], marker='o', label=k)
plt.title("Precision@K")
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.grid(True)
plt.legend()

# -------------------------
# 3. Recall@K
# -------------------------
plt.subplot(2,2,3)
for k in ["recall10","recall20","recall30","recall40","recall50"]:
    plt.plot(recall["epoch"], recall[k], marker='o', label=k)
plt.title("Recall@K")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.grid(True)
plt.legend()

# -------------------------
# 4. NDCG@K
# -------------------------
plt.subplot(2,2,4)
for k in ["ndcg10","ndcg20","ndcg30","ndcg40","ndcg50"]:
    plt.plot(ndcg["epoch"], ndcg[k], marker='o', label=k)
plt.title("NDCG@K")
plt.xlabel("Epoch")
plt.ylabel("NDCG")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
