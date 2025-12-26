import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Style an toàn trên mọi máy
plt.style.use("ggplot")

# Load CSV
loss = pd.read_csv("train_log.csv")
precision = pd.read_csv("precision.csv")
recall = pd.read_csv("recall.csv")
ndcg = pd.read_csv("ndcg.csv")

# Màu đẹp, đồng nhất
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

fig = plt.figure(figsize=(17,12))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25)

# 1. Loss
ax1 = fig.add_subplot(gs[0,0])
ax1.plot(loss["epoch"], loss["loss"], marker='o', linewidth=3, color=colors[0])
ax1.set_title("Loss Curve", fontsize=18)
ax1.set_xlabel("Epoch", fontsize=14)
ax1.set_ylabel("Loss", fontsize=14)

# 2. Precision@K
ax2 = fig.add_subplot(gs[0,1])
for i, k in enumerate(["precision10","precision20","precision30","precision40","precision50"]):
    ax2.plot(precision["epoch"], precision[k], marker='o', linewidth=3, label=k, color=colors[i])
ax2.set_title("Precision@K", fontsize=18)
ax2.set_xlabel("Epoch", fontsize=14)
ax2.set_ylabel("Precision", fontsize=14)
ax2.legend(fontsize=11)

# 3. Recall@K
ax3 = fig.add_subplot(gs[1,0])
for i, k in enumerate(["recall10","recall20","recall30","recall40","recall50"]):
    ax3.plot(recall["epoch"], recall[k], marker='o', linewidth=3, label=k, color=colors[i])
ax3.set_title("Recall@K", fontsize=18)
ax3.set_xlabel("Epoch", fontsize=14)
ax3.set_ylabel("Recall", fontsize=14)
ax3.legend(fontsize=11)

# 4. NDCG@K
ax4 = fig.add_subplot(gs[1,1])
for i, k in enumerate(["ndcg10","ndcg20","ndcg30","ndcg40","ndcg50"]):
    ax4.plot(ndcg["epoch"], ndcg[k], marker='o', linewidth=3, label=k, color=colors[i])
ax4.set_title("NDCG@K", fontsize=18)
ax4.set_xlabel("Epoch", fontsize=14)
ax4.set_ylabel("NDCG", fontsize=14)
ax4.legend(fontsize=11)

plt.tight_layout()
plt.show()
