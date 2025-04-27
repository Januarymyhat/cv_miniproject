import re
import matplotlib.pyplot as plt
from collections import defaultdict

with open("dataset/without_training_process.txt", "r", encoding="utf-8") as f:
    log_text = f.read()

# Find Epoch, Batch, D_loss, G_loss, L1
pattern = r"Epoch \[(\d+)/\d+\] Batch \[(\d+)/\d+\] D_loss: ([\d.]+), G_loss: ([\d.]+), L1: ([\d.]+)"
matches = re.findall(pattern, log_text)

# Store the time series of each loss
losses = defaultdict(lambda: {'epoch': [], 'D_loss': [], 'G_loss': [], 'L1': []})

for epoch, batch, d, g, l1 in matches:
    batch = int(batch)
    if batch % 20 == 0:
        losses[batch]['epoch'].append(int(epoch))
        losses[batch]['D_loss'].append(float(d))
        losses[batch]['G_loss'].append(float(g))
        losses[batch]['L1'].append(float(l1))

# Draw
plt.figure(figsize=(12, 7))
for batch in sorted(losses.keys()):
    plt.plot(losses[batch]['epoch'], losses[batch]['D_loss'], marker='o', linestyle='--', label=f'D_loss @Batch {batch}')
    plt.plot(losses[batch]['epoch'], losses[batch]['G_loss'], marker='s', linestyle='-', label=f'G_loss @Batch {batch}')
    plt.plot(losses[batch]['epoch'], losses[batch]['L1'], marker='^', linestyle=':', label=f'L1 @Batch {batch}')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Losses over Epochs (without encoding process)")
plt.legend(title="Loss Type + Batch", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
