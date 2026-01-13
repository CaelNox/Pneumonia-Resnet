import torch
import numpy as np
from torch.utils.data import DataLoader

from src.config import DATA_DIR, DEVICE
from src.utils import load_image_paths, create_stratified_folds
from src.dataset import ChestXRayDataset
from src.model import build_resnet50
from src.metrics import compute_metrics

# -------------------------
# Load data & folds
# -------------------------
data_dir = DATA_DIR + "/chest_xray"
image_paths, labels = load_image_paths(data_dir)

folds = create_stratified_folds(labels, n_splits=5)
fold = folds[0]

val_paths = [image_paths[i] for i in fold["val_idx"]]
val_labels = np.array([labels[i] for i in fold["val_idx"]])

val_dataset = ChestXRayDataset(val_paths, val_labels, augment=False)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

# -------------------------
# Load trained model
# -------------------------
model = build_resnet50().to(DEVICE)
model.load_state_dict(
    torch.load("experiments/folds/resnet50_fold0.pth", map_location=DEVICE)
)
model.eval()

# -------------------------
# Inference
# -------------------------
all_probs = []

with torch.no_grad():
    for images, _ in val_loader:
        images = images.to(DEVICE)
        outputs = model(images).squeeze()
        probs = torch.sigmoid(outputs)
        all_probs.extend(probs.cpu().numpy())

all_probs = np.array(all_probs)

# -------------------------
# Compute metrics
# -------------------------
metrics = compute_metrics(val_labels, all_probs)

print("\nEvaluation Metrics (Fold 0):")
for k, v in metrics.items():
    print(f"{k:20s}: {v:.4f}")

# -------------------------
# Save metrics to JSON
# -------------------------
import json
import os

os.makedirs("experiments/metrics", exist_ok=True)

with open("experiments/metrics/fold_0.json", "w") as f:
    json.dump(metrics, f, indent=4)
