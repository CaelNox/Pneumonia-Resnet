import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import DATA_DIR, DEVICE, NUM_FOLDS
from src.utils import load_image_paths, create_stratified_folds
from src.dataset import ChestXRayDataset
from src.model import build_resnet50
from src.train import train_one_epoch, get_loss_function

# -------------------------
# Load data
# -------------------------
data_dir = DATA_DIR + "/chest_xray"
image_paths, labels = load_image_paths(data_dir)

folds = create_stratified_folds(labels, n_splits=NUM_FOLDS)

# -------------------------
# Choose fold
# -------------------------
FOLD_ID = 0
fold = folds[FOLD_ID]

train_paths = [image_paths[i] for i in fold["train_idx"]]
train_labels = [labels[i] for i in fold["train_idx"]]

val_paths = [image_paths[i] for i in fold["val_idx"]]
val_labels = [labels[i] for i in fold["val_idx"]]

# -------------------------
# Datasets & loaders
# -------------------------
train_dataset = ChestXRayDataset(train_paths, train_labels, augment=True)
val_dataset = ChestXRayDataset(val_paths, val_labels, augment=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

# -------------------------
# Model, loss, optimizer
# -------------------------
model = build_resnet50().to(DEVICE)

criterion = get_loss_function(train_labels, DEVICE)

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

# -------------------------
# Training loop
# -------------------------
EPOCHS = 5

for epoch in range(EPOCHS):
    loss = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE
    )
    print(f"Fold {FOLD_ID} | Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")

# -------------------------
# Save model
# -------------------------
torch.save(
    model.state_dict(),
    f"experiments/folds/resnet50_fold{FOLD_ID}.pth"
)

print("Training complete.")
