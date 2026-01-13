from src.utils import load_image_paths, create_stratified_folds
from src.dataset import ChestXRayDataset
from src.config import DATA_DIR, NUM_FOLDS
from torch.utils.data import DataLoader

data_dir = DATA_DIR + "/chest_xray"
image_paths, labels = load_image_paths(data_dir)

folds = create_stratified_folds(labels, n_splits=NUM_FOLDS)

fold = folds[0]

train_ds = ChestXRayDataset(
    [image_paths[i] for i in fold["train_idx"]],
    [labels[i] for i in fold["train_idx"]],
    augment=True
)

val_ds = ChestXRayDataset(
    [image_paths[i] for i in fold["val_idx"]],
    [labels[i] for i in fold["val_idx"]],
    augment=False
)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

x, y = next(iter(train_loader))
print("Train batch:", x.shape, y)

x, y = next(iter(val_loader))
print("Val batch:", x.shape, y)
