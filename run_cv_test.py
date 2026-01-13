from src.utils import load_image_paths, create_stratified_folds
from src.config import DATA_DIR, NUM_FOLDS
import numpy as np

data_dir = DATA_DIR + "/chest_xray"
image_paths, labels = load_image_paths(data_dir)

folds = create_stratified_folds(labels, n_splits=NUM_FOLDS)

for f in folds:
    train_labels = np.array(labels)[f["train_idx"]]
    val_labels = np.array(labels)[f["val_idx"]]

    print(f"\nFold {f['fold']}")
    print(" Train size:", len(train_labels))
    print(" Val size:", len(val_labels))
    print(" Train Normal / Pneumonia:",
          (train_labels == 0).sum(), "/", (train_labels == 1).sum())
    print(" Val Normal / Pneumonia:",
          (val_labels == 0).sum(), "/", (val_labels == 1).sum())

    overlap = set(f["train_idx"]).intersection(set(f["val_idx"]))
    print(" Overlap:", len(overlap))
