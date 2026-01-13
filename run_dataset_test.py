from src.utils import load_image_paths
from src.dataset import ChestXRayDataset
from src.config import DATA_DIR
from torch.utils.data import DataLoader

data_dir = DATA_DIR + "/chest_xray"

image_paths, labels = load_image_paths(data_dir)

print("Total images:", len(image_paths))
print("Normal:", labels.count(0))
print("Pneumonia:", labels.count(1))

dataset = ChestXRayDataset(image_paths, labels, augment=True)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

images, targets = next(iter(loader))
print("Image batch shape:", images.shape)
print("Labels:", targets)
