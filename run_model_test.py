from src.model import build_resnet50

model = build_resnet50()

total_params = 0
trainable_params = 0

for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        print("Trainable:", name)

print("\nTotal parameters:", total_params)
print("Trainable parameters:", trainable_params)
