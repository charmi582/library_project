from ultralytics import YOLO

# Load a model
model = YOLO(r"C:\Users\user\Desktop\test\2號鏡頭\weights\best.pt")

# Training with custom augmentation parameters
model.train(data=r"C:\Users\user\Desktop\test\2\2data.yaml", epochs=50, hsv_h=0.03, hsv_s=0.6, hsv_v=0.5)

# Training without any augmentations (disabled values omitted for clarity)
model.train(
    data=r"C:\Users\user\Desktop\test\2\2data.yaml",
    epochs=50,
    hsv_h=0.2,
    hsv_s=0.3,
    hsv_v=0.2,
    translate=0.2,
    scale=0.2,
    fliplr=0.2,
    mosaic=0.2,
    erasing=0.2,
    auto_augment=None,
)