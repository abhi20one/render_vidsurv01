from ultralytics import YOLO
import torch

# Load YOLO model
model = YOLO("models/yolov8s.pt")

# Get the model architecture
model_structure = model.model

# Ensure strides are tensors, then force only stride 32
model_structure.stride = torch.tensor([32.0])  # YOLO expects a tensor, not a list

# Keep only stride 32 anchor set
model_structure.model[-1].anchors = model_structure.model[-1].anchors[-1:, :]  # Keep only last set of anchors

# Save the modified model
model.save("models/yolov8s_stride32.pt")  # Use the YOLO model to save, not model_structure

print("Modified YOLO model saved as yolov8s_stride32.pt")
