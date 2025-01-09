from ultralytics import YOLO

# Path to the dataset folder
dataset_path = 'Face-Detection-1/data.yaml'

# Load the YOLOv8 model (you can use 'yolov8n.pt', 'yolov8s.pt', etc.)
model = YOLO('yolov8n.pt')  # Choose the appropriate YOLO model for your task

# Start training the model
model.train(data=dataset_path, epochs=50, imgsz=640, batch=16)