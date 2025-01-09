from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt') 

# Perform inference on an image
results = model('Dog.jpeg') 

# Display results (e.g., print detections)
# Perform inference on an image with a custom save directory
results = model('Dog.jpeg', save=True, save_dir='output') 

# Visualize results
# results.plot()