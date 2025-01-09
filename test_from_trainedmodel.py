import cv2
import numpy as np
from ultralytics import YOLO

# Load the trained model (best.pt)
model = YOLO('runs/detect/train11/weights/best.pt')

# Perform inference on an image
results = model.predict('Face-Detection-1/train/images/1735810582055_jpeg.rf.9ee5d4e7ced95219d6f0fd06a9cc1f45.jpg')  # Replace with the path to your image

# Convert the results image to a format OpenCV can work with
img_with_boxes = results[0].plot()  # Get the image with bounding boxes (numpy array)

# Save the image using OpenCV (you can specify your desired format, such as '.jpg', '.png', etc.)
save_path = 'runs/detect/predict/custom_predicted_image_1.jpg'
cv2.imwrite(save_path, img_with_boxes)

# Optionally, print the results to check the prediction details
print(results)
