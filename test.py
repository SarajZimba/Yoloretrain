# from ultralytics import YOLO

# # Load the model
# model = YOLO('yolov8n.pt')

# # Perform prediction and save the image with bounding boxes
# # results = model.predict(
# #     source='Dog.jpeg',  # Input image
# #     conf=0.25,          # Confidence threshold
# #     save=True           # Save the output image with rectangles
# # )

# # Convert to ONNX
# model.export(format="onnx")


import torch
from ultralytics import YOLO

# Load your YOLOv8 model (replace with your trained YOLOv8 model file)
model = YOLO("yolov8n.pt")  # or replace with your own model path
model.eval()

dummy_input = torch.randn(1, 3, 640, 640)  

# Export the model to ONNX format
torch.onnx.export(
    model.model,  # access the underlying PyTorch model of YOLOv8
    dummy_input,
    "yolov8model.onnx",  # Specify the desired output ONNX file
    export_params=True,
    opset_version=11,  # Specify the ONNX opset version (11 is commonly used)
    input_names=["input"],  # Name for the input tensor
    output_names=["output"],  # Name for the output tensor
)

print("YOLOv8 model exported to ONNX format.")