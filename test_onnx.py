import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# YOLOv8 class labels
yolo_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# def detect_and_save_image(input_image_path, output_image_path):
#     """
#     Detect objects in an image using YOLOv8 and save the resulting image with bounding boxes.

#     :param input_image_path: Path to the input image file
#     :param output_image_path: Path to save the detected image
#     """
#     # Load image and prepare input
#     img = Image.open(input_image_path)
#     img_width, img_height = img.size
#     img_resized = img.resize((640, 640)).convert("RGB")
#     input_tensor = np.array(img_resized) / 255.0
#     input_tensor = input_tensor.transpose(2, 0, 1).reshape(1, 3, 640, 640).astype(np.float32)

#     # Load ONNX model and run inference
#     model = ort.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])
#     outputs = model.run(["output0"], {"images": input_tensor})
#     detections = process_output(outputs[0], img_width, img_height)

#     # Draw bounding boxes on the original image
#     draw = ImageDraw.Draw(img)
#     font = ImageFont.load_default()

#     for box in detections:
#         x1, y1, x2, y2, label, prob = box
#         draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
#         draw.text((x1, y1), f"{label} {prob:.2f}", fill="red", font=font)

#     # Save the resulting image
#     img.save(output_image_path)
#     print(f"Detected image saved to: {output_image_path}")


# def process_output(output, img_width, img_height):
#     """
#     Process YOLOv8 raw output into bounding boxes.

#     :param output: YOLOv8 raw output
#     :param img_width: Original image width
#     :param img_height: Original image height
#     :return: List of bounding boxes [[x1, y1, x2, y2, label, probability], ...]
#     """
#     output = output[0].astype(float).transpose()
#     boxes = []

#     for row in output:
#         prob = row[4:].max()
#         if prob < 0.5:
#             continue
#         class_id = row[4:].argmax()
#         label = yolo_classes[class_id]
#         xc, yc, w, h = row[:4]
#         x1 = (xc - w / 2) / 640 * img_width
#         y1 = (yc - h / 2) / 640 * img_height
#         x2 = (xc + w / 2) / 640 * img_width
#         y2 = (yc + h / 2) / 640 * img_height
#         boxes.append([x1, y1, x2, y2, label, prob])

#     return boxes


# # Example usage
# detect_and_save_image("Dog.jpeg", "dog_detected.jpg")

def detect_and_save_image(input_image_path, output_image_path):
    """
    Detect objects in an image using YOLOv8 and save the resulting image with a single bounding box.
    
    :param input_image_path: Path to the input image file
    :param output_image_path: Path to save the detected image
    """
    # Load image and prepare input
    img = Image.open(input_image_path)
    img_width, img_height = img.size
    img_resized = img.resize((640, 640)).convert("RGB")
    input_tensor = np.array(img_resized) / 255.0
    input_tensor = input_tensor.transpose(2, 0, 1).reshape(1, 3, 640, 640).astype(np.float32)

    # Load ONNX model and run inference
    model = ort.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])
    outputs = model.run(["output0"], {"images": input_tensor})
    detections = process_output(outputs[0], img_width, img_height)

    # If there are detections, draw the one with the highest probability
    if detections:
        highest_prob_detection = max(detections, key=lambda x: x[5])  # x[5] is the probability
        x1, y1, x2, y2, label, prob = highest_prob_detection

        # Draw bounding box on the original image
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{label} {prob:.2f}", fill="red", font=font)

        # Save the resulting image
        img.save(output_image_path)
        print(f"Detected image saved to: {output_image_path}")
    else:
        print("No objects detected.")


def process_output(output, img_width, img_height):
    """
    Process YOLOv8 raw output into bounding boxes.
    
    :param output: YOLOv8 raw output
    :param img_width: Original image width
    :param img_height: Original image height
    :return: List of bounding boxes [[x1, y1, x2, y2, label, probability], ...]
    """
    output = output[0].astype(float).transpose()
    boxes = []

    for row in output:
        prob = row[4:].max()
        if prob < 0.5:
            continue
        class_id = row[4:].argmax()
        label = yolo_classes[class_id]
        xc, yc, w, h = row[:4]
        x1 = (xc - w / 2) / 640 * img_width
        y1 = (yc - h / 2) / 640 * img_height
        x2 = (xc + w / 2) / 640 * img_width
        y2 = (yc + h / 2) / 640 * img_height
        boxes.append([x1, y1, x2, y2, label, prob])

    return boxes


# Example usage
detect_and_save_image("Dog.jpeg", "dog_detected_2.jpg")

