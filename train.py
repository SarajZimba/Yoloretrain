from roboflow import Roboflow

# Replace 'YOUR_API_KEY' with your actual API key
rf = Roboflow(api_key='N7BgbRtie2bXDot8wul6')

# Use the correct workspace and project names
project = rf.workspace('silverlinetester').project('face-detection-xmnwd')

# Download the dataset version (1 is the version number) in YOLOv8 format
dataset = project.version(1).download('yolov8')
