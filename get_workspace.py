from roboflow import Roboflow

rf = Roboflow(api_key='N7BgbRtie2bXDot8wul6')
project = rf.workspace()
print(project)
# dataset = project.version(1).download('yolov8')