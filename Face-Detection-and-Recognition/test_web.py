from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import cv2
import os

model = YOLO("yolov8n-face.pt")

# Specify the output directory
output_dir = "examples"
# output_path = os.path.join(output_dir, "output_video.mp4")
model.predict('Download (1).mp4', save=True, save_crop=True, imgsz=320, conf=0.83)

# Create the output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)
#
# # Predict and save the results
# results = model.predict(source="ss1.mp4", save=output_dir, show=True)

# Print the results
# print(results)