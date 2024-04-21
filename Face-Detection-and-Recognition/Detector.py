import os
import cv2
from ultralytics import YOLO

VIDEOS_DIR = os.path.join('.', 'videos')
FRAMES_DIR = os.path.join('.', 'frames')

video_path = 'ss1.mp4'
video_path_out = '.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = os.path.join('.', 'yolov8n-face.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.7
frame_count = 0
save_frame = False

if not os.path.exists(FRAMES_DIR):
    os.makedirs(FRAMES_DIR)

while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            save_frame = True
            # cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Frame', frame)
    out.write(frame)
    if score > 0.83:
        if save_frame:
            frame_filename = os.path.join(FRAMES_DIR, f"detected_frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            save_frame = False

    ret, frame = cap.read()
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
