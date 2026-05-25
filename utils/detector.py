from ultralytics import YOLO
import cv2
import time

# LOAD MODEL
model = YOLO("model/best.pt")

# DETECTION FUNCTION
def detect_frame(frame, conf=0.5):

    start = time.time()

    # Predict
    results = model(frame, conf=conf)

    # Annotated frame
    annotated_frame = results[0].plot()

    # FPS
    end = time.time()
    fps = 1 / (end - start)

    # Total detections
    total_detection = len(results[0].boxes)

    # Convert color
    annotated_frame = cv2.cvtColor(
        annotated_frame,
        cv2.COLOR_BGR2RGB
    )

    return annotated_frame, fps, total_detection