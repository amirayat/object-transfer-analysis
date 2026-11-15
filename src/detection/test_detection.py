
import cv2
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo11s.pt")

# Open input video
cap = cv2.VideoCapture("data/videos/demo2.mp4")

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

try:
    while True:
        # Step 1: Read video frame
        success, frame = cap.read()

        if not success:
            break

        # Step 2: Detect and track objects
        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0, 73],  # Person and Book
            conf=0.15,
            imgsz=1280,
            verbose=False
        )[0]

        # Step 3: Display detections
        annotated_frame = result.plot()

        cv2.imshow("YOLO Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
