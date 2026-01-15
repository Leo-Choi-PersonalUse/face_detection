# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import supervision as sv
import cv2

# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

# init camera
cap = cv2.VideoCapture(0)

print("Starting camera... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # inference
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # annotate
    for box, confidence in zip(detections.xyxy, detections.confidence):
        if confidence < 0.9:
            continue

        x1, y1, x2, y2 = map(int, box)
        # Draw a square (rectangle) around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display match percentage
        text = f"{confidence * 100:.1f}%"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # display
    cv2.imshow("Face Detection", frame)
    
    # break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
