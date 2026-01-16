# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import supervision as sv
import cv2
from FaceManager import FaceManager

# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

# init FaceManager
face_manager = FaceManager()

# init camera
cap = cv2.VideoCapture(0)

print("Starting camera...")
print("- Press 'q' to exit")
print("- Press 'r' to register the largest detected face")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # inference
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    largest_face = None
    largest_area = 0
    largest_box = None

    # annotate
    for box, confidence in zip(detections.xyxy, detections.confidence):
        if confidence < 0.9:
            continue

        x1, y1, x2, y2 = map(int, box)
        
        # Ensure coordinates are within frame boundsYY
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Calculate area for registration priority
        area = (x2 - x1) * (y2 - y1)
        if area > largest_area:
            largest_area = area
            largest_face = frame[y1:y2, x1:x2]
            largest_box = (x1, y1, x2, y2)

        # Recognize Face
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            name = face_manager.identify_face(face_crop)
            
            # Draw Process
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # display
    cv2.imshow("Face Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        if largest_face is not None:
            # Pause display slightly implies logic separation or just freeze
            print("\n!!! Registration Mode !!!")
            name = input("Enter name for the face: ").strip()
            if name:
                face_manager.register_face(largest_face, name)
            else:
                print("Registration cancelled.")
        else:
            print("No face detected to register!")

cap.release()
cv2.destroyAllWindows()
