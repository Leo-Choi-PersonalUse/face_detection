import os
import cv2
import face_recognition
import numpy as np
import time

class FaceManager:
    def __init__(self, db_path="dataset"):
        self.db_path = db_path
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
        
        self.load_known_faces()

    def load_known_faces(self):
        """Loads face encodings from the dataset directory."""
        print("Loading known faces...")
        self.known_face_encodings = []
        self.known_face_names = []

        for person_name in os.listdir(self.db_path):
            person_dir = os.path.join(self.db_path, person_name)
            if not os.path.isdir(person_dir):
                continue

            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(person_dir, filename)
                    try:
                        # Use cv2 to load image to ensure compatibility with dlib expectations
                        # face_recognition.load_image_file uses PIL which might handle some formats differently
                        image = cv2.imread(image_path)
                        if image is None:
                            print(f"Warning: Could not read image {image_path}")
                            continue
                            
                        # Convert to RGB as dlib expects RGB
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        encodings = face_recognition.face_encodings(rgb_image)
                        if encodings:
                            self.known_face_encodings.append(encodings[0])
                            self.known_face_names.append(person_name)
                    except Exception as e:
                        print(f"Error loading {image_path}: {e}")
        print(f"Loaded {len(self.known_face_names)} known faces.")

    def register_face(self, face_image, name):
        """Saves a new face and updates known encodings."""
        person_dir = os.path.join(self.db_path, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            
        timestamp = int(time.time())
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(person_dir, filename)
        
        # Save image (convert BGR to RGB for face_recognition/saving if needed, but cv2 writes BGR)
        cv2.imwrite(filepath, face_image)
        print(f"Saved {name}'s face to {filepath}")

        # Update in-memory encodings
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_face)
        if encodings:
            self.known_face_encodings.append(encodings[0])
            self.known_face_names.append(name)
            print(f"Registered {name} successfully!")
            return True
        else:
            print("Could not encode face. Registration failed.")
            return False

    def identify_face(self, face_image):
        """Identifies a face image against known encodings."""
        if not self.known_face_encodings:
            return "Unknown"

        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        # Optimization: detected by YOLO, so assume the whole crop is the face
        # Location format: (top, right, bottom, left)
        h, w, _ = rgb_face.shape
        face_locations = [(0, w, h, 0)]
        
        face_encodings = face_recognition.face_encodings(rgb_face, known_face_locations=face_locations)

        if not face_encodings:
            return "Unknown"

        # Use the first face found in the crop (should be only one)
        encoding = face_encodings[0]
        
        matches = face_recognition.compare_faces(self.known_face_encodings, encoding, tolerance=0.5)
        name = "Unknown"

        face_distances = face_recognition.face_distance(self.known_face_encodings, encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = self.known_face_names[best_match_index]
            
        return name
