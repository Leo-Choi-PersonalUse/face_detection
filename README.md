# Face Detection and Recognition System

This project implements a real-time face detection and recognition system using YOLOv8 for detection and dlib (`face_recognition`) for identity matching.

## Features

- **Real-time Face Detection**: Uses a YOLOv8 model for fast and accurate face detection.
- **Identity Recognition**: Identifies known faces and labels them. Unrecognized faces are labeled as "Unknown".
- **Dynamic Registration**: easy-to-use registration system to add new people to the database while the program is running.
- **Persistent Storage**: Faces are stored as images in a folder structure, making it easy to manage the dataset manually if needed.

## Setup

1. **Prerequisites**: Ensure you have Python installed (3.8+ recommended).
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `face_recognition` depends on `dlib`, which may require CMake to be installed on your system if a pre-built wheel is not available.*

3. **Project Structure**:
   ```
   ├── main.py            # Main application script
   ├── FaceManager.py     # Class for handling face recognition logic
   ├── requirements.txt   # List of dependencies
   └── dataset/           # Folder where face images are stored (auto-created)
   ```

## Usage

1. **Run the Application**:
   ```bash
   python main.py
   ```
   The application will download the required YOLO model on the first run.

2. **Controls**:
   - `q`: Quit the application.
   - `r`: Register the currently detected face.

3. **How to Register a New Face**:
   - Stand in front of the camera so your face is detected (green box).
   - Press `r`. The video feed window will freeze.
   - Look at the terminal window; it will prompt you: `Enter name for the face:`.
   - Type the name and press Enter.
   - The face is now saved in the `dataset/` folder and will be recognized immediately.

## Data Management

The `dataset/` directory is organized by person name:
```
dataset/
    ├── Alice/
    │   ├── Alice_1705623000.jpg
    ├── Bob/
    │   ├── Bob_1705623005.jpg
```
You can manually add or remove photos from these folders. Each time you restart the application, it reloads the dataset.
