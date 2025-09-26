"""
Smart Attendance System
=======================

This file contains a complete, ready-to-run Python implementation and teaching guide
for a face-recognition-based Smart Attendance System. It was extracted and consolidated
from the previous conversation content you provided.

Usage: place this file in the project root and run it after installing dependencies.

Contents:
- Setup & prerequisites
- Recommended dataset/folder structure
- Full Python code (in functions) for:
  * loading & encoding known faces
  * marking attendance (CSV)
  * recognizing faces from webcam, an image, or a folder
  * utility helpers
- Notes on improvements, deployment, and troubleshooting

Requirements
------------
- Python 3.8+
- Libraries: face_recognition, opencv-python, numpy, pandas
  Install with:
    pip install face_recognition opencv-python numpy pandas

  * On Windows you may need to install CMake and a C++ build toolchain for dlib.
  * On Linux, install build essentials if face_recognition/dlib pip fails.

Dataset / Project structure
---------------------------
Assumed structure (create and populate manually or with Kaggle dataset excerpts):

smart_attendance/
├── known_faces/          # one folder per person, folder name = person's name
│   ├── Alice/
│   │   ├── alice1.jpg
│   │   └── alice2.jpg
│   ├── Bob/
│   │   └── bob1.jpg
├── attendance.csv        # will be created/updated by the script
└── attendance.py         # this script (or import functions)

Notes: Put multiple images per person if possible (different angles/lighting).

----- CODE STARTS BELOW -----

"""

import os
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional

# ---------------------------
# Configuration / constants
# ---------------------------
KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_CSV = 'attendance.csv'
CAMERA_INDEX = 0  # default webcam
FACE_MATCH_THRESHOLD = 0.6  # maximum distance for a match (lower = stricter)
SCALE_FACTOR = 0.25  # resize factor for faster processing in real-time

# ---------------------------
# 1) Load and encode known faces
# ---------------------------

# def load_and_encode_faces(known_faces_dir: str = KNOWN_FACES_DIR) -> Tuple[List[np.ndarray], List[str]]:
#     """Scan `known_faces_dir`, load each person's images and compute face encodings.

#     Directory layout expected:
#         known_faces/<PersonName>/<image files...>

#     Returns:
#         known_encodings: list of 128-d face encoding vectors
#         known_names: list of names (aligned with known_encodings)
#     """
#     known_encodings = []
#     known_names = []

    

#     if not os.path.exists(known_faces_dir):
#         raise FileNotFoundError(f"Known faces directory not found: {known_faces_dir}")

#     for person_name in sorted(os.listdir(known_faces_dir)):
#         person_folder = os.path.join(known_faces_dir, person_name)
#         if not os.path.isdir(person_folder):
#             continue

#         for filename in sorted(os.listdir(person_folder)):
#             if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 continue

#             file_path = os.path.join(person_folder, filename)
#             image = face_recognition.load_image_file(file_path)
#             encs = face_recognition.face_encodings(image)
#             if len(encs) == 0:
#                 # no face found - skip but print a warning
#                 print(f"[WARN] No face found in: {file_path}")
#                 continue

#             encoding = encs[0]
#             known_encodings.append(encoding)
#             known_names.append(person_name)

#     print(f"[INFO] Loaded encodings for {len(known_encodings)} face images (people: {len(set(known_names))}).")
#     return known_encodings, known_names
def load_and_encode_faces(known_faces_dir: str = KNOWN_FACES_DIR) -> Tuple[List[np.ndarray], List[str]]:
    """
    Scan `known_faces_dir`, load each person's images and compute face encodings.

    Directory layout expected:
        known_faces/<PersonName>/<image files...>

    Returns:
        known_encodings: list of 128-d face encoding vectors
        known_names: list of names (aligned with known_encodings)
    """
    known_encodings = []
    known_names = []

    print(f"[DEBUG] Starting face encoding load process...")
    print(f"[DEBUG] Checking if directory exists: {known_faces_dir}")

    if not os.path.exists(known_faces_dir):
        raise FileNotFoundError(f"Known faces directory not found: {known_faces_dir}")

    print(f"[DEBUG] Directory found. Scanning subfolders for people...")

    for person_name in sorted(os.listdir(known_faces_dir)):
        person_folder = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_folder):
            print(f"[DEBUG] Skipping {person_folder} (not a folder).")
            continue

        print(f"\n[INFO] Processing person: {person_name}")
        print(f"[DEBUG] Looking into folder: {person_folder}")

        for filename in sorted(os.listdir(person_folder)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                print(f"[DEBUG] Skipping file {filename} (not an image).")
                continue

            file_path = os.path.join(person_folder, filename)
            print(f"[DEBUG] Loading image: {file_path}")

            try:
                image = face_recognition.load_image_file(file_path)
                print(f"[DEBUG] Image loaded successfully. Shape: {image.shape}")
            except Exception as e:
                print(f"[ERROR] Could not load image {file_path}. Error: {e}")
                continue

            encs = face_recognition.face_encodings(image)
            if len(encs) == 0:
                print(f"[WARN] No face found in: {file_path} — skipping.")
                continue

            encoding = encs[0]
            known_encodings.append(encoding)
            known_names.append(person_name)

            print(f"[DEBUG] Face encoding computed. Encoding vector length: {len(encoding)}")
            print(f"[INFO] Added encoding for {person_name} from file {filename}")

    print("\n[SUMMARY]")
    print(f" - Total images processed: {len(known_encodings)}")
    print(f" - Unique people encoded: {len(set(known_names))}")
    print(f" - People list: {sorted(set(known_names))}")

    return known_encodings, known_names


# ---------------------------
# 2) Mark attendance
# ---------------------------

def mark_attendance(name: str, attendance_file: str = ATTENDANCE_CSV, only_once_per_day: bool = True) -> None:
    """Append a row to the CSV for the recognized person.

    If only_once_per_day is True, it will not duplicate a name for the same date.
    CSV columns: Name, Date, Time
    """
    today_str = datetime.now().strftime('%Y-%m-%d')
    now_time = datetime.now().strftime('%H:%M:%S')

    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
    else:
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])

    if only_once_per_day:
        already = df[(df['Name'] == name) & (df['Date'] == today_str)]
        if not already.empty:
            # Already marked today
            return

    # Append row (pandas >= 2.0: avoid deprecated DataFrame.append)
    new_row = {'Name': name, 'Date': today_str, 'Time': now_time}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(attendance_file, index=False)
    print(f"[ATTENDANCE] {name} marked at {now_time} on {today_str}")


# ---------------------------
# 3) Recognition helpers
# ---------------------------


def _match_face(
    face_encoding: np.ndarray,
    known_encodings: List[np.ndarray],
    known_names: List[str],
    threshold: float = FACE_MATCH_THRESHOLD,
) -> Tuple[str, Optional[float]]:
    """Return (name, distance) of the best match or ('Unknown', None).

    Uses Euclidean distance via face_recognition.face_distance.
    """
    if len(known_encodings) == 0:
        return 'Unknown', None

    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_idx = np.argmin(distances)
    best_distance = float(distances[best_idx])
    if best_distance <= threshold:
        return known_names[best_idx], best_distance
    else:
        return 'Unknown', best_distance


# ---------------------------
# 4) Recognize from a single image (for testing)
# ---------------------------

def recognize_faces_in_image(image_path: str, known_encodings: List[np.ndarray], known_names: List[str], mark: bool = False) -> pd.DataFrame:
    """Run recognition on an image and optionally mark attendance. Returns a DataFrame of results.

    Output DataFrame columns: filename, true_name (None), pred_name, distance
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert BGR (OpenCV) to RGB and ensure memory is contiguous for dlib
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    locations = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, locations)

    results = []
    for loc, enc in zip(locations, encodings):
        distances = face_recognition.face_distance(known_encodings, enc) if len(known_encodings) > 0 else np.array([])
        if distances.size > 0:
            idx = np.argmin(distances)
            dist = float(distances[idx])
            if dist <= FACE_MATCH_THRESHOLD:
                name = known_names[idx]
            else:
                name = 'Unknown'
        else:
            name = 'Unknown'
            dist = None

        results.append({'filename': os.path.basename(image_path), 'pred_name': name, 'distance': dist})
        if mark and name != 'Unknown':
            mark_attendance(name)

    return pd.DataFrame(results)


# ---------------------------
# 5) Real-time recognition (webcam)
# ---------------------------

def recognize_faces_from_video(known_encodings: List[np.ndarray], known_names: List[str], attendance_file: str = ATTENDANCE_CSV, camera_index: int = CAMERA_INDEX):
    """Open webcam and recognize faces in real-time. Press 'q' to quit.

    This function draws boxes and names on the live feed and marks attendance.
    """
    video_capture = cv2.VideoCapture(camera_index)
    if not video_capture.isOpened():
        print('[ERROR] Could not open video source.')
        return

    print('[INFO] Starting video stream. Press q to quit.')

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Resize for speed
        small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        # Convert to RGB and ensure contiguity
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_small = np.ascontiguousarray(rgb_small)

        # Detect faces and compute encodings
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare to known encodings
            if len(known_encodings) > 0:
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_idx = np.argmin(distances)
                best_distance = float(distances[best_idx])
                if best_distance <= FACE_MATCH_THRESHOLD:
                    name = known_names[best_idx]
                else:
                    name = 'Unknown'
            else:
                name = 'Unknown'
                best_distance = None

            # Scale coordinates back to original frame size
            top = int(top / SCALE_FACTOR)
            right = int(right / SCALE_FACTOR)
            bottom = int(bottom / SCALE_FACTOR)
            left = int(left / SCALE_FACTOR)

            # Draw box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1)

            # Mark attendance for recognized faces
            if name != 'Unknown':
                # Append to CSV (skips duplicates for the same day by default)
                mark_attendance(name, attendance_file)

        cv2.imshow('Smart Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


# ---------------------------
# 6) Recognize from a folder of images (batch)
# ---------------------------

def recognize_folder(folder_path: str, known_encodings: List[np.ndarray], known_names: List[str], mark: bool = False) -> pd.DataFrame:
    """Run recognition on all images in folder_path and return a combined DataFrame of results."""
    rows = []
    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(folder_path, filename)
        df = recognize_faces_in_image(img_path, known_encodings, known_names, mark=mark)
        if not df.empty:
            rows.append(df)
    if len(rows) == 0:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# ---------------------------
# 7) Example main (script entrypoint)
# ---------------------------

if __name__ == '__main__':
    # 1. Load known faces
    try:
        known_encodings, known_names = load_and_encode_faces(KNOWN_FACES_DIR)
    except FileNotFoundError as e:
        print(str(e))
        print('Create the known_faces/ directory with subfolders per person and run again.')
        raise SystemExit(1)

    # 2. Uncomment one of the following to test

    # A) Real-time webcam attendance (press q to quit)
    # recognize_faces_from_video(known_encodings, known_names, attendance_file=ATTENDANCE_CSV)

    # B) Recognize a single test image and show predictions
    # sample_img = 'test_images/test1.jpg'
    # print(recognize_faces_in_image(sample_img, known_encodings, known_names, mark=False))

    # C) Batch recognize a folder of test images and optionally mark attendance
    # results_df = recognize_folder('test_images', known_encodings, known_names, mark=False)
    # print(results_df)

    print('\n[INFO] Script loaded. Uncomment desired action in the __main__ block to run tests or webcam.')

# ---------------------------
# Notes, improvements & teaching tips
# ---------------------------
# - Face encoding storage: for many users, store encodings in a serialized file (pickle, npz) to avoid recomputing each run.
# - Database: replace CSV with SQLite or a server DB (MySQL/Postgres) for multi-client systems.
# - Liveness detection: integrate anti-spoofing (eye-blink detection, texture checks) to avoid photo attacks.
# - Threshold tuning: adjust FACE_MATCH_THRESHOLD (0.4-0.6 typical). Lower reduces false accepts, higher reduces false rejects.
# - Performance: use GPU accelerated inference or smaller frame scales for Raspberry Pi / Jetson deployments. Consider using pretrained face detectors (MTCNN/RetinaFace) for robust detection.
# - GUI: build a Flask or Streamlit dashboard to view attendance logs in real-time.
# - Multi-person & tracking: If multiple faces appear in a frame, consider tracking face IDs to avoid marking the same person repeatedly within a short time window.

# End of file
