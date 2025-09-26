import os
from smart_attendance_system import (
    KNOWN_FACES_DIR,
    ATTENDANCE_CSV,
    load_and_encode_faces,
    recognize_faces_from_video,
)

if __name__ == "__main__":
    if not os.path.exists(KNOWN_FACES_DIR):
        raise SystemExit(f"Known faces directory not found: {KNOWN_FACES_DIR}. Create it and add subfolders per person.")

    encodings, names = load_and_encode_faces(KNOWN_FACES_DIR)
    recognize_faces_from_video(encodings, names, attendance_file=ATTENDANCE_CSV)
