import os
import sys
from smart_attendance_system import (
    KNOWN_FACES_DIR,
    load_and_encode_faces,
    recognize_faces_in_image,
)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_image.py <path_to_image>")
        raise SystemExit(2)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        raise SystemExit(f"Image not found: {image_path}")

    if not os.path.exists(KNOWN_FACES_DIR):
        raise SystemExit(f"Known faces directory not found: {KNOWN_FACES_DIR}. Create it and add subfolders per person.")

    encodings, names = load_and_encode_faces(KNOWN_FACES_DIR)
    df = recognize_faces_in_image(image_path, encodings, names, mark=False)
    if df.empty:
        print("No faces detected.")
    else:
        print(df.to_string(index=False))
