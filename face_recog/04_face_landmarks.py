"""
=============================================================================
LESSON 4: Extracting Facial Landmarks with `face_recognition.face_landmarks()`
=============================================================================

PURPOSE:
- Detect specific facial features (eyes, eyebrows, nose, mouth, chin outline).
- Draw lines connecting facial landmark points and save the visualization.

API METHOD COVERED:
- face_recognition.face_landmarks(image, face_locations=None, model="large")
  Returns a list of dicts, where each dict maps feature names to (x, y) coordinate lists.

FEATURE KEYS RETURNED:
  - chin
  - left_eyebrow, right_eyebrow
  - nose_bridge, nose_tip
  - left_eye, right_eye
  - top_lip, bottom_lip
"""

import face_recognition
from PIL import Image, ImageDraw

def main():
    image_path = "images/person_a.jpg"
    print(f"Loading image from: {image_path} ...")
    image = face_recognition.load_image_file(image_path)

    # Step 1: Detect facial landmarks
    print("Extracting facial landmarks...")
    landmarks_list = face_recognition.face_landmarks(image)

    print(f"\nFound facial landmarks for {len(landmarks_list)} face(s).")

    # Step 2: Convert to Pillow image for drawing landmark features
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Colors for different facial features
    feature_colors = {
        "chin": "cyan",
        "left_eyebrow": "green",
        "right_eyebrow": "green",
        "nose_bridge": "yellow",
        "nose_tip": "orange",
        "left_eye": "red",
        "right_eye": "red",
        "top_lip": "magenta",
        "bottom_lip": "pink"
    }

    # Step 3: Loop through each face and its landmarks dictionary
    for i, face_landmarks in enumerate(landmarks_list, start=1):
        print(f"\n--- Landmarks for Face #{i} ---")
        for facial_feature, points in face_landmarks.items():
            print(f"  Feature '{facial_feature}': {len(points)} point(s)")

            color = feature_colors.get(facial_feature, "white")

            # Draw line connecting the landmark coordinates
            draw.line(points, fill=color, width=3)

            # Draw small dots at each coordinate point
            for point in points:
                x, y = point
                draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill="white")

    # Step 4: Save visualization
    output_filename = "output_landmarks.jpg"
    pil_image.save(output_filename)
    print(f"\nSaved landmark visualization to: {output_filename}")

if __name__ == "__main__":
    main()
