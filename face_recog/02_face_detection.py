"""
=============================================================================
LESSON 2: Face Detection with `face_recognition.face_locations()`
=============================================================================

PURPOSE:
- Find the bounding box coordinates (top, right, bottom, left) for all faces in an image.
- Draw bounding box rectangles around detected faces and save the annotated image.

API METHOD COVERED:
- face_recognition.face_locations(image, model="hog")
  Note: model can be "hog" (CPU, fast) or "cnn" (GPU, more accurate). Default is "hog".
"""

import face_recognition
from PIL import Image, ImageDraw

def main():
    image_path = "images/group.jpg"
    print(f"Loading image from: {image_path} ...")
    image = face_recognition.load_image_file(image_path)

    # Step 1: Detect face locations
    # Returns a list of tuples formatted as: (top, right, bottom, left)
    print("Finding face locations...")
    face_locations = face_recognition.face_locations(image, model="hog")

    print(f"\nFound {len(face_locations)} face(s) in the image!")

    # Step 2: Convert NumPy array to PIL Image for drawing
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Step 3: Loop through each detected face coordinate
    for index, face_location in enumerate(face_locations, start=1):
        top, right, bottom, left = face_location
        print(f"  Face #{index}: Top={top}, Right={right}, Bottom={bottom}, Left={left}")

        # Draw a red rectangle box around the face (width = 3 pixels)
        draw.rectangle([left, top, right, bottom], outline="red", width=3)

    # Step 4: Save the output image with drawn bounding boxes
    output_filename = "output_faces_detected.jpg"
    pil_image.save(output_filename)
    print(f"\nSaved annotated image to: {output_filename}")

if __name__ == "__main__":
    main()
