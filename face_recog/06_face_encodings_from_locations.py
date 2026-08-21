"""
=============================================================================
LESSON 6: Encoding Faces from Known Locations
=============================================================================

PURPOSE:
- Pass pre-detected bounding boxes to `face_encodings()` to avoid re-detecting faces.
- Speeds up processing pipelines when locations are already computed.

API METHOD COVERED:
- face_recognition.face_encodings(image, known_face_locations=locations)
"""

import face_recognition

def main():
    image_path = "images/group.jpg"
    print(f"Loading image: {image_path} ...")
    image = face_recognition.load_image_file(image_path)

    # Step 1: Detect face locations first
    print("Step 1: Finding face locations...")
    locations = face_recognition.face_locations(image)
    print(f"Found {len(locations)} face location(s): {locations}")

    # Step 2: Generate encodings by passing the pre-detected locations
    print("\nStep 2: Generating encodings using known locations...")
    encodings = face_recognition.face_encodings(image, known_face_locations=locations)

    print(f"Successfully generated {len(encodings)} encoding(s)!")

    for i, (loc, enc) in enumerate(zip(locations, encodings), start=1):
        print(f"\nFace #{i}:")
        print(f"  Location coordinates: Top={loc[0]}, Right={loc[1]}, Bottom={loc[2]}, Left={loc[3]}")
        print(f"  Encoding shape: {enc.shape}")
        print(f"  First 3 vector values: {enc[:3]}")

if __name__ == "__main__":
    main()
