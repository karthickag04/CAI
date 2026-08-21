"""
=============================================================================
LESSON 5: Generating Face Encodings / Embeddings (`face_encodings()`)
=============================================================================

PURPOSE:
- Convert a face image into a numerical 128-dimensional embedding vector.
- This 128-number vector represents the unique biometric signature of a face.

API METHOD COVERED:
- face_recognition.face_encodings(image, known_face_locations=None, num_jitters=1, model="small")
  Returns a list of 128-element 1D NumPy arrays (one for each detected face).
"""

import face_recognition

def main():
    image_path = "images/person_a.jpg"
    print(f"Loading image from: {image_path} ...")
    image = face_recognition.load_image_file(image_path)

    # Step 1: Generate face encodings
    print("Generating 128-dimensional face encodings...")
    encodings = face_recognition.face_encodings(image)

    print(f"\nGenerated encodings for {len(encodings)} face(s).")

    if encodings:
        face_encoding = encodings[0]

        print("\n--- Face Encoding Inspection ---")
        print(f"Data type: {type(face_encoding)}")
        print(f"Vector shape: {face_encoding.shape}  (128 measurements)")
        print(f"Data type of values: {face_encoding.dtype}")

        print("\nFirst 10 numerical values of the 128D embedding vector:")
        for i, val in enumerate(face_encoding[:10], start=1):
            print(f"  Dimension #{i:02d}: {val:+.6f}")

        print("\nSummary: Person A's face is now represented by 128 unique numbers!")

if __name__ == "__main__":
    main()
