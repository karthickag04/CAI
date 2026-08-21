"""
=============================================================================
LESSON 7: Face Comparison and Distance Calculation
=============================================================================

PURPOSE:
- Compute numerical distance between face embeddings (`face_distance`).
- Turn numerical distance into Boolean matches based on a tolerance threshold (`compare_faces`).

API METHODS COVERED:
- face_recognition.face_distance(known_face_encodings, face_to_check)
  Returns an array of Euclidean distances. Smaller distance = higher similarity.
- face_recognition.compare_faces(known_face_encodings, face_to_check, tolerance=0.6)
  Returns a list of True/False booleans. Default tolerance is 0.6.
"""

import face_recognition

def main():
    print("Loading sample images...")
    img_person_a1 = face_recognition.load_image_file("images/person_a.jpg")
    img_person_a2 = face_recognition.load_image_file("images/person_a_2.jpg")
    img_person_b = face_recognition.load_image_file("images/person_b.jpg")

    print("Generating encodings for all faces...")
    encoding_a1 = face_recognition.face_encodings(img_person_a1)[0]
    encoding_a2 = face_recognition.face_encodings(img_person_a2)[0]
    encoding_b  = face_recognition.face_encodings(img_person_b)[0]

    # List of known encodings (e.g. Alice Photo 1 and Bob Photo)
    known_encodings = [encoding_a1, encoding_b]
    known_names = ["Person A (Alice)", "Person B (Bob)"]

    print("\n==================================================")
    print("TEST 1: Comparing Person A's second photo (Alice) against database")
    print("==================================================")

    # 1. Compute numerical distances
    distances_test1 = face_recognition.face_distance(known_encodings, encoding_a2)
    
    # 2. Compare faces with default tolerance = 0.6
    matches_test1 = face_recognition.compare_faces(known_encodings, encoding_a2, tolerance=0.6)

    for name, distance, match in zip(known_names, distances_test1, matches_test1):
        print(f"  Vs {name:20s}: Distance = {distance:.4f} | Match (tolerance=0.6) = {match}")

    print("\n==================================================")
    print("TEST 2: Understanding Tolerance Thresholds")
    print("==================================================")
    print("Distance between Person A (photo 1) and Person A (photo 2):", f"{distances_test1[0]:.4f}")
    print("Distance between Person A (photo 1) and Person B (photo):  ", f"{distances_test1[1]:.4f}")

    print("\nChecking Person A (photo 2) with strict vs loose tolerance:")
    for tol in [0.4, 0.5, 0.6, 0.7]:
        is_match = face_recognition.compare_faces([encoding_a1], encoding_a2, tolerance=tol)[0]
        print(f"  Tolerance = {tol:.1f} -> Match with Person A? {is_match}")

if __name__ == "__main__":
    main()
