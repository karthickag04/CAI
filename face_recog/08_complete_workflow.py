"""
=============================================================================
LESSON 8: Complete End-to-End Face Recognition Workflow
=============================================================================

PURPOSE:
- Combine all 5 core steps into a complete face identification system.

WORKFLOW PIPELINE:
  1. Load Known Images (`load_image_file`)
  2. Encode Known Faces (`face_encodings`)
  3. Load Target / Unknown Image (`load_image_file`)
  4. Find Target Face Locations (`face_locations`)
  5. Encode Target Faces (`face_encodings`)
  6. Compare & Identify (`face_distance` & `compare_faces`)
  7. Draw & Annotate Names on Output Image
"""

import face_recognition
from PIL import Image, ImageDraw, ImageFont

def main():
    print("==================================================")
    print("STEP 1 & 2: Building Known Faces Gallery")
    print("==================================================")

    # Dictionary of known people: Name -> Image File
    known_people_files = {
        "Alice": "images/person_a.jpg",
        "Bob": "images/person_b.jpg"
    }

    known_encodings = []
    known_names = []

    for name, file_path in known_people_files.items():
        print(f"  Encoding {name} from {file_path}...")
        img = face_recognition.load_image_file(file_path)
        encs = face_recognition.face_encodings(img)
        if encs:
            known_encodings.append(encs[0])
            known_names.append(name)
        else:
            print(f"  Warning: No face found in {file_path}")

    print(f"\nGallery initialized with {len(known_names)} known person(s): {known_names}")

    print("\n==================================================")
    print("STEP 3, 4 & 5: Processing Unknown Target Image")
    print("==================================================")
    
    target_image_path = "images/person_a_2.jpg"
    print(f"Loading target image for identification: {target_image_path}...")
    target_image = face_recognition.load_image_file(target_image_path)

    # Find locations and encodings for all faces in the target image
    target_locations = face_recognition.face_locations(target_image)
    target_encodings = face_recognition.face_encodings(target_image, known_face_locations=target_locations)

    print(f"Found {len(target_locations)} face(s) in target image.")

    print("\n==================================================")
    print("STEP 6 & 7: Matching and Annotating Output Image")
    print("==================================================")

    pil_image = Image.fromarray(target_image)
    draw = ImageDraw.Draw(pil_image)

    # Process each face detected in the target image
    for i, (location, encoding) in enumerate(zip(target_locations, target_encodings), start=1):
        top, right, bottom, left = location
        name = "Unknown"

        # Calculate distances against all known encodings in gallery
        distances = face_recognition.face_distance(known_encodings, encoding)

        if len(distances) > 0:
            best_match_index = distances.argmin()
            best_distance = distances[best_match_index]

            # Match threshold (0.6 is default standard tolerance)
            if best_distance < 0.6:
                name = known_names[best_match_index]

            print(f"Face #{i}: Identified as '{name}' (Best Distance = {best_distance:.4f})")
        else:
            print(f"Face #{i}: No known faces in database to compare against.")

        # --- Draw Bounding Box & Label ---
        box_color = "lime" if name != "Unknown" else "red"
        draw.rectangle([left, top, right, bottom], outline=box_color, width=4)

        # Draw name label banner below the face
        label_text = f"{name}"
        text_bbox = draw.textbbox((left, bottom), label_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        draw.rectangle([left, bottom, left + text_width + 12, bottom + text_height + 10], fill=box_color)
        draw.text((left + 6, bottom + 5), label_text, fill="black")

    # Step 8: Save final annotated image
    output_filename = "output_final_recognition.jpg"
    pil_image.save(output_filename)
    print(f"\nSaved final annotated image to: {output_filename}")

if __name__ == "__main__":
    main()
