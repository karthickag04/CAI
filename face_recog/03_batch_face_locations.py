"""
=============================================================================
LESSON 3: Batch Face Detection with `face_recognition.batch_face_locations()`
=============================================================================

PURPOSE:
- Detect faces across multiple images at once in a single batch call.
- Primarily used when processing videos or large batches of images efficiently.

API METHOD COVERED:
- face_recognition.batch_face_locations([image1, image2, image3], number_of_times_to_upsample=1)
"""

import face_recognition

def main():
    image_paths = [
        "images/person_a.jpg",
        "images/person_b.jpg",
        "images/group.jpg"
    ]

    print("Loading and resizing images for batch processing...")
    # NOTE: batch_face_locations requires ALL images in the batch list to have identical dimensions!
    # We resize to (400, 400) to keep CPU batch detection fast.
    target_size = (400, 400)
    
    from PIL import Image
    import numpy as np
    
    # Load and resize all images to standardized 400x400 arrays
    images = [np.array(Image.open(p).convert('RGB').resize(target_size)) for p in image_paths]

    # Step 1: Detect face locations for all images at once
    # Returns a list of face location lists: [ [locs_img1], [locs_img2], [locs_img3] ]
    print("\nRunning batch_face_locations()...")
    batch_results = face_recognition.batch_face_locations(images)

    # Step 2: Iterate over the batch results
    print("\n--- Batch Results Summary ---")
    for path, face_locations in zip(image_paths, batch_results):
        print(f"Image '{path}': Found {len(face_locations)} face(s)")
        for i, (top, right, bottom, left) in enumerate(face_locations, start=1):
            print(f"   Face #{i} at Top={top}, Right={right}, Bottom={bottom}, Left={left}")

if __name__ == "__main__":
    main()
