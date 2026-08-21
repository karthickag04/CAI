"""
=============================================================================
LESSON 1: Loading Images with `face_recognition.load_image_file()`
=============================================================================

PURPOSE:
- Learn how to load an image file into Python using face_recognition.
- Understand how images are represented in memory as NumPy arrays.

API METHOD COVERED:
- face_recognition.load_image_file("path_to_image.jpg")
"""

import face_recognition
import cv2

def main():
    image_path = "images/person_a.jpg"
    print(f"Loading image from: {image_path} ...")

    # Step 1: Load the image file into a NumPy array
    # The image is automatically converted to RGB format.
    image = face_recognition.load_image_file(image_path)

    # Step 2: Inspect the loaded image property
    print("\n--- Image Information ---")
    print(f"Type of loaded object: {type(image)}")
    print(f"Image shape (Height, Width, Color Channels): {image.shape}")
    print(f"Data type of pixels: {image.dtype}")

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Extract dimensions
    height, width, channels = image.shape
    print(f"Summary: Image is {width} pixels wide and {height} pixels high with {channels} color channels (RGB).")

    # Inspect sample pixel value at coordinates (y=100, x=100)
    sample_pixel = image[100, 100]
    print(f"Sample RGB pixel at (100, 100): Red={sample_pixel[0]}, Green={sample_pixel[1]}, Blue={sample_pixel[2]}")

if __name__ == "__main__":
    main()
