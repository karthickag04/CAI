"""Image I/O Basics
- Load an image
- Display it
- Save a copy

Run: python 01_image_io.py
Press any key on an image window to close.
"""
import cv2

IMG_PATH = 'images/image2.png'  # adjust if needed

img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f'Could not load image at {IMG_PATH}')

cv2.imshow('Original', img)
cv2.imwrite('images/saved_copy.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
