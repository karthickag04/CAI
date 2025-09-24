"""Basic Image Processing
Includes:
- Grayscale conversion
- Resize
- Flip
- Rotate
- Threshold (global & adaptive)
- Histogram Equalization

Run: python 03_basic_processing.py
"""
import cv2

IMG_PATH = 'images/image2.png'
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError('Image not found')

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Resize (width=300, height=300)
resized = cv2.resize(img, (300, 300))
# Flips
flip_h = cv2.flip(img, 1)
flip_v = cv2.flip(img, 0)
# Rotate 90 deg clockwise
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
# Thresholding
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
# Histogram Equalization
equalized = cv2.equalizeHist(gray)

cv2.imshow('Original', img)
cv2.imshow('Gray', gray)
cv2.imshow('Resized', resized)
# cv2.imshow('Flip Horizontal', flip_h)
# cv2.imshow('Flip Vertical', flip_v)
# cv2.imshow('Rotated', rotated)
# cv2.imshow('Threshold', thresh)
# cv2.imshow('Adaptive', adaptive)
# cv2.imshow('Equalized', equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
