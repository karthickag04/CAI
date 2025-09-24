"""Segmentation (Watershed)
- Threshold -> Noise removal -> Distance transform -> Watershed

Run: python 10_segmentation_watershed.py
"""
import cv2
import numpy as np

IMG_PATH = 'images/image2.png'
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError('Image not found')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(opening, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0
markers = cv2.watershed(img, markers)
img[markers == -1] = [0,0,255]

cv2.imshow('Watershed', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
