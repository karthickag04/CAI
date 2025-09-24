"""Geometric Transformations
- Affine transform
- Perspective transform
- Rotation

Run: python 09_geometric_transforms.py
"""
import cv2
import numpy as np

IMG_PATH = 'images/image2.png'
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError('Image not found')
rows, cols = img.shape[:2]

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M_affine = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(img, M_affine, (cols, rows))

# Perspective (using arbitrary points; adjust as needed)
pts1_p = np.float32([[10,10],[cols-10,10],[10,rows-10],[cols-10,rows-10]])
pts2_p = np.float32([[0,0],[300,0],[0,300],[300,300]])
M_persp = cv2.getPerspectiveTransform(pts1_p, pts2_p)
perspective = cv2.warpPerspective(img, M_persp, (300,300))

M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
rotated = cv2.warpAffine(img, M_rot, (cols, rows))

cv2.imshow('Affine', affine)
cv2.imshow('Perspective', perspective)
cv2.imshow('Rotated', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
