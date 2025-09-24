"""Feature Detection & Matching (ORB)
- Detect keypoints & descriptors in two images
- Match with Brute Force matcher

Run: python 06_feature_matching_orb.py
Provide two images; here we reuse image2.png twice for demo if second missing.
"""
import cv2

IMG1_PATH = 'images/image2.png'
IMG2_PATH = 'images/image3.PNG'

img1 = cv2.imread(IMG1_PATH, 0)
img2 = cv2.imread(IMG2_PATH, 0)
if img1 is None or img2 is None:
    raise FileNotFoundError('One of the images not found')

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None)
cv2.imshow('ORB Matches', match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
