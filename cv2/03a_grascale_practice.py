
"""Color Space Conversion Practice (Beginner)

This script demonstrates many common OpenCV color conversions.
Each conversion opens its own window so you can visually compare.

Close all windows: focus one window and press any key.

Conversions covered:
    - BGR <-> GRAY
    - BGR <-> RGB
    - BGR <-> HSV
    - BGR <-> LAB
    - BGR <-> YCrCb
    - BGR <-> HLS
    - BGR <-> XYZ
    - BGR <-> LUV
    - BGR <-> YUV
    - BGR <-> BGRA (alpha channel)
    - GRAY <-> BGRA
"""

import cv2

IMG_PATH = 'images/image2.png'
img = cv2.imread(IMG_PATH)
if img is None:
        raise FileNotFoundError(f'Image not found at {IMG_PATH}')

# --- Basic: BGR <-> GRAY ---
bgr2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2bgr = cv2.cvtColor(bgr2gray, cv2.COLOR_GRAY2BGR)

# --- BGR <-> RGB (channel order swap) ---
bgr2rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb2bgr = cv2.cvtColor(bgr2rgb, cv2.COLOR_RGB2BGR)

# --- BGR <-> HSV ---
bgr2hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv2bgr = cv2.cvtColor(bgr2hsv, cv2.COLOR_HSV2BGR)

# --- BGR <-> LAB ---
bgr2lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
lab2bgr = cv2.cvtColor(bgr2lab, cv2.COLOR_LAB2BGR)

# --- BGR <-> YCrCb ---
bgr2ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
ycrcb2bgr = cv2.cvtColor(bgr2ycrcb, cv2.COLOR_YCrCb2BGR)

# --- BGR <-> HLS ---
bgr2hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
hls2bgr = cv2.cvtColor(bgr2hls, cv2.COLOR_HLS2BGR)

# --- BGR <-> XYZ ---
bgr2xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
xyz2bgr = cv2.cvtColor(bgr2xyz, cv2.COLOR_XYZ2BGR)

# --- BGR <-> LUV ---
bgr2luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
luv2bgr = cv2.cvtColor(bgr2luv, cv2.COLOR_LUV2BGR)

# --- BGR <-> YUV ---
bgr2yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
yuv2bgr = cv2.cvtColor(bgr2yuv, cv2.COLOR_YUV2BGR)

# --- Alpha channel related ---
bgr2bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # add alpha (fully opaque)
bgra2bgr = cv2.cvtColor(bgr2bgra, cv2.COLOR_BGRA2BGR)
bgra2gray = cv2.cvtColor(bgr2bgra, cv2.COLOR_BGRA2GRAY)
gray2bgra = cv2.cvtColor(bgr2gray, cv2.COLOR_GRAY2BGRA)

# Display originals & conversions
cv2.imshow('original_bgr', img)
cv2.imshow('bgr2gray', bgr2gray)
cv2.imshow('gray2bgr', gray2bgr)
cv2.imshow('bgr2rgb', bgr2rgb)
cv2.imshow('rgb2bgr', rgb2bgr)
cv2.imshow('bgr2hsv', bgr2hsv)
cv2.imshow('hsv2bgr', hsv2bgr)
cv2.imshow('bgr2lab', bgr2lab)
cv2.imshow('lab2bgr', lab2bgr)
cv2.imshow('bgr2ycrcb', bgr2ycrcb)
cv2.imshow('ycrcb2bgr', ycrcb2bgr)
cv2.imshow('bgr2hls', bgr2hls)
cv2.imshow('hls2bgr', hls2bgr)
cv2.imshow('bgr2xyz', bgr2xyz)
cv2.imshow('xyz2bgr', xyz2bgr)
cv2.imshow('bgr2luv', bgr2luv)
cv2.imshow('luv2bgr', luv2bgr)
cv2.imshow('bgr2yuv', bgr2yuv)
cv2.imshow('yuv2bgr', yuv2bgr)
cv2.imshow('bgr2bgra', bgr2bgra)
cv2.imshow('bgra2bgr', bgra2bgr)
cv2.imshow('bgra2gray', bgra2gray)
cv2.imshow('gray2bgra', gray2bgra)

print('Opened windows for all conversions. Press any key in an image window to close.')
cv2.waitKey(0)
cv2.destroyAllWindows()

