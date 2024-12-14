import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load stereo calibration results
calib_data = np.load('stereo_calib.npz')
mtx_l = calib_data['mtx_l']
dist_l = calib_data['dist_l']
mtx_r = calib_data['mtx_r']
dist_r = calib_data['dist_r']
R1 = calib_data['R1']
R2 = calib_data['R2']
P1 = calib_data['P1']
P2 = calib_data['P2']
left_map1 = calib_data['left_map1']
left_map2 = calib_data['left_map2']
right_map1 = calib_data['right_map1']
right_map2 = calib_data['right_map2']

# Read the first pair of stereo images
img_left = cv2.imread('left/0.jpg')
img_right = cv2.imread('right/0.jpg')

# Undistort and rectify images
rectified_left = cv2.remap(img_left, left_map1, left_map2, cv2.INTER_LINEAR)
rectified_right = cv2.remap(img_right, right_map1, right_map2, cv2.INTER_LINEAR)

# Convert to grayscale
gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)

# SGBM parameters
minDisparity = 0
numDisparities = 16 * 5  # Must be divisible by 16
blockSize = 11
P1 = 8 * 3 * blockSize**2
P2 = 32 * 3 * blockSize**2
disp12MaxDiff = 1
uniquenessRatio = 15
speckleWindowSize = 100
speckleRange = 32

# Create SGBM object
stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                               numDisparities=numDisparities,
                               blockSize=blockSize,
                               P1=P1,
                               P2=P2,
                               disp12MaxDiff=disp12MaxDiff,
                               uniquenessRatio=uniquenessRatio,
                               speckleWindowSize=speckleWindowSize,
                               speckleRange=speckleRange)

# Compute disparity maps
disparity_left = stereo.compute(gray_left, gray_right).astype(np.int16)
disparity_right = stereo.compute(gray_right, gray_left).astype(np.int16)

# Adjust disparity maps
disparity_left = np.abs(disparity_left)
disparity_right = np.abs(disparity_right)

# Set unreliable disparity values
disparity_left[disparity_left == -1] = 0
disparity_right[disparity_right == -1] = (numDisparities + 1) * 16

# Normalize disparity maps for visualization
disparity_left_vis = cv2.normalize(disparity_left, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_left_vis = np.uint8(disparity_left_vis)
disparity_right_vis = cv2.normalize(disparity_right, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_right_vis = np.uint8(disparity_right_vis)

# Display the disparity maps using Matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Left Disparity Map')
plt.imshow(disparity_left_vis, cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title('Right Disparity Map')
plt.imshow(disparity_right_vis, cmap='gray')
plt.colorbar()

plt.show()
