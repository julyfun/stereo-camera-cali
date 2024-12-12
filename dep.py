import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# Load calibration results
calib_data = np.load('stereo_calib.npz')
mtx_l = calib_data['mtx_l']
dist_l = calib_data['dist_l']
mtx_r = calib_data['mtx_r']
dist_r = calib_data['dist_r']
R = calib_data['R']
T = calib_data['T']
R1 = calib_data['R1']
R2 = calib_data['R2']
P1 = calib_data['P1']
P2 = calib_data['P2']
Q = calib_data['Q']
left_map1 = calib_data['left_map1']
left_map2 = calib_data['left_map2']
right_map1 = calib_data['right_map1']
right_map2 = calib_data['right_map2']

# Read images
images_left = sorted(glob.glob('left/*.jpg'))
images_right = sorted(glob.glob('right/*.jpg'))

# Process each pair of images
for img_left, img_right in zip(images_left, images_right):
    img_l = cv2.imread(img_left)
    img_r = cv2.imread(img_right)

    # Rectify images
    rectified_l = cv2.remap(img_l, left_map1, left_map2, cv2.INTER_LINEAR)
    rectified_r = cv2.remap(img_r, right_map1, right_map2, cv2.INTER_LINEAR)

    # Convert to grayscale
    gray_l = cv2.cvtColor(rectified_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rectified_r, cv2.COLOR_BGR2GRAY)

    # Feature detection in center region of the left image
    sift = cv2.SIFT_create()
    center_x, center_y = gray_l.shape[1] // 2, gray_l.shape[0] // 2
    center_region = gray_l[center_y - 50:center_y + 50, center_x - 50:center_x + 50]
    kp_l, des_l = sift.detectAndCompute(center_region, None)

    # Adjust keypoints coordinates to the full image
    for kp in kp_l:
        kp.pt = (kp.pt[0] + center_x - 50, kp.pt[1] + center_y - 50)

    # Feature detection in the full right image
    kp_r, des_r = sift.detectAndCompute(gray_r, None)

    if des_l is None or des_r is None:
        print(f"No features found for {img_left} and {img_right}")
        continue

    # Feature matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_l, des_r, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) == 0:
        print(f"No good matches found for {img_left} and {img_right}")
        continue

    # Extract matched points
    pts_l = np.float32([kp_l[m.queryIdx].pt for m in good_matches])
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in good_matches])

    # Triangulate points
    points_4d = cv2.triangulatePoints(P1, P2, pts_l.T, pts_r.T)
    points_4d /= points_4d[3]  # Convert from homogeneous coordinates

    # Calculate average depth (z-coordinate)
    average_depth = np.mean(points_4d[2])

    print(f"Average depth for {img_left} and {img_right}: {average_depth:.2f} meters")

    # Draw matches
    img_matches = cv2.drawMatches(rectified_l, kp_l, rectified_r, kp_r, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Plot matches
    plt.figure(figsize=(20, 10))
    plt.title(f"Feature Matches for {img_left} and {img_right}")
    plt.imshow(img_matches)
    plt.show()
