import cv2
import numpy as np
import glob

# Chessboard dimensions
chessboard_size = (11, 8)
square_size = 45 / 1000.0  # in meters

# Prepare object points
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Storage for object points and image points
objpoints = []
imgpoints_left = []
imgpoints_right = []

# Read images
images_left = glob.glob('left/*.jpg')
images_right = glob.glob('right/*.jpg')

# Collect points
for img_left, img_right in zip(images_left, images_right):
    img_l = cv2.imread(img_left)
    img_r = cv2.imread(img_right)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, None)

    if ret_l and ret_r:
        objpoints.append(objp)
        imgpoints_left.append(corners_l)
        imgpoints_right.append(corners_r)
    else:
        print('Not enough corners found')

# Calibrate left camera
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
    objpoints, imgpoints_left, gray_l.shape[::-1], None, None)

# Calibrate right camera
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
    objpoints, imgpoints_right, gray_r.shape[::-1], None, None)

# Display intrinsic parameters and distortion coefficients
print("Left Camera Intrinsic Matrix:\n", mtx_l)
print("Left Camera Distortion Coefficients:\n", dist_l)
print("Right Camera Intrinsic Matrix:\n", mtx_r)
print("Right Camera Distortion Coefficients:\n", dist_r)

# Calculate reprojection error for each image
def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    errors = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        errors.append(error)
        total_error += error
    mean_error = total_error / len(objpoints)
    return mean_error, errors

mean_error_l, errors_l = calculate_reprojection_error(objpoints, imgpoints_left, rvecs_l, tvecs_l, mtx_l, dist_l)
mean_error_r, errors_r = calculate_reprojection_error(objpoints, imgpoints_right, rvecs_r, tvecs_r, mtx_r, dist_r)

print(f"Mean Reprojection Error (Left): {mean_error_l}")
print(f"Mean Reprojection Error (Right): {mean_error_r}")

for i, (err_l, err_r) in enumerate(zip(errors_l, errors_r)):
    print(f"Image {i+1}: Left Error = {err_l}, Right Error = {err_r}")

# Stereo calibration
flags = cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_l, dist_l, mtx_r, dist_r,
    gray_l.shape[::-1], criteria=criteria, flags=flags)

# Stereo rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1], R, T, alpha=0)

# Compute rectification maps
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    mtx_l, dist_l, R1, P1, gray_l.shape[::-1], cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    mtx_r, dist_r, R2, P2, gray_r.shape[::-1], cv2.CV_16SC2)

# Save calibration results
np.savez('stereo_calib.npz', mtx_l=mtx_l, dist_l=dist_l, mtx_r=mtx_r, dist_r=dist_r, R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, left_map1=left_map1, left_map2=left_map2, right_map1=right_map1, right_map2=right_map2)

print("Stereo calibration completed and saved to 'stereo_calib.npz'")
