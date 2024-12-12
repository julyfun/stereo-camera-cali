import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob

print(f'ok')

# 棋盘格尺寸
chessboard_size = (11, 8)
# 棋盘格每个方格的大小（例如，20mm）
square_size = 45 / 1000.0

# 准备对象点，例如 (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 用于存储所有图像的对象点和图像点
objpoints = []  # 3d point in real world space
imgpoints_left = []  # 2d points in image plane for left camera
imgpoints_right = []  # 2d points in image plane for right camera

# 读取图像
images_left = glob.glob('left/*.jpg')
images_right = glob.glob('right/*.jpg')

for img_left, img_right in zip(images_left, images_right):
    img_l = cv2.imread(img_left)
    img_r = cv2.imread(img_right)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)


    # 寻找左右图像中的棋盘格角点
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, None)

    # 如果找到足够的角点，将其添加到对象点和图像点列表中
    if ret_l and ret_r:
        print('enough corner')
        objpoints.append(objp)
        imgpoints_left.append(corners_l)
        imgpoints_right.append(corners_r)
    else:
        print('not enough corner')

# 标定左相机
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_left, gray_l.shape[::-1], None, None)

# 标定右相机
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_right, gray_r.shape[::-1], None, None)

# 双目标定
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_l, dist_l, mtx_r, dist_r,
    gray_l.shape[::-1], criteria=criteria, flags=flags)

# 立体校正
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_l, dist_l, mtx_r, dist_r, gray_l.shape[::-1], R, T, alpha=0)

# 计算校正映射
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    mtx_l, dist_l, R1, P1, gray_l.shape[::-1], cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    mtx_r, dist_r, R2, P2, gray_r.shape[::-1], cv2.CV_16SC2)

# 保存标定结果
np.savez('stereo_calib.npz', mtx_l=mtx_l, dist_l=dist_l, mtx_r=mtx_r, dist_r=dist_r, R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, left_map1=left_map1, left_map2=left_map2, right_map1=right_map1, right_map2=right_map2)

print("双目标定完成并保存到文件 'stereo_calib.npz'")
