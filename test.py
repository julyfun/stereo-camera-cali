import numpy as np

# Load calibration data
calib_data = np.load('stereo_calib.npz')
T = calib_data['T']

# Calculate the baseline (distance between the two cameras)
baseline = np.linalg.norm(T)

print(f"Estimated distance (baseline) between the two cameras: {baseline} meters")
