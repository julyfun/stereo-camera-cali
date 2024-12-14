# %%
from transformers import pipeline
from PIL import Image
import glob
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Initialize the depth estimation pipeline
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)  # Use device 0 for GPU

# %%
# Load stereo calibration parameters
calib_data = np.load('stereo_calib.npz')
Q = calib_data['Q']
print(Q[2, 3], Q[3, 2])
focal = Q[2, 3]
baseline = -1 / Q[3, 2]
C = focal * baseline
print(f'C: {C}')

# Load and display the first image's depth map
imgs = glob.glob('right/*.jpg')
if imgs:
    first_img = imgs[0]
    image = Image.open(first_img)
    st = time.time()
    predictions = pipe(image)
    print(f'{first_img} time: {time.time()-st}')

    # Get the depth map from predictions
    depth_map = predictions["depth"]

    # Convert depth map (PIL Image) to a NumPy array
    depth_map_np = np.array(depth_map)

    # Convert depth map to disparity map (assuming depth is inversely proportional to distance)
    # disparity_map = 1.0 / (depth_map_np + 1e-6)  # Add a small value to avoid division by zero

    # Display the distance map
    plt.imshow(np.minimum(C / depth_map_np, 5), cmap='viridis')
    plt.title(f'Distance Map for {first_img}')
    plt.axis('off')
    plt.colorbar(label='Distance (meters)')
    plt.show()
