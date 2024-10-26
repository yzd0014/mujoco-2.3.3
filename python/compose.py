import cv2
import numpy as np

# List of frames to overlay in order
frame_num = 20
frames = [cv2.imread(f"frames/frame_{frame:04d}.jpg") for frame in range(0, frame_num)]

# Initialize base image with the first frame
trailing_image = frames[0].astype(float)  # Convert to float for accumulation

# Define opacity factors (these could be tuned to your preference)

opacities = [None] * frame_num
for i in range(frame_num):
    opacities[i] = 1 - i / frame_num

# Overlay each frame with decreasing opacity
for i, frame in enumerate(frames[1:], start=1):
    trailing_image = cv2.addWeighted(trailing_image, 1, frame.astype(float), 0.4, 0)

# Convert back to 8-bit and save the final trailing effect image
trailing_image = np.clip(trailing_image, 0, 255).astype(np.uint8)
cv2.imwrite("frames/trailing_effect.jpg", trailing_image)
