import cv2
import numpy as np

cover = cv2.imread("data/input/BOSSbase_256/1.pgm", cv2.IMREAD_GRAYSCALE)
print(f"Cover shape: {cover.shape}")
print(f"Min pixel: {cover.min()}")
print(f"Max pixel: {cover.max()}")
print(f"Mean pixel: {cover.mean():.2f}")
print(f"\nFirst 10 pixels: {cover.flatten()[:10]}")
print(f"Histogram (first 20 values):")
for i in range(20):
    count = np.sum(cover == i)
    if count > 0:
        print(f"  Value {i}: {count} pixels")
