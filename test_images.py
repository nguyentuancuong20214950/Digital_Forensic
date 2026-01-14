"""
Test với ảnh khác có ít pixels boundary
"""
import cv2
import numpy as np

# Test nhiều ảnh
for img_name in ["10.pgm", "100.pgm", "1000.pgm"]:
    cover = cv2.imread(f"data/input/BOSSbase_256/{img_name}", cv2.IMREAD_GRAYSCALE)
    if cover is not None:
        print(f"\n{img_name}:")
        print(f"  Min: {cover.min()}, Max: {cover.max()}, Mean: {cover.mean():.1f}")
        
        # Count boundary pixels
        boundary = np.sum((cover <= 5) | (cover >= 250))
        print(f"  Boundary pixels (<=5 or >=250): {boundary} / {cover.size} ({100*boundary/cover.size:.1f}%)")
        
        # First 20 pixels
        print(f"  First 20 pixels: {cover.flatten()[:20]}")
