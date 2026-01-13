"""
Debug PVD - check if embed actually modifies stego
"""
import cv2
import numpy as np
import random
from core.pvd import PVD

cover = cv2.imread('data/input/BOSSbase_256/1.pgm', cv2.IMREAD_GRAYSCALE)
print(f"Cover[0,0:10]: {cover[0, :10]}")

message = "AB"
key = 12345

stego, _ = PVD.embed(cover, message, key)
print(f"Stego[0,0:10]: {stego[0, :10]}")

print(f"\nAre they different? {not np.array_equal(cover, stego)}")
print(f"Num different pixels: {np.sum(cover != stego)}")

# Check specific coordinate from random shuffle
rng = random.Random(key)
h, w = cover.shape
coords = [(r, c) for r in range(h) for c in range(w-1)]
rng.shuffle(coords)
r, c = coords[0]
print(f"\nFirst coordinate: ({r}, {c})")
print(f"Cover[{r},{c}:{c+2}]: {cover[r, c:c+2]}")
print(f"Stego[{r},{c}:{c+2}]: {stego[r, c:c+2]}")
