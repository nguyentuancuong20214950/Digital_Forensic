"""
Debug PVD - check Pair 0 specifically
"""
import cv2
import numpy as np
import random

cover = cv2.imread('data/input/BOSSbase_256/1.pgm', cv2.IMREAD_GRAYSCALE)
message = "AB"
key = 12345

# Get coordinate 0
rng = random.Random(key)
h, w = cover.shape
coords = [(r, c) for r in range(h) for c in range(w-1)]
rng.shuffle(coords)
r0, c0 = coords[0]

print(f"Pair 0 coordinate: ({r0}, {c0})")
p1, p2 = int(cover[r0, c0]), int(cover[r0, c0+1])
print(f"Cover pixels: p1={p1}, p2={p2}")

# Message to binary
data = message.encode('utf-8') + b'\x00\x00\x00'
bin_msg = ''.join([format(b, "08b") for b in data])
print(f"Binary: {bin_msg[:24]}")

# Simulate embed for Pair 0
d = abs(p1 - p2)
print(f"d = |{p1} - {p2}| = {d}")

ranges = [0, 8, 16, 32, 64, 128, 256]
for range_idx in range(len(ranges)-1):
    if ranges[range_idx] <= d < ranges[range_idx+1]:
        low, high = ranges[range_idx], ranges[range_idx+1]-1
        t = int(np.log2(high - low + 1))
        print(f"Range: [{low}, {high}], t={t}")
        
        bits = bin_msg[0:t]
        print(f"Bits to embed: '{bits}'")
        
        b = int(bits.ljust(t, '0'), 2)
        d_new = low + b
        diff = d_new - d
        print(f"b={b}, d_new={d_new}, diff={diff}")
        
        # Adjust pixels
        if p1 >= p2:
            new_p1 = p1 + int(np.ceil(diff/2.0))
            new_p2 = p2 - int(np.floor(diff/2.0))
            print(f"p1 >= p2, so new_p1={p1}+{int(np.ceil(diff/2.0))}={new_p1}, new_p2={p2}-{int(np.floor(diff/2.0))}={new_p2}")
        else:
            new_p1 = p1 - int(np.floor(diff/2.0))
            new_p2 = p2 + int(np.ceil(diff/2.0))
            print(f"p1 < p2, so new_p1={p1}-{int(np.floor(diff/2.0))}={new_p1}, new_p2={p2}+{int(np.ceil(diff/2.0))}={new_p2}")
        
        in_bounds = 0 <= new_p1 <= 255 and 0 <= new_p2 <= 255
        print(f"Bounds check: 0 <= {new_p1} <= 255 and 0 <= {new_p2} <= 255 = {in_bounds}")
        
        if in_bounds:
            print("✓ WILL EMBED")
        else:
            print("✗ WILL SKIP")
        break

# Now actually embed
from core.pvd import PVD
stego, _ = PVD.embed(cover, message, key)
print(f"\nAfter embed:")
print(f"Stego pixels at ({r0}, {c0}): {int(stego[r0, c0])}, {int(stego[r0, c0+1])}")
print(f"Changed: {int(stego[r0, c0]) != p1 or int(stego[r0, c0+1]) != p2}")
