#!/usr/bin/env python3
"""Debug PVD embed/extract process"""
import numpy as np
import cv2
from core.pvd import PVD

# Load test image
cover_path = "data/input/BOSSbase_256/1.pgm"
cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)

message = "AB"
key = "key"

print("=" * 70)
print("DEBUG: PVD Embed/Extract Process")
print("=" * 70)

# Expected binary
data = message.encode('utf-8') + b'\x00\x00\x00'
bin_msg = ''.join([format(b, "08b") for b in data])
print(f"Message: '{message}'")
print(f"Binary: {bin_msg}")
print(f"Message bytes: {data.hex()}")
print(f"Message binary parts:")
for i, b in enumerate(data):
    print(f"  Byte {i}: {b:3d} = {b:08b}")

print("\nEmbedding...")
stego, param = PVD.embed(cover, message, key)

print(f"\nExtracting...")
extracted = PVD.extract(stego, key)

print(f"\nResult:")
print(f"  Expected:  '{message}'")
print(f"  Extracted: '{extracted}'")
print(f"  Expected bytes:  {message.encode('utf-8').hex()}")
print(f"  Got bytes:       {extracted.encode('utf-8').hex()}")

# Also check what bits were actually extracted
print(f"\nDebug extract process:")
import random
rng = random.Random(key)
h, w = stego.shape
coords = [(r, c) for r in range(h) for c in range(w-1)]
rng.shuffle(coords)

bin_extracted = ""
for idx, (r, c) in enumerate(coords[:50]):  # Just first 50
    p1, p2 = int(stego[r, c]), int(stego[r, c+1])
    d = abs(p1 - p2)
    
    # Find range
    ranges = [0, 8, 16, 32, 64, 128, 256]
    for range_idx in range(len(ranges)-1):
        if ranges[range_idx] <= d < ranges[range_idx+1]:
            low = ranges[range_idx]
            high = ranges[range_idx+1] - 1
            t = int(np.log2(high - low + 1))
            b = d - low
            extracted_bits = format(b, f'0{t}b')
            bin_extracted += extracted_bits
            
            if idx < 3:  # Print first few
                print(f"  Pair {idx}: p1={p1}, p2={p2}, d={d}, range=[{low},{high}], t={t}, b={b}, bits={extracted_bits}")
            break

print(f"\nExtracted binary (first 32 bits): {bin_extracted[:32]}")
print(f"Expected binary  (first 32 bits): {bin_msg[:32]}")
