#!/usr/bin/env python3
"""Debug PVD embed process in detail"""
import numpy as np
import cv2
import random

# Load test image
cover_path = "data/input/BOSSbase_256/1.pgm"
cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)

message = "AB"
key = "key"

print("=" * 70)
print("DEBUG: PVD Embed Process DETAIL")
print("=" * 70)

# Expected binary
data = message.encode('utf-8') + b'\x00\x00\x00'
bin_msg = ''.join([format(b, "08b") for b in data])
print(f"Message: '{message}'")
print(f"Binary to embed: {bin_msg}")

# Simulate embed with debugging
rng = random.Random(key)
h, w = cover.shape
coords = [(r, c) for r in range(h) for c in range(w-1)]
rng.shuffle(coords)

stego = cover.copy().astype(np.int32)
idx = 0
ranges = [0, 8, 16, 32, 64, 128, 256]

print("\nEmbedding details:")
for coord_num, (r, c) in enumerate(coords):
    if idx >= len(bin_msg) or coord_num >= 20:  # First 20 pairs
        break
        
    p1, p2 = stego[r, c], stego[r, c+1]
    d = abs(p1 - p2)
    
    # Find range
    for range_idx in range(len(ranges)-1):
        if ranges[range_idx] <= d < ranges[range_idx+1]:
            low, high = ranges[range_idx], ranges[range_idx+1]-1
            t = int(np.log2(high - low + 1))
            
            # Get t bits to embed
            bits = bin_msg[idx : idx+t]
            if not bits:
                break
                
            b = int(bits.ljust(t, '0'), 2)
            d_new = low + b
            diff = d_new - d
            
            # Adjust pixels
            if p1 >= p2:
                new_p1 = p1 + int(np.ceil(diff/2))
                new_p2 = p2 - int(np.floor(diff/2))
            else:
                new_p1 = p1 - int(np.floor(diff/2))
                new_p2 = p2 + int(np.ceil(diff/2))
            
            # Check bounds
            if 0 <= new_p1 <= 255 and 0 <= new_p2 <= 255:
                old_d = d
                new_d = abs(new_p1 - new_p2)
                print(f"  Pair {coord_num}: p1={p1}->{new_p1}, p2={p2}->{new_p2}, d={old_d}->{new_d}, range=[{low},{high}], t={t}, b={b}, bits_embedded='{bits}'")
                
                stego[r, c] = new_p1
                stego[r, c+1] = new_p2
                idx += len(bits)
            else:
                print(f"  Pair {coord_num}: SKIPPED (out of bounds) - would be p1={new_p1}, p2={new_p2}")
            break

print(f"\nTotal bits embedded: {idx}")
print(f"Total bits to embed: {len(bin_msg)}")

print("\n" + "=" * 70)
print("Now comparing with extract...")
print("=" * 70)

# Now extract
rng2 = random.Random(key)
coords2 = [(r, c) for r in range(h) for c in range(w-1)]
rng2.shuffle(coords2)

bin_extracted = ""
print("\nExtracting details:")
for coord_num, (r, c) in enumerate(coords2):
    if len(bin_extracted) >= len(bin_msg) or coord_num >= 20:
        break
        
    p1, p2 = int(stego[r, c]), int(stego[r, c+1])
    d = abs(p1 - p2)
    
    # Find range
    for range_idx in range(len(ranges)-1):
        if ranges[range_idx] <= d < ranges[range_idx+1]:
            low = ranges[range_idx]
            high = ranges[range_idx+1] - 1
            t = int(np.log2(high - low + 1))
            b = d - low
            extracted_bits = format(b, f'0{t}b')
            
            print(f"  Pair {coord_num}: p1={p1}, p2={p2}, d={d}, range=[{low},{high}], t={t}, b={b}, bits_extracted='{extracted_bits}'")
            bin_extracted += extracted_bits
            break

print(f"\nBinary extracted: {bin_extracted[:40]}")
print(f"Binary expected:  {bin_msg[:40]}")
