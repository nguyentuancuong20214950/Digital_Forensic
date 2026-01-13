"""
Debug PVD - trace skip-aware logic
"""
import cv2
import numpy as np
import random
from core.pvd import PVD

# Load cover image
cover = cv2.imread('data/input/BOSSbase_256/1.pgm', cv2.IMREAD_GRAYSCALE)
print(f"Cover shape: {cover.shape}")

# Simple message
message = "AB"
key = 12345
rng = random.Random(key)

# Message to binary
data = message.encode('utf-8') + b'\x00\x00\x00'
bin_msg = ''.join([format(b, "08b") for b in data])
print(f"Message: '{message}'")
print(f"Binary to embed: {bin_msg[:48]}")  # Show first 48 bits

# Get same coords as embed
h, w = cover.shape
coords = [(r, c) for r in range(h) for c in range(w-1)]
rng.shuffle(coords)

# Trace embed logic
print("\n=== EMBED TRACE ===")
idx = 0
for pair_num, (r, c) in enumerate(coords[:20]):  # First 20 pairs
    if idx >= len(bin_msg):
        break
        
    p1, p2 = cover[r, c], cover[r, c+1]
    d = abs(p1 - p2)
    
    # Find range
    ranges = [0, 8, 16, 32, 64, 128, 256]
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
                new_p1 = p1 + int(np.ceil(diff/2.0))
                new_p2 = p2 - int(np.floor(diff/2.0))
            else:
                new_p1 = p1 - int(np.floor(diff/2.0))
                new_p2 = p2 + int(np.ceil(diff/2.0))
            
            # Check bounds
            in_bounds = 0 <= new_p1 <= 255 and 0 <= new_p2 <= 255
            
            print(f"Pair {pair_num}: p1={p1}, p2={p2}, d={d}, range=[{low},{high}], t={t}, "
                  f"bits='{bits}' (idx {idx}), d_new={d_new}, new_p1={new_p1}, new_p2={new_p2}, "
                  f"BOUNDS={'OK' if in_bounds else 'SKIP'}")
            
            if in_bounds:
                idx += len(bits)
            break

# Now trace extract
print("\n=== EXTRACT TRACE (same random seed) ===")
rng2 = random.Random(key)
coords2 = [(r, c) for r in range(h) for c in range(w-1)]
rng2.shuffle(coords2)

bin_extracted = ""
for pair_num, (r, c) in enumerate(coords2[:20]):
    p1, p2 = cover[r, c], cover[r, c+1]  # Extract from ORIGINAL (not stego)
    d = abs(p1 - p2)
    
    # Find range
    ranges = [0, 8, 16, 32, 64, 128, 256]
    for range_idx in range(len(ranges)-1):
        if ranges[range_idx] <= d < ranges[range_idx+1]:
            low = ranges[range_idx]
            high = ranges[range_idx+1] - 1
            t = int(np.log2(high - low + 1))
            b = d - low
            bits = format(b, f'0{t}b')
            bin_extracted += bits
            
            print(f"Pair {pair_num}: p1={p1}, p2={p2}, d={d}, range=[{low},{high}], t={t}, "
                  f"b={b}, bits='{bits}'")
            break

print(f"\nExtracted binary (first 48 bits): {bin_extracted[:48]}")
print(f"Expected binary (first 48 bits): {bin_msg[:48]}")
print(f"Match: {bin_extracted[:48] == bin_msg[:48]}")
