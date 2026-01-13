"""
Debug PVD - detailed extract trace
"""
import cv2
import numpy as np
import random

# Load cover image
cover = cv2.imread('data/input/BOSSbase_256/1.pgm', cv2.IMREAD_GRAYSCALE)

message = "AB"
key = 12345

# Message to binary
data = message.encode('utf-8') + b'\x00\x00\x00'
bin_msg_expected = ''.join([format(b, "08b") for b in data])
print(f"Message: '{message}'")
print(f"Expected binary: {bin_msg_expected}")
print(f"Total bits: {len(bin_msg_expected)}")

# Embed
from core.pvd import PVD
stego, _ = PVD.embed(cover, message, key)

# Extract - trace
rng = random.Random(key)
h, w = stego.shape
coords = [(r, c) for r in range(h) for c in range(w-1)]
rng.shuffle(coords)

ranges = [0, 8, 16, 32, 64, 128, 256]
bin_msg = ""

print("\n=== EXTRACT TRACE ===")
for pair_idx, (r, c) in enumerate(coords[:30]):  # First 30 pairs
    p1, p2 = int(stego[r, c]), int(stego[r, c+1])
    d = abs(p1 - p2)
    
    # Find range
    for range_idx in range(len(ranges)-1):
        if ranges[range_idx] <= d < ranges[range_idx+1]:
            low = ranges[range_idx]
            high = ranges[range_idx+1] - 1
            t = int(np.log2(high - low + 1))
            
            # Extract bits
            b = d - low
            bits = format(b, f'0{t}b')
            bin_msg += bits
            
            print(f"Pair {pair_idx}: p1={p1}, p2={p2}, d={d}, range=[{low},{high}], t={t}, b={b}, bits='{bits}'")
            print(f"  Total bits so far: {len(bin_msg)}")
            break
    
    if len(bin_msg) >= 64:  # Stop after enough bits
        break

print(f"\nExtracted binary: {bin_msg}")
print(f"Expected binary:  {bin_msg_expected}")

# Convert to bytes
byte_list = []
for i in range(0, len(bin_msg), 8):
    byte_str = bin_msg[i:i+8]
    if len(byte_str) < 8:
        break
    byte_val = int(byte_str, 2)
    byte_list.append(byte_val)
    print(f"Byte {i//8}: '{chr(byte_val)}' (0x{byte_val:02x})")

extracted_msg = bytes(byte_list).decode('utf-8', errors='ignore')
print(f"\nExtracted message: '{extracted_msg}'")
