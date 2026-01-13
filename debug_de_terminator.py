#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import cv2

cover = cv2.imread("data/input/BOSSbase_256/2.pgm", cv2.IMREAD_GRAYSCALE)
message = "The quick brown fox jumps over the lazy dog"
key = 54321

data = message.encode('utf-8') + b'\x00\x00\x00'
bin_msg_expected = ''.join([format(b, "08b") for b in data])

from core.difference_expansion import DifferenceExpansion
stego, _ = DifferenceExpansion.embed(cover, message, key)

# Trace extract
rng = random.Random(key)
h, w = stego.shape
coords = []
for r in range(h):
    for c in range(0, w-1):
        coords.append((r, c))
rng.shuffle(coords)

bin_msg = ""
for pair_idx, (r, c) in enumerate(coords):
    x = int(stego[r, c])
    y = int(stego[r, c+1])
    
    l = (x + y) // 2
    d_new = x - y
    b = d_new % 2
    bin_msg += str(b)
    
    # Check for terminator
    if len(bin_msg) >= 24 and "000000000000000000000000" in bin_msg[-48:]:
        print(f"Break at pair {pair_idx}, bin_msg length {len(bin_msg)}")
        print(f"Last 48 bits: {bin_msg[-48:]}")
        break

print(f"\nExpected binary ({len(bin_msg_expected)} bits):")
print(bin_msg_expected)
print(f"\nExtracted binary ({len(bin_msg)} bits):")
print(bin_msg)

# Find where 24 zeros start
for i in range(0, len(bin_msg) - 23):
    if bin_msg[i:i+24] == "000000000000000000000000":
        print(f"\n24 zeros found at position {i} (byte {i//8})")
        break
