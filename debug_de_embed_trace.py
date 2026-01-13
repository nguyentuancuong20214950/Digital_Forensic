#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import cv2
import numpy as np

cover = cv2.imread("data/input/BOSSbase_256/2.pgm", cv2.IMREAD_GRAYSCALE)
message = "The quick brown fox jumps over the lazy dog"
key = 54321

data = message.encode('utf-8') + b'\x00\x00\x00'
bin_msg = ''.join([format(b, "08b") for b in data])
print(f"Expected binary length: {len(bin_msg)} bits")

# Trace embed
rng = random.Random(key)
h, w = cover.shape
coords = []
for r in range(h):
    for c in range(0, w-1):
        coords.append((r, c))
rng.shuffle(coords)

stego = cover.copy().astype(np.int32)
msg_idx = 0
embed_count = 0
skip_count = 0

for pair_idx, (r, c) in enumerate(coords[:400]):  # First 400 pairs
    if msg_idx >= len(bin_msg):
        break
    
    x = int(stego[r, c])
    y = int(stego[r, c+1])
    
    l = (x + y) // 2
    d = x - y
    
    b = int(bin_msg[msg_idx])
    d_new = 2 * d + b
    
    x_new = l + (d_new + 1) // 2
    y_new = l - d_new // 2
    
    if 0 <= x_new <= 255 and 0 <= y_new <= 255:
        stego[r, c] = x_new
        stego[r, c+1] = y_new
        msg_idx += 1
        embed_count += 1
    else:
        skip_count += 1
    
    if pair_idx < 30 or msg_idx in [352-8, 352, 352+8]:  # Show around byte 44
        status = "EMBED" if (0 <= x_new <= 255 and 0 <= y_new <= 255) else "SKIP"
        byte_num = msg_idx // 8
        print(f"Pair {pair_idx}: msg_idx={msg_idx}, byte={byte_num}, status={status}, "
              f"x={x}, y={y}, d={d}, d_new={d_new}, x_new={x_new}, y_new={y_new}")

print(f"\nTotal embed: {embed_count}, skip: {skip_count}")
print(f"Bits embedded: {msg_idx}")
