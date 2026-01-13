#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import random
from core.difference_expansion import DifferenceExpansion

cover = cv2.imread("data/input/BOSSbase_256/2.pgm", cv2.IMREAD_GRAYSCALE)
message = "The quick brown fox jumps over the lazy dog"
key = 54321

print(f"Message: {message}")
print(f"Length: {len(message)} chars")

# Get binary
data = message.encode('utf-8') + b'\x00\x00\x00'
bin_expected = ''.join([format(b, "08b") for b in data])
print(f"Binary to embed: {bin_expected[:96]}...")  # First 96 bits
print(f"Total bits: {len(bin_expected)}")

# Embed
stego, _ = DifferenceExpansion.embed(cover, message, key)

# Now trace extract
rng = random.Random(key)
h, w = stego.shape
coords = []
for r in range(h):
    for c in range(0, w-1):
        coords.append((r, c))
rng.shuffle(coords)

bin_extracted = ""
for pair_idx, (r, c) in enumerate(coords[:400]):  # First 400 pairs = 400 bits
    x = int(stego[r, c])
    y = int(stego[r, c+1])
    
    l = (x + y) // 2
    d_new = x - y
    b = d_new % 2
    bin_extracted += str(b)

print(f"\nExtracted binary: {bin_extracted[:96]}...")
print(f"Expected binary:  {bin_expected[:96]}...")

# Convert to bytes
byte_list = []
for i in range(0, len(bin_extracted), 8):
    byte_str = bin_extracted[i:i+8]
    if len(byte_str) < 8:
        break
    byte_val = int(byte_str, 2)
    if byte_val == 0:
        print(f"Byte {i//8}: NULL (terminator)")
        break
    byte_list.append(byte_val)
    if i < 96:  # First 12 bytes
        print(f"Byte {i//8}: 0x{byte_val:02x} = '{chr(byte_val)}'")

extracted = bytes(byte_list).decode('utf-8', errors='ignore')
print(f"\nExtracted: {extracted}")
print(f"Expected:  {message}")
print(f"Match: {extracted == message}")
