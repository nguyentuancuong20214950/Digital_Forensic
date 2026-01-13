#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
from core.difference_expansion import DifferenceExpansion

cover = cv2.imread("data/input/BOSSbase_256/2.pgm", cv2.IMREAD_GRAYSCALE)
message = "The quick brown fox jumps over the lazy dog"
key = 54321

stego, _ = DifferenceExpansion.embed(cover, message, key)
extracted = DifferenceExpansion.extract(stego, key)

print(f"Expected: {message}")
print(f"Got:      {extracted}")
print(f"Expected len: {len(message)}")
print(f"Got len:      {len(extracted)}")
print()
print(f"Expected bytes: {message.encode().hex()}")
print(f"Got bytes:      {extracted.encode().hex()}")
print()
print(f"Expected repr: {repr(message)}")
print(f"Got repr:      {repr(extracted)}")

# Show character by character
print("\nCharacter by character comparison:")
for i in range(min(len(message), len(extracted))):
    if message[i] == extracted[i]:
        print(f"{i}: OK '{message[i]}'")
    else:
        print(f"{i}: MISMATCH expected '{message[i]}' got '{extracted[i]}'")

if len(message) > len(extracted):
    print(f"\nExpected has {len(message) - len(extracted)} more chars:")
    for i in range(len(extracted), len(message)):
        print(f"{i}: missing '{message[i]}'")
elif len(extracted) > len(message):
    print(f"\nGot has {len(extracted) - len(message)} extra chars:")
    for i in range(len(message), len(extracted)):
        print(f"{i}: extra '{extracted[i]}'")
