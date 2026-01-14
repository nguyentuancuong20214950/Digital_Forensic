"""
Test với ảnh 100.pgm
"""
import cv2
import numpy as np
from core.pvd import PVD
from core.interpolation import Interpolation
from core.difference_expansion import DifferenceExpansion

cover = cv2.imread("data/input/BOSSbase_256/100.pgm", cv2.IMREAD_GRAYSCALE)
message = "Hello World!"
key = "test123"

print("Testing with 100.pgm (fewer boundary pixels)...")
print(f"Image stats: min={cover.min()}, max={cover.max()}, mean={cover.mean():.1f}")

# PVD
print("\n" + "="*60)
print("PVD")
stego, param = PVD.embed(cover, message, key)
extracted = PVD.extract(stego, key)
print(f"Original:  {message}")
print(f"Extracted: {extracted}")
print(f"Match: {extracted == message}")

# DE
print("\n" + "="*60)
print("Difference Expansion")
stego, layers = DifferenceExpansion.embed(cover, message, key)
extracted = DifferenceExpansion.extract(stego, key, layers=layers)
print(f"Original:  {message}")
print(f"Extracted: {extracted}")
print(f"Match: {extracted == message}")

# Interpolation
print("\n" + "="*60)
print("Interpolation")
stego, n_bits = Interpolation.embed(cover, message, key)
extracted = Interpolation.extract(stego, key, n_bits=n_bits)
print(f"Original:  {message}")
print(f"Extracted: {extracted}")
print(f"Match: {extracted == message}")
