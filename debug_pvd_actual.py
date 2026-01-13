"""
Debug PVD - trace skip-aware logic with ACTUAL embed/extract
"""
import cv2
import numpy as np
import random
from core.pvd import PVD

# Load cover image
cover = cv2.imread('data/input/BOSSbase_256/1.pgm', cv2.IMREAD_GRAYSCALE)
print(f"Cover shape: {cover.shape}")

message = "AB"
key = 12345

# Embed
print("\n=== EMBEDDING ===")
stego, param = PVD.embed(cover, message, key)
print(f"Embedded '{message}' with param={param}")

# Show first few pixels
print(f"First 5 pixels - Cover:  {cover[0, :5]}")
print(f"First 5 pixels - Stego:  {stego[0, :5]}")

# Extract
print("\n=== EXTRACTING ===")
extracted = PVD.extract(stego, key)
print(f"Extracted: '{extracted}'")

# Compare
print(f"\nMatch: {extracted == message}")
print(f"Expected: '{message}'")
print(f"Got: '{extracted}'")
print(f"Expected bytes: {message.encode('utf-8').hex()}")
print(f"Got bytes: {extracted.encode('utf-8').hex()}")
