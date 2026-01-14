"""Quick test for updated interpolation method"""
import numpy as np
from core.interpolation import Interpolation

# Simple test
cover = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
message = "Hello World!"

print("Testing Interpolation method...")
print(f"Message: '{message}'")

# Embed
stego, param = Interpolation.embed(cover, message)
print(f"Embed successful, param: {param}")
print(f"Stego shape: {stego.shape}, dtype: {stego.dtype}")

# Extract
extracted = Interpolation.extract(stego, param)
print(f"Extracted: '{extracted}'")

# Verify
if extracted == message:
    print("✓ TEST PASSED!")
else:
    print("✗ TEST FAILED!")
    print(f"Expected: '{message}'")
    print(f"Got: '{extracted}'")
