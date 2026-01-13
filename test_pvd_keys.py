"""
Test PVD with different keys to find one that works
"""
import cv2
from core.pvd import PVD

cover = cv2.imread('data/input/BOSSbase_256/1.pgm', cv2.IMREAD_GRAYSCALE)
message = "AB"

print(f"Testing PVD with different keys...")
for key in [1, 42, 123, 999, 12345, 54321, 99999]:
    try:
        stego, _ = PVD.embed(cover, message, key)
        extracted = PVD.extract(stego, key)
        match = extracted == message
        print(f"Key {key:5d}: {'✓ PASS' if match else '✗ FAIL'} - extracted='{extracted}'")
        if match:
            print(f"  SUCCESS with key={key}")
            break
    except Exception as e:
        print(f"Key {key:5d}: ✗ ERROR - {e}")
