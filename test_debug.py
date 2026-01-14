"""
Debug test cho từng thuật toán failed
"""
import cv2
import numpy as np
from core.pvd import PVD
from core.interpolation import Interpolation
from core.difference_expansion import DifferenceExpansion

def test_pvd():
    print("\n" + "="*60)
    print("DEBUG PVD")
    print("="*60)
    
    cover = cv2.imread("data/input/BOSSbase_256/1.pgm", cv2.IMREAD_GRAYSCALE)
    message = "Hello World!"
    key = "test123"
    
    print(f"Original message: {message}")
    print(f"Message length: {len(message)} chars")
    
    # Embed
    stego, param = PVD.embed(cover, message, key)
    print(f"Embed param: {param}")
    print(f"Stego shape: {stego.shape}")
    
    # Extract
    extracted = PVD.extract(stego, key)
    print(f"Extracted: {extracted}")
    print(f"Match: {extracted == message}")
    
    return extracted == message

def test_interpolation():
    print("\n" + "="*60)
    print("DEBUG INTERPOLATION")
    print("="*60)
    
    cover = cv2.imread("data/input/BOSSbase_256/1.pgm", cv2.IMREAD_GRAYSCALE)
    message = "Hello World!"
    key = "test123"
    
    print(f"Original message: {message}")
    print(f"Message length: {len(message)} chars")
    
    # Embed
    stego, n_bits = Interpolation.embed(cover, message, key)
    print(f"Embed n_bits: {n_bits}")
    print(f"Stego shape: {stego.shape}")
    
    # Extract with same n_bits
    extracted = Interpolation.extract(stego, key, n_bits=n_bits)
    print(f"Extracted: {extracted}")
    print(f"Match: {extracted == message}")
    
    return extracted == message

def test_difference_expansion():
    print("\n" + "="*60)
    print("DEBUG DIFFERENCE EXPANSION")
    print("="*60)
    
    cover = cv2.imread("data/input/BOSSbase_256/1.pgm", cv2.IMREAD_GRAYSCALE)
    message = "Hello World!"
    key = "test123"
    
    print(f"Original message: {message}")
    print(f"Message length: {len(message)} chars")
    
    # Embed
    stego, layers = DifferenceExpansion.embed(cover, message, key)
    print(f"Embed layers: {layers}")
    print(f"Stego shape: {stego.shape}")
    
    # Extract with same layers
    extracted = DifferenceExpansion.extract(stego, key, layers=layers)
    print(f"Extracted: {extracted}")
    print(f"Extracted type: {type(extracted)}")
    print(f"Match: {extracted == message}")
    
    return extracted == message

if __name__ == "__main__":
    pvd_pass = test_pvd()
    interp_pass = test_interpolation()
    de_pass = test_difference_expansion()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"PVD: {'✅ PASS' if pvd_pass else '❌ FAIL'}")
    print(f"Interpolation: {'✅ PASS' if interp_pass else '❌ FAIL'}")
    print(f"Difference Expansion: {'✅ PASS' if de_pass else '❌ FAIL'}")
