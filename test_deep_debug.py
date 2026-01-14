"""
Deep debug với binary output
"""
import cv2
import numpy as np
from core.pvd import PVD
from core.interpolation import Interpolation
from core.difference_expansion import DifferenceExpansion

def debug_pvd():
    print("\n" + "="*60)
    print("DEEP DEBUG PVD")
    print("="*60)
    
    cover = cv2.imread("data/input/BOSSbase_256/1.pgm", cv2.IMREAD_GRAYSCALE)
    message = "Hi"  # Short message
    key = "test123"
    
    # Manual embed trace
    bin_msg = ''.join([format(ord(i), "08b") for i in message]) + "00000000"
    print(f"Original message: {message}")
    print(f"Binary (with sentinel): {bin_msg}")
    print(f"Binary length: {len(bin_msg)} bits")
    
    # Embed
    stego, param = PVD.embed(cover, message, key)
    
    # Manual extract trace
    flat = stego.flatten().astype(np.int32)
    bin_extracted = ""
    
    for i in range(0, min(10, len(flat)-1), 2):  # First 5 pairs
        p1, p2 = flat[i], flat[i+1]
        d = abs(p1 - p2)
        for r in range(len(PVD.ranges)-1):
            if PVD.ranges[r] <= d < PVD.ranges[r+1]:
                low, high = PVD.ranges[r], PVD.ranges[r+1]-1
                t = int(np.log2(high - low + 1))
                b = d - low
                bits = format(b, f'0{t}b')
                bin_extracted += bits
                print(f"Pair {i//2}: d={d}, range=[{PVD.ranges[r]},{PVD.ranges[r+1]}), t={t}, b={b}, bits={bits}")
                break
    
    print(f"\nExtracted binary (first {len(bin_extracted)} bits): {bin_extracted}")
    print(f"Expected binary (first {len(bin_extracted)} bits): {bin_msg[:len(bin_extracted)]}")
    print(f"Match: {bin_extracted == bin_msg[:len(bin_extracted)]}")

def debug_de():
    print("\n" + "="*60)
    print("DEEP DEBUG DIFFERENCE EXPANSION")
    print("="*60)
    
    cover = cv2.imread("data/input/BOSSbase_256/1.pgm", cv2.IMREAD_GRAYSCALE)
    message = "Hi"
    key = "test123"
    
    bin_msg = ''.join([format(ord(i), "08b") for i in message]) + "00000000"
    print(f"Original message: {message}")
    print(f"Binary: {bin_msg}")
    print(f"Binary length: {len(bin_msg)} bits")
    
    # Embed
    stego, layers = DifferenceExpansion.embed(cover, message, key)
    print(f"Embed layers: {layers}")
    
    # Manual extract
    flat = stego.flatten().astype(np.int32)
    bin_extracted = ""
    
    for i in range(0, min(20, len(flat) - 1), 2):
        x, y = flat[i], flat[i+1]
        l = (x + y) // 2
        d_new = x - y
        
        # Extract bit
        if d_new >= 0:
            b = d_new % 2
        else:
            b = abs(d_new) % 2
        
        bin_extracted += str(b)
        
        if i < 10:
            print(f"Pair {i//2}: x={x}, y={y}, d_new={d_new}, bit={b}")
    
    print(f"\nExtracted binary: {bin_extracted[:24]}")
    print(f"Expected binary: {bin_msg[:24]}")
    print(f"Match: {bin_extracted[:24] == bin_msg[:24]}")
    
    # Check for NULL sentinel
    print(f"\nLooking for '00000000' in extracted...")
    if "00000000" in bin_extracted:
        print(f"Found at position {bin_extracted.index('00000000')}")
    else:
        print("NOT FOUND")

if __name__ == "__main__":
    debug_pvd()
    debug_de()
