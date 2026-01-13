#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import sys
from core.pvd import PVD
from core.difference_expansion import DifferenceExpansion

# Redirect to utf-8
sys.stdout.reconfigure(encoding='utf-8')

# Load test image
cover_path = "data/input/BOSSbase_256/2.pgm"
cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)

print("=" * 70)
print("COMPREHENSIVE TEST: PVD & DIFFERENCE EXPANSION")
print("=" * 70)
print("Cover shape: " + str(cover.shape))
print("=" * 70)

# Test messages
test_cases = [
    ("Simple", "Hello World!"),
    ("Short", "Test"),
    ("Medium", "The quick brown fox jumps over the lazy dog"),
    ("UTF-8", "Xin chao Viet Nam!"),
    ("Numbers", "1234567890"),
    ("Special", "!@#$%^&*()_+-=[]{}"),
]

def test_algorithm(algo_name, algo_class, test_cases):
    print("\n" + "=" * 70)
    print("Testing: " + algo_name)
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for test_name, message in test_cases:
        print("\nTest: " + test_name)
        print("  Message: '" + message + "' (" + str(len(message)) + " chars)")
        
        key = 54321
        
        try:
            # Embed
            stego, param = algo_class.embed(cover, message, key)
            print("  [OK] Embed: param=" + param)
            
            # Extract
            extracted = algo_class.extract(stego, key)
            print("  [OK] Extract: '" + extracted + "'")
            
            # Compare
            if extracted == message:
                print("  [PASS]")
                passed += 1
            else:
                print("  [FAIL]")
                print("     Expected: '" + message + "'")
                print("     Got: '" + extracted + "'")
                failed += 1
                
        except Exception as e:
            print("  [ERROR]: " + str(e))
            failed += 1
    
    print("\nResults: " + str(passed) + " passed, " + str(failed) + " failed")
    print("-" * 70)
    
    return passed, failed


# Test PVD
pvd_passed, pvd_failed = test_algorithm("PVD (Pixel Value Differencing)", PVD, test_cases)

# Test DE
de_passed, de_failed = test_algorithm("Difference Expansion", DifferenceExpansion, test_cases)

# Summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print("PVD:                 " + str(pvd_passed) + " passed, " + str(pvd_failed) + " failed")
print("Difference Expansion: " + str(de_passed) + " passed, " + str(de_failed) + " failed")
print("")
print("Total: " + str(pvd_passed + de_passed) + " passed, " + str(pvd_failed + de_failed) + " failed")
print("=" * 70)
