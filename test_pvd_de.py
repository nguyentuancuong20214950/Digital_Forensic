#!/usr/bin/env python3
"""
Comprehensive test for PVD and Difference Expansion algorithms
"""
import numpy as np
import cv2
from core.pvd import PVD
from core.difference_expansion import DifferenceExpansion

# Load test image
cover_path = "data/input/BOSSbase_256/2.pgm"
cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)

print("=" * 70)
print("COMPREHENSIVE TEST: PVD & DIFFERENCE EXPANSION")
print("=" * 70)
print(f"Cover: {cover_path}")
print(f"Cover shape: {cover.shape}")
print("=" * 70)

# Test messages
test_cases = [
    ("Simple", "Hello World!"),
    ("Short", "Test"),
    ("Medium", "The quick brown fox jumps over the lazy dog"),
    ("UTF-8", "Xin chào Việt Nam!"),
    ("Numbers", "1234567890"),
    ("Special", "!@#$%^&*()_+-=[]{}|;:',.<>?"),
]

def test_algorithm(algo_name, algo_class, test_cases):
    print(f"\n{'='*70}")
    print(f"Testing: {algo_name}")
    print(f"{'='*70}")
    
    passed = 0
    failed = 0
    
    for test_name, message in test_cases:
        print(f"\nTest: {test_name}")
        print(f"  Message: '{message}' ({len(message)} chars)")
        
        key = 54321  # Use integer key that works well with PVD
        
        try:
            # Embed
            stego, param = algo_class.embed(cover, message, key)
            print(f"  [OK] Embed: param={param}")
            
            # Extract
            extracted = algo_class.extract(stego, key)
            print(f"  [OK] Extract: '{extracted}'")
            
            # Compare
            if extracted == message:
                print(f"  [PASS]")
                passed += 1
            else:
                print(f"  [FAIL]")
                print(f"     Expected: '{message}'")
                print(f"     Got:      '{extracted}'")
                print(f"     Expected bytes: {message.encode('utf-8').hex()}")
                print(f"     Got bytes:      {extracted.encode('utf-8').hex()}")
                failed += 1
                
        except Exception as e:
            print(f"  [ERROR]: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print(f"\n{'-'*70}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'-'*70}")
    
    return passed, failed

# Test PVD
pvd_passed, pvd_failed = test_algorithm("PVD (Pixel Value Differencing)", PVD, test_cases)

# Test Difference Expansion
de_passed, de_failed = test_algorithm("Difference Expansion", DifferenceExpansion, test_cases)

# Summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"PVD:                 {pvd_passed} passed, {pvd_failed} failed")
print(f"Difference Expansion: {de_passed} passed, {de_failed} failed")
print(f"\nTotal: {pvd_passed + de_passed} passed, {pvd_failed + de_failed} failed")
print("=" * 70)

# Additional stress test with longer message
print("\n" + "=" * 70)
print("STRESS TEST: Long Message")
print("=" * 70)

long_message = """Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."""

print(f"Message length: {len(long_message)} chars")

for algo_name, algo_class in [("PVD", PVD), ("Difference Expansion", DifferenceExpansion)]:
    print(f"\n{algo_name}:")
    try:
        stego, param = algo_class.embed(cover, long_message, 54321)
        extracted = algo_class.extract(stego, 54321)
        
        if extracted == long_message:
            print(f"  ✅ PASS: Long message correctly embedded/extracted")
        else:
            print(f"  ❌ FAIL: Message mismatch")
            print(f"     Expected length: {len(long_message)}")
            print(f"     Got length: {len(extracted)}")
            print(f"     First 50 chars expected: '{long_message[:50]}'")
            print(f"     First 50 chars got:      '{extracted[:50]}'")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
