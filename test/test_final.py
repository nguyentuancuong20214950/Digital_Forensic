"""
Simple test for all steganography methods
"""
import numpy as np
import cv2
from core.lsb_sub import LSB_Sub
from core.lsb_matching import LSB_Matching
from core.pvd import PVD
from core.emd import EMD
from core.histogram_shifting import HistogramShifting
from core.interpolation import Interpolation
from core.difference_expansion import DifferenceExpansion

def test_method(name, embed_fn, extract_fn, message, use_key=False):
    """Test a steganography method"""
    # Load test image
    cover = np.random.randint(50, 200, (256, 256), dtype=np.uint8)
    
    # Embed
    if use_key:
        result = embed_fn(cover, message, "testkey123")
    else:
        result = embed_fn(cover, message)
    
    if isinstance(result, tuple):
        stego, param = result
    else:
        stego = result
        param = None
    
    # Extract
    if use_key and param:
        extracted = extract_fn(stego, "testkey123", **{list(extract_fn.__code__.co_varnames)[2]: param})
    elif use_key:
        extracted = extract_fn(stego, "testkey123")
    elif param and isinstance(param, int):
        extracted = extract_fn(stego, **{list(extract_fn.__code__.co_varnames)[1]: param})
    else:
        extracted = extract_fn(stego)
    
    success = (extracted == message)
    status = "PASS" if success else "FAIL"
    print(f"{name:20s} [{status}] - Expected: '{message}' | Got: '{extracted}'")
    return success

print("="*70)
print("STEGANOGRAPHY METHODS TEST")
print("="*70)

msg = "LMAO LMEO"
results = {}

# Test all methods
results['LSB Sub'] = test_method("LSB Substitution", LSB_Sub.embed, LSB_Sub.extract, msg, use_key=True)
results['LSB Match'] = test_method("LSB Matching", LSB_Matching.embed, LSB_Matching.extract, msg, use_key=True)
results['PVD'] = test_method("PVD", PVD.embed, PVD.extract, msg)
results['DE'] = test_method("Difference Expansion", DifferenceExpansion.embed, DifferenceExpansion.extract, msg)
results['EMD'] = test_method("EMD", EMD.embed, EMD.extract, msg)
results['HS'] = test_method("Histogram Shifting", HistogramShifting.embed, HistogramShifting.extract, msg)
results['Interpolation'] = test_method("Interpolation", Interpolation.embed, Interpolation.extract, msg)

# Summary
print("="*70)
passed = sum(1 for v in results.values() if v)
total = len(results)
print(f"RESULT: {passed}/{total} tests passed")
print("="*70)
