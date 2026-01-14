"""
Test tất cả 7 thuật toán steganography với thiết kế mới (tất cả đều có key parameter)
"""
import cv2
import numpy as np
from core.lsb_sub import LSB_Sub
from core.lsb_matching import LSB_Matching
from core.pvd import PVD
from core.emd import EMD
from core.histogram_shifting import HistogramShifting
from core.interpolation import Interpolation
from core.difference_expansion import DifferenceExpansion

def test_method(method_name, embed_func, extract_func, cover, message, key):
    """Test một thuật toán cụ thể"""
    print(f"\n{'='*60}")
    print(f"Testing {method_name}...")
    print(f"{'='*60}")
    
    try:
        # Embed
        result = embed_func(cover, message, key)
        
        # Xử lý kết quả embed (có thể là tuple hoặc stego image)
        if isinstance(result, tuple):
            stego, param = result
            print(f"✓ Embed thành công! Param: {param}")
        else:
            stego = result
            param = None
            print(f"✓ Embed thành công!")
        
        # Extract
        if method_name in ["EMD"]:
            # EMD cần n_digits parameter
            extracted = extract_func(stego, key, n_digits=param)
        elif method_name in ["Histogram Shifting"]:
            # HS cần peak parameter
            extracted = extract_func(stego, key, peak=param)
        elif method_name in ["Difference Expansion"]:
            # DE cần layers parameter
            extracted = extract_func(stego, key, layers=param)
        elif method_name in ["Interpolation"]:
            # Interpolation cần n_bits parameter
            extracted = extract_func(stego, key, n_bits=param)
        else:
            # LSB, PVD không cần param đặc biệt
            extracted = extract_func(stego, key)
        
        # Verify
        if extracted == message:
            print(f"✅ {method_name} PASSED!")
            print(f"   Original: {message[:50]}...")
            print(f"   Extracted: {extracted[:50]}...")
            return True
        else:
            print(f"❌ {method_name} FAILED!")
            print(f"   Expected: {message[:50]}")
            print(f"   Got: {extracted[:50] if extracted else 'None'}")
            return False
            
    except Exception as e:
        print(f"❌ {method_name} ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Load cover image - use 100.pgm as it has fewer boundary pixels
    cover = cv2.imread("data/input/BOSSbase_256/100.pgm", cv2.IMREAD_GRAYSCALE)
    
    if cover is None:
        print("❌ Không thể load ảnh cover!")
        return
    
    print(f"Cover image shape: {cover.shape}")
    print(f"Image stats: min={cover.min()}, max={cover.max()}, mean={cover.mean():.1f}\n")
    
    # Test messages
    short_msg = "Hello, this is a test message!"
    medium_msg = "This is a longer test message to check capacity. " * 5
    
    key = "test_key_123"
    
    # Test all 7 methods
    results = {}
    
    # 1. LSB Substitution
    results["LSB Substitution"] = test_method(
        "LSB Substitution",
        LSB_Sub.embed,
        LSB_Sub.extract,
        cover, short_msg, key
    )
    
    # 2. LSB Matching
    results["LSB Matching"] = test_method(
        "LSB Matching",
        LSB_Matching.embed,
        LSB_Matching.extract,
        cover, short_msg, key
    )
    
    # 3. PVD
    results["PVD"] = test_method(
        "PVD",
        PVD.embed,
        PVD.extract,
        cover, short_msg, key
    )
    
    # 4. EMD
    results["EMD"] = test_method(
        "EMD",
        EMD.embed,
        EMD.extract,
        cover, short_msg, key
    )
    
    # 5. Histogram Shifting
    results["Histogram Shifting"] = test_method(
        "Histogram Shifting",
        HistogramShifting.embed,
        HistogramShifting.extract,
        cover, short_msg, key
    )
    
    # 6. Interpolation
    results["Interpolation"] = test_method(
        "Interpolation",
        Interpolation.embed,
        Interpolation.extract,
        cover, short_msg, key
    )
    
    # 7. Difference Expansion
    results["Difference Expansion"] = test_method(
        "Difference Expansion",
        DifferenceExpansion.embed,
        DifferenceExpansion.extract,
        cover, short_msg, key
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS:")
    print(f"{'='*60}")
    
    passed = sum(results.values())
    total = len(results)
    
    for method, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {method}")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {passed}/{total} methods passed")
    print(f"{'='*60}")
    
    if passed == total:
        print("\n🎉 TẤT CẢ THUẬT TOÁN ĐỀU HOẠT ĐỘNG ĐÚNG!")
        return True
    else:
        print(f"\n⚠️ CÓ {total - passed} THUẬT TOÁN FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
