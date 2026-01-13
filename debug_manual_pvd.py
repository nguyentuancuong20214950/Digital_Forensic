"""
Manual PVD embed trace - detailed
"""
import numpy as np
import random

# Simulate cover
cover = np.ones((10, 10), dtype=np.uint8) * 255

message = "AB"
key = 12345

# Message to binary
data = message.encode('utf-8') + b'\x00\x00\x00'
bin_msg = ''.join([format(b, "08b") for b in data])
print(f"Message: '{message}'")
print(f"Binary: {bin_msg}")

# Embed
rng = random.Random(key)
h, w = cover.shape
coords = [(r, c) for r in range(h) for c in range(w-1)]
print(f"Total coords: {len(coords)}")
rng.shuffle(coords)
print(f"First 5 coords: {coords[:5]}")

stego = cover.copy().astype(np.int32)
idx = 0
ranges = [0, 8, 16, 32, 64, 128, 256]

print(f"\n=== EMBED ===")
for pair_num, (r, c) in enumerate(coords[:5]):
    if idx >= len(bin_msg):
        break
        
    p1, p2 = int(stego[r, c]), int(stego[r, c+1])
    d = abs(p1 - p2)
    
    print(f"\nPair {pair_num}: ({r},{c}) p1={p1}, p2={p2}, d={d}")
    
    # Find range
    for range_idx in range(len(ranges)-1):
        if ranges[range_idx] <= d < ranges[range_idx+1]:
            low, high = ranges[range_idx], ranges[range_idx+1]-1
            t = int(np.log2(high - low + 1))
            
            bits = bin_msg[idx : idx+t]
            if not bits:
                break
                
            b = int(bits.ljust(t, '0'), 2)
            d_new = low + b
            diff = d_new - d
            
            if p1 >= p2:
                new_p1 = p1 + int(np.ceil(diff/2.0))
                new_p2 = p2 - int(np.floor(diff/2.0))
            else:
                new_p1 = p1 - int(np.floor(diff/2.0))
                new_p2 = p2 + int(np.ceil(diff/2.0))
            
            print(f"  Range [{low},{high}], t={t}, bits='{bits}', b={b}, d_new={d_new}, diff={diff}")
            print(f"  new_p1={new_p1}, new_p2={new_p2}")
            
            in_bounds = 0 <= new_p1 <= 255 and 0 <= new_p2 <= 255
            print(f"  Bounds check: {in_bounds}")
            
            if in_bounds:
                stego[r, c] = new_p1
                stego[r, c+1] = new_p2
                print(f"  MODIFIED: stego[{r},{c}]={stego[r,c]}, stego[{r},{c+1}]={stego[r,c+1]}")
                idx += len(bits)
            else:
                print(f"  SKIPPED")
            break

print(f"\nFinal stego[0:3,0:3]:\n{stego[0:3,0:3].astype(np.uint8)}")
