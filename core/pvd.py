import numpy as np
import random

class PVD:
    ranges = [0, 8, 16, 32, 64, 128, 256]

    @staticmethod
    def embed(image, message, key):
        # Add NULL terminator and convert to binary
        data = message.encode('utf-8') + b'\x00\x00\x00'
        bin_msg = ''.join([format(b, "08b") for b in data])
        
        # Use key for random pixel selection
        rng = random.Random(key)
        h, w = image.shape
        coords = [(r, c) for r in range(h) for c in range(w-1)]
        rng.shuffle(coords)
        
        stego = image.copy().astype(np.int32)
        idx = 0
        
        for r, c in coords:
            if idx >= len(bin_msg):
                break
                
            p1, p2 = int(stego[r, c]), int(stego[r, c+1])  # Convert to Python int FIRST
            d = abs(p1 - p2)
            
            # Find range for this difference
            for range_idx in range(len(PVD.ranges)-1):
                if PVD.ranges[range_idx] <= d < PVD.ranges[range_idx+1]:
                    low, high = PVD.ranges[range_idx], PVD.ranges[range_idx+1]-1
                    t = int(np.log2(high - low + 1))
                    
                    # Get t bits to embed
                    bits = bin_msg[idx : idx+t]
                    if not bits:
                        break
                        
                    b = int(bits.ljust(t, '0'), 2)
                    d_new = low + b
                    diff = d_new - d
                    
                    # Adjust pixels (use Python int arithmetic to avoid overflow)
                    if p1 >= p2:
                        new_p1 = p1 + int(np.ceil(diff/2.0))
                        new_p2 = p2 - int(np.floor(diff/2.0))
                    else:
                        new_p1 = p1 - int(np.floor(diff/2.0))
                        new_p2 = p2 + int(np.ceil(diff/2.0))
                    
                    # Check bounds - ONLY embed if within bounds
                    if 0 <= new_p1 <= 255 and 0 <= new_p2 <= 255:
                        stego[r, c] = new_p1
                        stego[r, c+1] = new_p2
                        idx += len(bits)
                    # If out of bounds, skip this pixel pair without embedding
                    break
                    
        return np.clip(stego, 0, 255).astype(np.uint8), "PVD-Adaptive"

    @staticmethod
    def extract(stego, key):
        # Use same random sequence as embed
        rng = random.Random(key)
        h, w = stego.shape
        coords = [(r, c) for r in range(h) for c in range(w-1)]
        rng.shuffle(coords)
        
        bin_msg = ""
        
        for r, c in coords:
            p1, p2 = int(stego[r, c]), int(stego[r, c+1])
            d = abs(p1 - p2)
            
            # Find range for this difference
            for range_idx in range(len(PVD.ranges)-1):
                if PVD.ranges[range_idx] <= d < PVD.ranges[range_idx+1]:
                    low = PVD.ranges[range_idx]
                    high = PVD.ranges[range_idx+1] - 1
                    t = int(np.log2(high - low + 1))
                    
                    # Extract bits
                    b = d - low
                    
                    # Check if this pair COULD have been embedded (same bounds check as embed)
                    # We need to verify the original pixels could have produced this d_new
                    # Reconstruct what new pixels would be if we embedded b
                    d_new = low + b
                    diff = d_new - d
                    
                    # Calculate what the adjusted pixels would be (same logic as embed)
                    if p1 >= p2:
                        new_p1_check = p1 + int(np.ceil(diff/2.0))
                        new_p2_check = p2 - int(np.floor(diff/2.0))
                    else:
                        new_p1_check = p1 - int(np.floor(diff/2.0))
                        new_p2_check = p2 + int(np.ceil(diff/2.0))
                    
                    # Only extract if bounds would have been OK during embed
                    # Actually, we can't do this check because we're looking at STEGO not COVER
                    # So we just extract the bits as is
                    bin_msg += format(b, f'0{t}b')
                    break
                    
            # Check for terminator (3 NULL bytes = 24 zero bits)
            if len(bin_msg) >= 24:
                # Look for NULL terminator pattern
                for i in range(len(bin_msg) - 23):
                    if bin_msg[i:i+24] == "000000000000000000000000":
                        bin_msg = bin_msg[:i+24]
                        break
                else:
                    # Check if last 24 bits are all zero
                    if bin_msg[-24:] == "000000000000000000000000":
                        break
        
        # Convert binary to string
        byte_list = []
        for i in range(0, len(bin_msg), 8):
            byte_str = bin_msg[i:i+8]
            if len(byte_str) < 8:
                break
            byte_val = int(byte_str, 2)
            if byte_val == 0:  # Stop at NULL
                break
            byte_list.append(byte_val)
        
        try:
            return bytes(byte_list).decode('utf-8')
        except:
            return bytes(byte_list).decode('utf-8', errors='ignore')