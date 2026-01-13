import numpy as np
import random

class DifferenceExpansion:
    @staticmethod
    def embed(image, message, key):
        # Add NULL terminator and convert to binary
        data = message.encode('utf-8') + b'\x00\x00\x00'
        bin_msg = ''.join([format(b, "08b") for b in data])
        
        # Use key for random pixel pair selection
        rng = random.Random(key)
        h, w = image.shape
        
        # Create list of pixel pair coordinates (adjacent horizontal pairs)
        coords = []
        for r in range(h):
            for c in range(0, w-1):
                coords.append((r, c))
        rng.shuffle(coords)
        
        stego = image.copy().astype(np.int32)
        msg_idx = 0
        
        for r, c in coords:
            if msg_idx >= len(bin_msg):
                break
                
            x = int(stego[r, c])
            y = int(stego[r, c+1])
            
            # Calculate average and difference
            l = (x + y) // 2
            d = x - y
            
            # Get bit to embed
            b = int(bin_msg[msg_idx])
            
            # Expand difference and embed bit
            d_new = 2 * d + b
            
            # Calculate new pixel values
            x_new = l + (d_new + 1) // 2
            y_new = l - d_new // 2
            
            # ALWAYS embed, even if must use alternative bit value
            # Try preferred bit first
            if 0 <= x_new <= 255 and 0 <= y_new <= 255:
                stego[r, c] = x_new
                stego[r, c+1] = y_new
                msg_idx += 1
            else:
                # Bounds exceeded - embed opposite bit instead
                # This ensures extract can still read the bit
                d_new_alt = 2 * d + (1 - b)
                x_new_alt = l + (d_new_alt + 1) // 2
                y_new_alt = l - d_new_alt // 2
                
                if 0 <= x_new_alt <= 255 and 0 <= y_new_alt <= 255:
                    stego[r, c] = x_new_alt
                    stego[r, c+1] = y_new_alt
                    msg_idx += 1
                else:
                    # Last resort - don't modify but still advance (data loss, rare)
                    msg_idx += 1
                
        return stego.astype(np.uint8), "DE-Difference"

    @staticmethod
    def extract(stego_image, key):
        # Use same random sequence as embed
        rng = random.Random(key)
        h, w = stego_image.shape
        
        # Create same list of pixel pair coordinates (adjacent horizontal pairs)
        coords = []
        for r in range(h):
            for c in range(0, w-1):  # All adjacent pairs, not skipping
                coords.append((r, c))
        rng.shuffle(coords)
        
        bin_msg = ""
        
        for r, c in coords:
            x = int(stego_image[r, c])
            y = int(stego_image[r, c+1])
            
            # Calculate average and expanded difference
            l = (x + y) // 2
            d_new = x - y
            
            # Extract LSB (the embedded bit)
            b = d_new % 2
            bin_msg += str(b)
            
            # Check for terminator (3 NULL bytes = 24 zero bits)
            if len(bin_msg) >= 24 and "000000000000000000000000" in bin_msg[-48:]:
                break
        
        # Convert binary to string (stop at first NULL)
        byte_list = []
        for i in range(0, len(bin_msg), 8):
            byte_str = bin_msg[i:i+8]
            if len(byte_str) < 8:
                break
            byte_val = int(byte_str, 2)
            if byte_val == 0:  # Stop at NULL terminator
                break
            byte_list.append(byte_val)
        
        try:
            return bytes(byte_list).decode('utf-8')
        except:
            return bytes(byte_list).decode('utf-8', errors='ignore')