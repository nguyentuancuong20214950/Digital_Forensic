import numpy as np

class DifferenceExpansion:
    @staticmethod
    def embed(image, message, key):
        bin_msg = ''.join([format(ord(i), "08b") for i in message]) + "00000000"
        msg_idx = 0
        h, w = image.shape
        stego = image.copy().astype(np.int32)
        flat_stego = stego.flatten()
        
        for i in range(0, len(flat_stego) - 1, 2):
            if msg_idx >= len(bin_msg):
                break
            
            x, y = int(flat_stego[i]), int(flat_stego[i+1])
            
            # Skip boundary pixels to avoid overflow
            if x <= 10 or x >= 245 or y <= 10 or y >= 245:
                continue
            
            l = (x + y) // 2
            d = x - y
            
            b = int(bin_msg[msg_idx])
            d_new = 2 * d + b
            
            x_new = l + (d_new + 1) // 2
            y_new = l - d_new // 2
            
            # Double check bounds
            if 0 <= x_new <= 255 and 0 <= y_new <= 255:
                flat_stego[i] = x_new
                flat_stego[i+1] = y_new
                msg_idx += 1

        return flat_stego.reshape(image.shape).astype(np.uint8), 1

    @staticmethod
    def extract(stego_image, key, layers=1):
        flat = stego_image.flatten().astype(np.int32)
        bin_msg = ""
        
        for i in range(0, len(flat) - 1, 2):
            x, y = int(flat[i]), int(flat[i+1])
            
            # Skip same boundary pixels as in embed
            if x <= 10 or x >= 245 or y <= 10 or y >= 245:
                continue
            
            d_new = x - y
            
            # Extract LSB of d_new
            b = d_new % 2 if d_new >= 0 else (-d_new) % 2
            bin_msg += str(b)
            
            # Check for sentinel
            if len(bin_msg) >= 24 and bin_msg[-24:] == "0" * 24:
                break
        
        # Convert binary to string
        chars = ""
        for i in range(0, len(bin_msg), 8):
            byte_str = bin_msg[i:i+8]
            if len(byte_str) < 8: 
                break
            code = int(byte_str, 2)
            if code == 0: 
                break
            chars += chr(code)
        return chars