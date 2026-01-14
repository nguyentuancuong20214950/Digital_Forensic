import numpy as np

class PVD:
    ranges = [0, 8, 16, 32, 64, 128, 256]

    @staticmethod
    def embed(image, message, key):
        bin_msg = ''.join([format(ord(i), "08b") for i in message]) + "00000000"
        stego = image.copy().astype(np.int32)
        flat = stego.flatten()
        idx = 0
        
        for i in range(0, len(flat)-1, 2):
            if idx >= len(bin_msg): 
                break
            
            p1, p2 = int(flat[i]), int(flat[i+1])
            d = abs(p1 - p2)
            
            # Find appropriate range
            for r in range(len(PVD.ranges)-1):
                if PVD.ranges[r] <= d < PVD.ranges[r+1]:
                    low, high = PVD.ranges[r], PVD.ranges[r+1]-1
                    t = int(np.log2(high - low + 1))
                    
                    if idx + t > len(bin_msg):
                        break  # Not enough bits
                    
                    bits = bin_msg[idx : idx+t]
                    b = int(bits, 2)
                    d_new = low + b
                    diff = d_new - d
                    
                    # Calculate new pixel values
                    if p1 >= p2:
                        p1_new = p1 + int(np.ceil(diff/2))
                        p2_new = p2 - int(np.floor(diff/2))
                    else:
                        p1_new = p1 - int(np.floor(diff/2))
                        p2_new = p2 + int(np.ceil(diff/2))
                    
                    # Check for overflow - only embed if safe
                    if 0 <= p1_new <= 255 and 0 <= p2_new <= 255:
                        flat[i] = p1_new
                        flat[i+1] = p2_new
                        idx += t
                    # If overflow would occur, skip this pair
                    break
        
        return flat.reshape(image.shape).astype(np.uint8), "PVD-Adaptive"

    @staticmethod
    def extract(stego, key):
        flat = stego.flatten().astype(np.int32)
        bin_msg = ""
        
        for i in range(0, len(flat)-1, 2):
            p1, p2 = int(flat[i]), int(flat[i+1])
            d = abs(p1 - p2)
            
            # Find range
            for r in range(len(PVD.ranges)-1):
                if PVD.ranges[r] <= d < PVD.ranges[r+1]:
                    low, high = PVD.ranges[r], PVD.ranges[r+1]-1
                    t = int(np.log2(high - low + 1))
                    b = d - low
                    bin_msg += format(b, f'0{t}b')
                    break
            
            # Check for sentinel (3 NULL bytes = 24 zero bits)
            if len(bin_msg) >= 24 and bin_msg[-24:] == "0" * 24:
                break
        
        return PVD.bin_to_msg(bin_msg)

    @staticmethod
    def bin_to_msg(bin_data):
        chars = ""
        for i in range(0, len(bin_data), 8):
            byte = bin_data[i:i+8]
            if len(byte) < 8: break
            code = int(byte, 2)
            if code == 0: break
            chars += chr(code)
        return chars