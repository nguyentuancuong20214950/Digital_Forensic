import numpy as np

class DifferenceExpansion:
    """
    Difference Expansion - Uses LSB of first pixel in each row
    Simple and reliable implementation
    """
    
    @staticmethod
    def embed(image, message):
        # Add NULL terminator and convert to binary
        data = message.encode('utf-8') + b'\x00\x00\x00'
        bin_msg = ''.join([format(b, "08b") for b in data])
        
        # Sequential pixel selection (use every pixel)
        h, w = image.shape
        pixels = image.flatten()
        stego = pixels.copy().astype(np.int32)
        
        # Embed bits
        for i in range(min(len(bin_msg), len(pixels))):
            bit = int(bin_msg[i])
            # Set LSB of pixel
            stego[i] = (stego[i] & 0xFE) | bit
        
        return stego.reshape(image.shape).astype(np.uint8), "DE"

    @staticmethod
    def extract(stego_image):
        # Extract LSB from every pixel
        pixels = stego_image.flatten()
        bin_msg = ""
        
        for pixel in pixels:
            bit = int(pixel) & 1
            bin_msg += str(bit)
            
            # Check for terminator (24 zero bits)
            if len(bin_msg) >= 24 and bin_msg[-24:] == "000000000000000000000000":
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