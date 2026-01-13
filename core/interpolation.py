import numpy as np
import cv2


class Interpolation:
    @staticmethod
    def embed(image, message):
        # Simple LSB embedding on pixels
        pixels = image.flatten().astype(np.int32)
        bin_msg = ''.join([format(b, "08b") for b in message.encode('utf-8')]) + "00000000"
        
        for i in range(min(len(bin_msg), len(pixels))):
            bit = int(bin_msg[i])
            pixels[i] = (pixels[i] & ~1) | bit
        
        stego = pixels.reshape(image.shape)
        return stego.astype(np.uint8), len(bin_msg)

    @staticmethod
    def extract(stego, msg_len):
        pixels = stego.flatten()
        bin_msg = ""
        for i in range(min(msg_len, len(pixels))):
            bit = int(pixels[i]) & 1
            bin_msg += str(bit)
        
        # Convert binary to bytes then decode UTF-8
        byte_list = []
        for i in range(0, len(bin_msg), 8):
            byte = bin_msg[i:i+8]
            if len(byte) < 8:
                break
            byte_val = int(byte, 2)
            if byte_val == 0:  # NULL terminator
                break
            byte_list.append(byte_val)
        
        try:
            return bytes(byte_list).decode('utf-8')
        except:
            return bytes(byte_list).decode('utf-8', errors='ignore')