import numpy as np
import random

class LSB_Sub:
    @staticmethod
    def embed(image, message, key):
        # 1. Chuyển tin nhắn sang mảng Byte (Xử lý được cả Tiếng Việt/Ký tự đặc biệt từ file)
        # Thêm byte 0 (NULL) để đánh dấu kết thúc
        data = message.encode('utf-8') + b'\x00'
        bin_msg = ''.join([format(b, "08b") for b in data])
        
        h, w = image.shape
        total_pixels = h * w
        n_bits = int(np.ceil(len(bin_msg) / total_pixels))
        n_bits = min(max(n_bits, 1), 4)

        random.seed(key)
        coords = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(coords)

        stego = image.copy().astype(np.uint8)
        idx = 0
        for r, c in coords:
            if idx >= len(bin_msg): break
            
            bits = bin_msg[idx : idx + n_bits]
            actual_n = len(bits)
            val = int(stego[r, c])
            
            mask = (0xFF << actual_n) & 0xFF
            bit_val = int(bits.ljust(actual_n, '0'), 2)
            
            stego[r, c] = (val & mask) | bit_val
            idx += actual_n
            
        return stego, n_bits

    @staticmethod
    def extract(stego, key, n_bits=1):
        h, w = stego.shape
        random.seed(key)
        coords = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(coords)
        
        bin_msg = ""
        for r, c in coords:
            val = int(stego[r, c])
            bin_msg += format(val & ((1 << n_bits) - 1), f'0{n_bits}b')
            
            # Kiểm tra xem đã có byte NULL (8 bit 0) ở cuối chuỗi bit chưa
            if len(bin_msg) >= 8 and bin_msg[-8:] == "00000000":
                break
        
        # Chuyển bit thành mảng byte
        byte_list = []
        for i in range(0, len(bin_msg), 8):
            byte_str = bin_msg[i:i+8]
            if len(byte_str) < 8: break
            byte_val = int(byte_str, 2)
            if byte_val == 0: break # Kết thúc tại byte NULL
            byte_list.append(byte_val)
            
        try:
            return bytes(byte_list).decode('utf-8')
        except:
            return "Lỗi giải mã: Có thể sai khóa K hoặc file text chứa ký tự lạ."