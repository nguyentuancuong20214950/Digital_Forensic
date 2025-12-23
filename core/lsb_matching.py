import numpy as np
import random

class LSB_Matching:
    @staticmethod
    def embed(image, message, key):
        # 1. Chuyển tin nhắn sang bit chuẩn (UTF-8)
        data = message.encode('utf-8') + b'\x00\x00\x00' # Tăng thêm byte NULL để an toàn
        bin_msg = ''.join([format(b, "08b") for b in data])
        
        h, w = image.shape
        total_pixels = h * w
        
        # 2. Tính số bit cần nhúng (n_bits)
        n_bits = int(np.ceil(len(bin_msg) / total_pixels))
        n_bits = min(max(n_bits, 1), 4)

        # 3. TẠO 2 BỘ NGẪU NHIÊN RIÊNG BIỆT (Cực kỳ quan trọng)
        # Bộ 1: Chỉ dùng để Shuffle tọa độ
        rng_coords = random.Random(key)
        all_coords = [(r, c) for r in range(h) for c in range(w)]
        rng_coords.shuffle(all_coords)
        
        # Bộ 2: Chỉ dùng để chọn cộng hoặc trừ (Matching)
        # Việc này giúp khi Extract (không gọi bộ này), bộ Shuffle vẫn không bị lệch
        rng_matching = random.Random(key)

        stego = image.copy().astype(np.int16)
        msg_idx = 0
        mod_val = 2**n_bits

        for r, c in all_coords:
            if msg_idx >= len(bin_msg): break
            
            bits = bin_msg[msg_idx : msg_idx + n_bits]
            actual_n = len(bits)
            target_val = int(bits.ljust(actual_n, '0'), 2)
            
            # Tính giá trị hiện tại theo modulo
            current_mod = stego[r, c] % (2**actual_n)
            
            if current_mod != target_val:
                # Thực hiện Matching (+/- 1)
                if stego[r, c] == 0:
                    stego[r, c] += 1
                elif stego[r, c] == 255:
                    stego[r, c] -= 1
                else:
                    # Dùng bộ ngẫu nhiên riêng để không làm hỏng thứ tự Shuffle
                    stego[r, c] += rng_matching.choice([1, -1])
                
                # Sau khi +/- 1, nếu vẫn chưa khớp mod (do toán học), ép bit bằng Sub
                if stego[r, c] % (2**actual_n) != target_val:
                    stego[r, c] = (stego[r, c] // (2**actual_n)) * (2**actual_n) + target_val
            
            stego[r, c] = np.clip(stego[r, c], 0, 255)
            msg_idx += actual_n
            
        return stego.astype(np.uint8), n_bits

    @staticmethod
    def extract(stego, key, n_bits=1):
        h, w = stego.shape
        # Dùng bộ ngẫu nhiên tọa độ giống hệt lúc nhúng
        rng_coords = random.Random(key)
        all_coords = [(r, c) for r in range(h) for c in range(w)]
        rng_coords.shuffle(all_coords)
        
        bin_msg = ""
        mod_val = 2**n_bits
        
        for r, c in all_coords:
            val = int(stego[r, c]) % mod_val
            bin_msg += format(val, f'0{n_bits}b')
            
            # Kiểm tra kết thúc sớm (2 bytes NULL liên tiếp)
            if len(bin_msg) % 8 == 0 and "0000000000000000" in bin_msg[-32:]:
                break
        
        # Chuyển bit sang byte và decode
        byte_list = []
        for i in range(0, len(bin_msg), 8):
            byte_str = bin_msg[i:i+8]
            if len(byte_str) < 8: break
            b_val = int(byte_str, 2)
            if b_val == 0: break
            byte_list.append(b_val)
            
        return bytes(byte_list).decode('utf-8', errors='ignore')