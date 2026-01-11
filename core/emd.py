import numpy as np

class EMD:
    @staticmethod
    def embed(image, message, key):
        """
        Embeds a message into an image using EMD (Exploiting Modification Direction).
        n=2 pixels per group, base=5.
        """
        # 1. Convert Message to Binary
        # Convert string to binary stream. We append a null terminator to handle end-of-string safely if needed,
        # though main.py relies on the returned 'length' (len(digits)).
        bin_msg = ''.join([format(ord(i), "08b") for i in message]) + "00000000"
        
        # 2. Convert Binary to Base-5 Digits
        # We treat the whole binary as one large integer.
        val = int(bin_msg, 2)
        digits = []
        if val == 0: digits = [0]
        
        while val > 0:
            digits.append(val % 5)
            val //= 5
        # digits are now in reverse order (LSD first), which fits our extraction logic.

        # 3. Prepare Image (Overflow Protection)
        # CRITICAL FIX: Ensure pixels are in range [1, 254] BEFORE modification.
        # This prevents +1 going to 256 (overflow) or -1 going to -1 (underflow).
        stego = image.flatten().astype(np.int32)
        stego = np.clip(stego, 1, 254) 

        # Check Capacity (2 pixels needed per digit)
        if len(digits) > len(stego) // 2:
            raise ValueError(f"Message too long! Capacity: {len(stego)//2} digits, Needed: {len(digits)}")

        # 4. Embedding Loop
        for i in range(len(digits)):
            idx1, idx2 = 2*i, 2*i+1
            p1, p2 = stego[idx1], stego[idx2]
            
            # Calculate extraction function f
            f = (p1 * 1 + p2 * 2) % 5
            
            # Calculate difference s
            d = digits[i]
            s = (d - f) % 5
            
            # Modify pixels based on s (EMD Logic)
            if s == 1: p1 += 1
            elif s == 2: p2 += 1
            elif s == 3: p2 -= 1
            elif s == 4: p1 -= 1
            
            stego[idx1], stego[idx2] = p1, p2

        # 5. Return formatted image and the length of digits (needed for extraction)
        return np.clip(stego.reshape(image.shape), 0, 255).astype(np.uint8), len(digits)

    @staticmethod
    def extract(stego, key, n_digits):
        """
        Extracts the message from the EMD stego-image.
        Requires n_digits (returned by embed function) to know how much to read.
        """
        flat = stego.flatten().astype(np.int32)
        digits = []
        
        # 1. Extraction Loop
        # We must read exactly n_digits pairs
        for i in range(n_digits):
            p1, p2 = flat[2*i], flat[2*i+1]
            f = (p1 * 1 + p2 * 2) % 5
            digits.append(f)
            
        # 2. Reconstruct Integer from Base-5 Digits
        # Since embed loop did LSD first (val % 5), we reconstruct using powers of 5.
        val = 0
        power = 1
        for d in digits:
            val += d * power
            power *= 5
            
        # 3. Convert Integer to Bytes
        # Remove '0b' prefix
        bin_msg = bin(val)[2:]
        
        # Restore leading zeros to ensure byte alignment (multiple of 8)
        missing_zeros = (8 - len(bin_msg) % 8) % 8
        bin_msg = '0' * missing_zeros + bin_msg
        
        # 4. Convert Binary to String
        chars = []
        for i in range(0, len(bin_msg), 8):
            byte = bin_msg[i:i+8]
            if len(byte) < 8: break
            chars.append(chr(int(byte, 2)))
            
        full_msg = "".join(chars)
        
        # Remove the null terminator "00000000" we added during embed
        if full_msg.endswith('\x00'):
            full_msg = full_msg[:-1]
            
        return full_msg