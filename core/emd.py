import numpy as np

class EMD:
    @staticmethod
    def embed(image, message, key):
        """
        Embeds message using EMD (2 pixels per group, Base-5).
        """
        # --- 1. PREPARE MESSAGE (Sentinel Bit Fix) ---
        # We prepend '1' to preserve leading zeros in the binary stream.
        # Example: Message '\x00' -> Binary '00000000' -> Int 0 (ZEROS LOST!)
        # Fixed:   Message '\x00' -> Binary '100000000' -> Int 256 (ZEROS PRESERVED)
        bin_msg = '1' + ''.join([format(ord(i), "08b") for i in message])
        
        # Convert full binary string to a single integer
        val = int(bin_msg, 2)
        
        # Convert integer to Base-5 digits (reversed order for extraction efficiency)
        digits = []
        while val > 0:
            digits.append(val % 5)
            val //= 5
            
        # --- 2. PREPARE IMAGE ---
        # Flatten and squash pixels to [1, 254] to prevent overflow/underflow
        # If pixel is 255 and we add 1 -> 256 (Error). If 0 and sub 1 -> -1 (Error).
        stego = image.flatten().astype(np.int32)
        stego = np.clip(stego, 1, 254) 

        # Capacity Check (2 pixels needed per digit)
        if len(digits) > len(stego) // 2:
            raise ValueError(f"Data too large! Need {len(digits)} pixel groups, have {len(stego)//2}.")

        # --- 3. EMBEDDING CORE ---
        for i in range(len(digits)):
            idx1, idx2 = 2*i, 2*i+1
            p1, p2 = stego[idx1], stego[idx2]
            
            # EMD Extraction Function f = (p1*1 + p2*2) mod 5
            f = (p1 * 1 + p2 * 2) % 5
            
            # Calculate difference s = (secret - f) mod 5
            d = digits[i]
            s = (d - f) % 5
            
            # Modify pixels based on difference
            if s == 1: p1 += 1
            elif s == 2: p2 += 1
            elif s == 3: p2 -= 1
            elif s == 4: p1 -= 1
            
            stego[idx1], stego[idx2] = p1, p2

        # Return image as uint8 and the CRITICAL param (length of digits)
        return np.clip(stego.reshape(image.shape), 0, 255).astype(np.uint8), len(digits)

    @staticmethod
    def extract(stego, key, n_digits):
        """
        Extracts message. Requires 'n_digits' (returned from embed) to function.
        """
        if not n_digits:
            return ""

        flat = stego.flatten().astype(np.int32)
        digits = []
        
        # --- 1. EXTRACT DIGITS ---
        for i in range(n_digits):
            p1, p2 = flat[2*i], flat[2*i+1]
            # Calculate f to get the secret digit back
            f = (p1 * 1 + p2 * 2) % 5
            digits.append(f)
            
        # --- 2. RECONSTRUCT VALUE ---
        # Rebuild the large integer from Base-5 digits
        val = 0
        power = 1
        for d in digits:
            val += d * power
            power *= 5
            
        # --- 3. RECONSTRUCT BINARY (Sentinel Removal) ---
        # Convert back to binary string, removing the '0b' prefix
        raw_bin = bin(val)[2:]
        
        # REMOVE SENTINEL: The first bit must be '1' (our sentinel).
        # If it's not, the data is corrupted or parameters are wrong.
        if len(raw_bin) > 0:
            bin_msg = raw_bin[1:] # Strip the leading '1'
        else:
            return ""

        # --- 4. BINARY TO CHARS ---
        # Ensure we have valid 8-bit chunks
        # (The sentinel guarantees we don't need manual zero-padding here)
        chars = []
        for i in range(0, len(bin_msg), 8):
            byte = bin_msg[i:i+8]
            if len(byte) < 8: break # Ignore trailing fragments
            chars.append(chr(int(byte, 2)))
            
        return "".join(chars)