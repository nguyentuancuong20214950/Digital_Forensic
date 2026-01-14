import numpy as np
import random

MAX_BITS = 4

class Interpolation:
    @staticmethod
    def downscale(image):
        """Downscale by 2 using block 2X2 averaging."""
        h, w = image.shape
        H, W = h // 2, w // 2
        small = np.zeros((H, W), dtype=np.uint8)
        for i in range(H):
            for j in range(W):
                block = image[2*i:2*i+2, 2*j:2*j+2]
                small[i, j] = int(round(block.mean()))
        return small

    @staticmethod
    def upscale(small, target_shape):
        """
        Upsample small (h/2 x w/2) back to target_shape (h x w) using Neighbor Mean Interpolation (NMI).
        Anchors (original small pixels) are placed at even indices (0,2,4,...).
        Interpolated pixels are computed as neighbor means.
        """
        Ht, Wt = target_shape
        up = np.zeros((Ht, Wt), dtype=np.uint8)

        sh, sw = small.shape
        # place anchors at even indices
        for i in range(sh):
            for j in range(sw):
                up[2*i, 2*j] = small[i, j]

        # horizontal (even row, odd col)
        for i in range(0, Ht, 2):
            for j in range(1, Wt, 2):
                left = int(up[i, j-1])
                right = int(up[i, j+1]) if j+1 < Wt else left
                up[i, j] = np.uint8(round((left + right) / 2))

        # vertical (odd row, even col)
        for i in range(1, Ht, 2):
            for j in range(0, Wt, 2):
                top = int(up[i-1, j])
                bottom = int(up[i+1, j]) if i+1 < Ht else top
                up[i, j] = np.uint8(round((top + bottom) / 2))

        # center (odd row, odd col)
        for i in range(1, Ht, 2):
            for j in range(1, Wt, 2):
                tl = int(up[i-1, j-1])
                tr = int(up[i-1, j+1]) if j+1 < Wt else tl
                bl = int(up[i+1, j-1]) if i+1 < Ht else tl
                br = int(up[i+1, j+1]) if (i+1 < Ht and j+1 < Wt) else tl
                up[i, j] = np.uint8(round((tl + tr + bl + br) / 4))

        return up

    @staticmethod
    def _anchor_coords_for_shape(shape):
        """Return set of anchor coordinates (even indices) for shape produced by downscale"""
        H, W = shape
        anchors = set()
        for i in range(0, H, 2):
            for j in range(0, W, 2):
                anchors.add((i, j))
        return anchors

    @staticmethod
    def embed(image, message, key):
        """
        Embed message into image.
        Signature: embed(image, message, key)
        Returns: (stego_image, n_bits)
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        h, w = image.shape

        # 1. Downscale by 2 and upsample back with NMI to create deterministic cover
        small = Interpolation.downscale(image)
        cover = Interpolation.upscale(small, (h, w))

        # 2. Prepare message bits with 3 NULL bytes sentinel
        data = message.encode('utf-8') + b'\x00\x00\x00'
        bin_msg = ''.join(format(b, '08b') for b in data)

        # 3. Determine embeddable positions (non-anchor pixels)
        anchors = Interpolation._anchor_coords_for_shape((h, w))
        embeddable = [(i, j) for i in range(h) for j in range(w) if (i, j) not in anchors]

        total_pixels = len(embeddable)
        # 4. Decide n_bits (bounded by MAX_BITS)
        n_bits = int(np.ceil(len(bin_msg) / max(1, total_pixels)))
        n_bits = max(1, min(n_bits, MAX_BITS))

        # 5. Pad bin_msg to multiple of n_bits
        while len(bin_msg) % n_bits != 0:
            bin_msg += '0'

        # 6. Shuffle embeddable positions deterministically with key
        rng = random.Random(key)
        rng.shuffle(embeddable)

        stego = cover.copy()
        idx = 0
        for (i, j) in embeddable:
            if idx >= len(bin_msg):
                break
            bits = bin_msg[idx: idx + n_bits]
            bit_val = int(bits, 2)
            e = int(cover[i, j])  # expectation from NMI cover
            new_val = int(np.clip(e + bit_val, 0, 255))
            stego[i, j] = np.uint8(new_val)
            idx += n_bits

        return stego, n_bits

    @staticmethod
    def extract(stego, key, n_bits=1):
        """
        Extract message from stego image.
        Signature: extract(stego, key, n_bits)
        Returns: decoded string (UTF-8, errors ignored if needed)
        """
        if stego.dtype != np.uint8:
            stego = stego.astype(np.uint8)
        h, w = stego.shape

        # 1. Extract anchors from stego to recreate small image
        small = np.zeros((h // 2, w // 2), dtype=np.uint8)
        for si in range(h // 2):
            for sj in range(w // 2):
                small[si, sj] = stego[si * 2, sj * 2]
        
        # 2. Upscale to get expected cover
        cover = Interpolation.upscale(small, (h, w))

        # 3. Recompute embeddable positions with same logic as embed
        anchors = Interpolation._anchor_coords_for_shape((h, w))
        embeddable = [(i, j) for i in range(h) for j in range(w) if (i, j) not in anchors]
        
        # 4. Shuffle with same key
        rng = random.Random(key)
        rng.shuffle(embeddable)

        bin_msg = ""
        base = 1 << n_bits
        for (i, j) in embeddable:
            delta = int(stego[i, j]) - int(cover[i, j])
            # clamp delta to valid range
            delta = max(0, min(delta, base - 1))
            bin_msg += format(delta, f'0{n_bits}b')
            # sentinel check: 3 NULL bytes (24 zeros) within last 48 bits
            if len(bin_msg) % 8 == 0 and "000000000000000000000000" in bin_msg[-48:]:
                break

        # 3. Convert bits to bytes and stop at first NULL byte
        byte_list = []
        for k in range(0, len(bin_msg), 8):
            bstr = bin_msg[k:k+8]
            if len(bstr) < 8:
                break
            val = int(bstr, 2)
            if val == 0:
                break
            byte_list.append(val)

        try:
            return bytes(byte_list).decode('utf-8')
        except Exception:
            return bytes(byte_list).decode('utf-8', errors='ignore')
