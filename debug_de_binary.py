#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

message = "The quick brown fox jumps over the lazy dog"
data = message.encode('utf-8') + b'\x00\x00\x00'
bin_msg = ''.join([format(b, "08b") for b in data])

print(f"Message: {message}")
print(f"Message bytes: {data.hex()}")
print(f"Total bits: {len(bin_msg)}")
print()

# Show byte by byte
for i in range(0, len(bin_msg), 8):
    byte_str = bin_msg[i:i+8]
    byte_val = int(byte_str, 2)
    byte_num = i // 8
    if byte_val == 0:
        print(f"Byte {byte_num}: {byte_str} = 0x{byte_val:02x} (NULL - TERMINATOR)")
    else:
        print(f"Byte {byte_num}: {byte_str} = 0x{byte_val:02x} = '{chr(byte_val)}'")

# Expected to find 3 NULLs at end
print("\nLast 5 bytes:")
for i in range(len(bin_msg) - 40, len(bin_msg), 8):
    byte_str = bin_msg[i:i+8]
    byte_num = i // 8
    byte_val = int(byte_str, 2)
    print(f"Byte {byte_num}: {byte_str} = 0x{byte_val:02x}")
