import os
import json
import base64
from PIL import Image

def encode_lsb(image_path, data_str, output_path):
    """
    Encodes a string into an image using LSB steganography.
    """
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        width, height = img.size
        pixels = list(img.getdata())
        
        # Convert data to binary
        binary_data = ''.join(format(ord(char), '08b') for char in data_str)
        # Add delimiter to mark end of data
        binary_data += '1111111111111110' 
        
        data_len = len(binary_data)
        if data_len > len(pixels) * 3:
            # print("Data too large for image")
            return False

        new_pixels = []
        data_idx = 0
        
        for pixel in pixels:
            if data_idx < data_len:
                r, g, b = pixel
                
                # Encode R
                if data_idx < data_len:
                    r = (r & ~1) | int(binary_data[data_idx])
                    data_idx += 1
                
                # Encode G
                if data_idx < data_len:
                    g = (g & ~1) | int(binary_data[data_idx])
                    data_idx += 1
                
                # Encode B
                if data_idx < data_len:
                    b = (b & ~1) | int(binary_data[data_idx])
                    data_idx += 1
                
                new_pixels.append((r, g, b))
            else:
                new_pixels.append(pixel)
        
        img.putdata(new_pixels)
        img.save(output_path, "PNG") # Lossless
        return True
    except Exception as e:
        # print(f"Stega Encode Error: {e}")
        return False

def decode_lsb(image_path):
    """
    Decodes a string from an image using LSB steganography.
    """
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        pixels = list(img.getdata())
        
        binary_data = ""
        for pixel in pixels:
            r, g, b = pixel
            binary_data += str(r & 1)
            binary_data += str(g & 1)
            binary_data += str(b & 1)

        # Split by bytes
        bytes_data = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
        
        decoded_str = ""
        for byte in bytes_data:
            if len(byte) < 8: break
            char_code = int(byte, 2)
            if char_code == 255 and len(decoded_str) > 0: # Check roughly for delimiter
                 # Refined check for delimiter '1111111111111110' logic would be better but simple char 
                 # check: 11111111 is 255 (ÿ). 
                 pass
            
            # Simple delimiter check during reconstruction
            # We look for the 16-bit delimiter manually? 
            # Or just decode until it looks readable.
            # Simplified: Stop at ÿÿ if we used that, but we used 1111111111111110
            
            decoded_str += chr(char_code)
            if decoded_str.endswith("ÿþ"): # approximate representation of 1111111111111110 -> 255, 254
                return decoded_str[:-2]
                
        return decoded_str.split("ÿþ")[0] # Safety split
        
    except Exception as e:
        return ""
