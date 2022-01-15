import numpy as np
import Levenshtein

def leven(raw_image, string:str):
    
    
    avg = raw_image.mean()
            
    hash_val = ''
    
    for x in range(raw_image.shape[0]):
        for y in range(raw_image.shape[1]):
            if raw_image[x,y] > avg:
                hash_val += '1'
            else:
                hash_val += '0'
    
    return Levenshtein.ratio(hash_val, string)
