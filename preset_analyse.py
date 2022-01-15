from fileinput import filename
from PIL import Image 
import numpy as np
import cv2
import os
from icecream import ic

preset_path = "./preset"

store = {}

def get_preset():
    for root, dirs, files in os.walk(preset_path):
        for f in files:
            file_full_path = os.path.join(root, f)
            image = Image.open(file_full_path) # 用PIL中的Image.open打开图像
            
            arr = np.array(image) # 转化成numpy数组
            
            file_ext = f.rsplit('.', maxsplit=1)
            file_name = file_ext[0]
            
            avg = arr.mean()
            
            hash_val = ''
            
            
            for x in range(arr.shape[0]):
                for y in range(arr.shape[1]):
                    if arr[x,y] > avg:
                        hash_val += '1'
                    else:
                        hash_val += '0'
            
            store[file_name] = hash_val
    
    return store


        


