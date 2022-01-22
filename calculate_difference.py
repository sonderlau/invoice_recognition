import numpy as np
import Levenshtein

def leven(raw_image, string:str):
    """由 Levenshtein 算法计算相似度

    Args:
        raw_image (numpy.array): 图片数据的矩阵
        string (str): 另外一张图片的感知哈希字符串

    Returns:
        float: 相似度
    """
    
    avg = raw_image.mean()
            
    hash_val = ''
    
    for x in range(raw_image.shape[0]):
        for y in range(raw_image.shape[1]):
            if raw_image[x,y] > avg:
                hash_val += '1'
            else:
                hash_val += '0'
    
    return Levenshtein.ratio(hash_val, string)
