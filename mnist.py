# %%
import numpy as np
import matplotlib.pyplot as plt
import Levenshtein

# %%
def read_idx3(filename):
    with open(filename, 'rb') as fo:
        buf = fo.read()
        
        index = 0
        header = np.frombuffer(buf, '>i', 4, index)
        
        index += header.size * header.itemsize
        data = np.frombuffer(buf, '>B', header[1] * header[2] * header[3], index).reshape(header[1], -1)
        
        return data
    
def read_idx1(filename):
    with open(filename, 'rb') as fo:
        buf = fo.read()
        
        index = 0
        header = np.frombuffer(buf, '>i', 2, index)
        
        index += header.size * header.itemsize
        data = np.frombuffer(buf, '>B', header[1], index)
        
        return data

# %%
train_labels = read_idx1("mnist/train-labels.idx1-ubyte")

train_images = read_idx3("mnist/train-images.idx3-ubyte")

print(train_labels.shape, train_images.shape)

# %%
print(train_images[0])

print(train_labels[0])

# %%
plt.subplot(121)
plt.imshow(train_images[0, :].reshape(28, -1), cmap='gray')
plt.title('train 0')

print(train_labels[0])

# %%
# 获取测试集合

test_labels = read_idx1("mnist/t10k-labels.idx1-ubyte")

test_images = read_idx3("mnist/t10k-images.idx3-ubyte")


# %%
print(test_labels[0])

plt.subplot(122)
plt.imshow(test_images[0, :].reshape(28, -1), cmap='gray')
plt.title('test 0')

# %%
print(test_images.shape)

# 使用测试集 作为预处理

from collections import defaultdict

data = defaultdict(lambda : [])

def sHash(img):
    """感知哈希

    Args:
        img ([type]): 一维 784 的数组

    Returns:
        [str]: 感知哈希
    """
    # 感知 哈希
    hash_val = ''
    avg = img.mean()
    
    for x in range(len(img)):
        if img[x] > avg:
            hash_val += '1'
        else:
            hash_val += '0'
    return hash_val

for i in range(len(test_images)):
    img = test_images[i, :]
    # 感知 哈希
    
    data[test_labels[i]].append(sHash(img))

# %%
# 使用训练集的第一张用来测试

to_test_image = train_images[0, :]

test_hash = sHash(to_test_image)

def recognize_number(to_test_image_sHash:str):
    
    result = [ 0 for i in range(10)]
    
    
    for k,v in data.items():
    # k - 数字  v - 每个数字的所有感知哈希值
    # 遍历所有的哈希并计算值
        for hash_val in v:
            leven_val = Levenshtein.ratio(to_test_image_sHash, hash_val)
            if leven_val > result[k]:
                result[k] = leven_val

    return result



# %%

result = recognize_number(test_hash)
print(max(result))

print(result.index(max(result)))

print(result)


# %%
# 使用我们自己写的图片

from PIL import Image

diy_image = Image.open('MNIST-4.jpg')


diy_arr = np.array(diy_image).flatten()

plt.subplot(122)
plt.imshow(diy_arr.reshape(28, -1), cmap='gray')
plt.title('diy 0')

diy_arr = diy_arr.flatten()
# print(sHash(diy_arr))
r = recognize_number(sHash(diy_arr))
print(max(r))

print(r.index(max(r)))

print(r)


# %%
# 测试结果准确率

statis = {}

for i in range(0, 10):
    statis[i] = {}
    
    statis[i]["correct"] = 0
    statis[i]["all"] = 0

for i in range(100):
    shash_val = sHash(train_images[i, :])
    
    r = recognize_number(shash_val)
    
    real_val = train_labels[i]
    if r.index(max(r)) == real_val:
        statis[real_val]["correct"] += 1
    
    statis[real_val]["all"] += 1



# %%
from icecream import ic



for i in range(10):
    print(i, statis[i]["correct"] / statis[i]["all"])


