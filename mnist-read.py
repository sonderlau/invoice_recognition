import numpy as np

with open("./mnist/train-images.idx3-ubyte", 'rb') as imgpath:

    train_images = np.fromfile(imgpath, dtype=np.uint8).reshape(60000, 28*28)
    
    print(train_images.shape)

with open("./mnist/train-images.idx3-ubyte", 'rb') as imgpath:

    train_images = np.fromfile(imgpath, dtype=np.uint8).reshape(60000, 28*28)
    
    print(train_images.shape)