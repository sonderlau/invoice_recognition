from preset_analyse import get_preset
from calculate_difference import leven
from PIL import Image
import cv2
import numpy as np
from icecream import ic

store = get_preset()

im = Image.open("./data/test2.png")

print(im.size)

# 发票代码
code = im.crop((850, 15, 1180, 155))

code.save("./out/cut.png")


img = cv2.imread("./out/cut.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("./out/gray.jpg", gray)

# 阈值分割
ret, thresh = cv2.threshold(gray, 120, 255, 1)

cv2.imwrite("./out/thresh.jpg", thresh)

contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# print(contours)

for i in range(0, len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    # print((x, y), (x + w, y + h))
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 1)


    if h*w >= 75:
        newimage = thresh[y : y + h , x  : x + w ]
        new_image_array = np.array(newimage)
        
        # 比较
        
        init = -1
        index = -1
        
        for k, v in store.items():
            rsp = leven(new_image_array, v)
            ic(rsp, v)
            if rsp >= init:
                index = k
                init = rsp
        
        print("result", index)
        cv2.imwrite("./result/" + str(i) + "- "+  str(index) + ".jpg", img[y : y + h , x  : x + w ])
        
        
        



