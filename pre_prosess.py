from PIL import Image, ImageFilter
import cv2
import numpy as np
from icecream import ic

im = Image.open("./data/test1.png")

print(im.size)

# 发票代码
code = im.crop((850, 15, 1180, 155))

code.save("cut.png")


img = cv2.imread("cut.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("gray.jpg", gray)

# 阈值分割
ret, thresh = cv2.threshold(gray, 95, 255, 1)

cv2.imwrite("thresh.jpg", thresh)

contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(contours)

for i in range(0, len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    print((x, y), (x + w, y + h))
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 1)

    # 存储截图
    if h*w >= 75:
        newimage = thresh[y : y + h , x  : x + w ]
        cv2.imwrite("./rec/" + str(i) + ".jpg", newimage)

cv2.imwrite("save.jpg", img)

