import cv2
import json
import numpy as np


# 把narray的图片存成json
img = cv2.imread("01.jpg")
img_list = img.tolist()
print(len(img_list))
print(len(img_list[0]))
print(len(img_list[0][0]))

img_dict = {}
img_dict['name'] = "test-image"
img_dict['img'] = img_list


json_data = json.dump(img_dict, open("test.txt", "w"))
# with open("test.txt", "w") as f:
#     f.write(json_data)

# 从json中读取图片成narray形式
img_dict = json.load(open("test.txt"))
name = img_dict['name']
img = np.asarray(img_dict['img'])
print(name)
cv2.imwrite("test.jpg", img)