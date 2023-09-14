import cv2
import os
import numpy as np

data_path = "../../data/sample_img_data"

dirs = os.listdir(data_path)
dirs.sort()
print(dirs)

H = 96
W = 96
C = 3

for dir in dirs:
    images = os.listdir(data_path + "/" + dir + "/Img")
    images.sort()
    i = 0
    for im in images:
        i += 1
        image_data = np.array(cv2.imread(data_path + "/" + dir+ "/Img/" + im))
        image_data = image_data/255.0
        if i == 1:
            image_list = image_data.reshape(1, H, W, C)
        else:
            image_list = np.append(image_list, image_data).reshape(i, H, W, C)
        print(image_list.shape)
        
    np.save(data_path + "/" + dir + "/image.npy", image_list)