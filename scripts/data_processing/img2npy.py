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
    images = os.listdir(data_path + "/" + dir + "/img")
    images.sort()
    i = 0
    for im in images:
        i += 1
        image_data = cv2.imread(data_path + "/" + dir + "/img/" + im)
        
        # 画像を96x96にリサイズ
        image_data = cv2.resize(image_data, (W, H))
        
        image_data = image_data / 255.0
        if i == 1:
            image_list = image_data.reshape(1, H, W, C)
        else:
            image_list = np.append(image_list, image_data.reshape(1, H, W, C), axis=0)
        print(image_list.shape)
        
    np.save(data_path + "/" + dir + "/image.npy", image_list)
