
# import numpy as np
# import cv2
# import torch
# images_cv = cv2.imread('dog_manga2.png')
# images = np.array(images_cv)
# print(images_cv.shape,images.shape)
# img = torch.from_numpy(images).permute(2,0,1).unsqueeze(0)
# img.shape

# dim = (32, 32)
# resized = cv2.resize(images_cv, dim)


# de = 0 

import os

cats_path = "./Ganglin/data/cartoon_dog/dogs"
cats = os.listdir(cats_path)

print("cats",len(cats))
