import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')
primary_directories = ['test','train']
secondary_directories = ['greyscale','binary']
target_directories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  
cwd = os.getcwd()
save_path = f'{cwd}\\processed_images\\'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

for dir1 in primary_directories:
    for dir2 in target_directories:
        path = os.path.realpath(f'Data/{dir1}/{dir2}/')
        save_path = f'{cwd}\\processed_images\\{dir1}\\'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        for filename in os.listdir(path):

            #Read in Image
            filepath = f"{path}\\{filename}"
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            # # img = cv2.fastNlMeansDenoising(img,None,10,7,21)

            # # Apply blur for thresholding
            # blur = cv2.GaussianBlur(img,(3,3), 150)

            # # find otsu's threshold value with OpenCV function
            # _, img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

            # img = cv2.erode(img, (7,7), iterations=3)

            # #Resize Image
            # desired_size = 28
            # old_size = img.shape[:2] # old_size is in (height, width) format

            # ratio = float(desired_size)/max(old_size)
            # new_size = tuple([int(x*ratio) for x in old_size])

            # img = cv2.resize(img, (new_size[1], new_size[0]))

            # delta_w = desired_size - new_size[1]
            # delta_h = desired_size - new_size[0]
            # top, bottom = delta_h//2, delta_h-(delta_h//2)
            # left, right = delta_w//2, delta_w-(delta_w//2)

            # color = [255, 255, 255]
            # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

            # Save image
            cv2.imwrite(os.path.join(save_path, filename), img)