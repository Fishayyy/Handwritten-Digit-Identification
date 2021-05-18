import os
import cv2
import skimage
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
import warnings

warnings.filterwarnings('ignore')

target_directories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

cwd = os.getcwd()
save_path = f'{cwd}\\processed_images\\'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

for dir in target_directories:
    path = os.path.realpath(f'Data/{dir}/')

    for filename in os.listdir(path):

        # Read in Image
        filepath = f"{path}\\{filename}"
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # Apply blur for thresholding
        blur = cv2.GaussianBlur(img, (3, 3), 150)

        # Background Subtraction
        # cv2.subtract(blur,img,img)

        # find otsu's threshold value with OpenCV function
        _, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # swaps pixel values to white foreground / black background - used for binary morphological operations
        #img = cv2.bitwise_not(img)

        black = [0,0,0]
        white = [255,255,255]



        # Thinning - function says dilation but functions as erosion
        kernel = np.ones((3, 3), np.uint8)

        #img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)

        # Resize Image
        desired_size = 64
        old_size = img.shape[:2]  # old_size is in (height, width) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        img = cv2.resize(img, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        # Save image
        cv2.imwrite(os.path.join(save_path, filename), img)