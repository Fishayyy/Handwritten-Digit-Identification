import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


desired_size = 368
im_path = "C:/Users/roger/OneDrive/Desktop/Full Sets/"

def loadImages(path):
    image_files = sorted([os.path.join(path,'1',file)
                              for file in os.listdir(path + "1") if
                              file.endswith('.png')])
    return image_files

def processing(data):
    for image in data:
        new_file = image.removesuffix('.png')+'_1.png'
        print(new_file)
        os.system('magick ' + image + '-strip' + new_file)
    img = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in data]

    print('Original size', img[0].shape)


def main():
    global im_path

    dataset = loadImages(im_path)

    pro = processing(dataset)


main()