import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')

target_directories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

cwd = os.getcwd()

for dir in target_directories:
    path = os.path.realpath(f'Data/{dir}/')

    for filename in os.listdir(path):

        # Read in Image
        filepath = f"{path}\\{filename}"
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # Apply blur for thresholding
        blur = cv2.GaussianBlur(img, (5, 5), 0)

        # find normalized_histogram, and its cumulative distribution function
        hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.max()
        Q = hist_norm.cumsum()

        bins = np.arange(256)

        fn_min = np.inf
        thresh = -1

        for i in range(1, 256):
            p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
            q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
            b1, b2 = np.hsplit(bins, [i])  # weights

            # finding means and variances
            m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
            v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2

            # calculates the minimization function
            fn = v1 * q1 + v2 * q2
            if fn < fn_min:
                fn_min = fn
                thresh = i

        # find otsu's threshold value with OpenCV function
        _, img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

        color = [255, 255, 255]
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)



        # Save image
        cv2.imwrite(os.path.join(path, f'{filename}_thresh.png'), img)