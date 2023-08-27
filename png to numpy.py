from PIL import Image
import os, sys
import cv2
import numpy as np

'''
Converts all images in a directory to '.npy' format.
Use np.save and np.load to save and load the images.
Use it for training your neural networks in ML/DL projects. 
'''

# Path to image directory
path = "./data for train/PRISM copy/train"
dirs = os.listdir(path)
dirs.sort()
x_train = []


def load_dataset():
    # Append images to a list
    for item in dirs:
        if os.path.isfile(path + item):
            im = Image.open(path + item).convert("RGB")
            im = np.array(im)
            x_train.append(im)


if __name__ == "__main__":
    load_dataset()

    # Convert and save the list of images in '.npy' format
    imgset = np.array(x_train)
    np.save("imgds.npy", imgset)