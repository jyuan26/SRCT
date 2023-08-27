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
path = "/Users/jasonyuan/ESRT 2/data for train/PRISMx4 copy/valx4/"
dirs = os.listdir(path)
dirs.sort()
x_train = []


def load_dataset():
    print('load dataset start')
    # Append images to a list
    for item in dirs:
        #print('loop')
        print(item)
        if os.path.isfile(path + item):
            print(path, item)
            im = Image.open(path + item).convert("RGB")
            im = np.array(im)
            x_train.append(im)
            print(im)

    print('load_dataset end')


if __name__ == "__main__":
    load_dataset()
    print(len(x_train))

    imgset = np.array(x_train, dtype=int)
    print('yo')
    print(len(imgset))
    for i in range(len(imgset)):
        print(i)
        np.save("imgds" + str(i) + ".npy", imgset[i])