import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle

# File converts Roomba images in 32x32 and puts them into a .bin file
# in similar format to CIFAR 10 dataset so roomba dataset can be read
# in tensorflow
def makeDatabase(path):

    """
    Takes a path to a folder containing images, resizes those images
    to 32x32 images and puts them all into a .bin file in the same way
    as the CIFAR-10 dataset.
    :param path: a path to a folder containing images
    """
    out = []
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for n in range(0, len(onlyfiles)):
        im = join(path, onlyfiles[n])
        im = cv2.imread(im)
        im = cv2.resize(im, (32, 32), cv2.INTER_CUBIC)
        im = (np.array(im))
        r = im[:, :, 0].flatten()
        g = im[:, :, 1].flatten()
        b = im[:, :, 2].flatten()
        label = [1]
        out = np.append(out, np.array(list(label)+ list(r) + list(g) + list(b), np.uint8))
    with open('roomba.bin', 'wb') as fout:
        pickle.dump(out, fout)
def main():
    path = 'C:/Users/Malcolm/Documents/GitHub/Roomba-Detection/roomba_photos/roomba'
    makeDatabase(path)

if __name__ == "__main__":
    main()