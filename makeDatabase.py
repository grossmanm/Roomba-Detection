import cv2
from os import listdir
from os.path import isfile, join
import numpy as np

# File converts Roomba images in 32x32 and puts them into a .bin file
# in similar format to CIFAR 10 dataset so roomba dataset can be read
# in tensorflow
def resize(path):

    """

    :param path: a path to a folder containing images
    :return: distortedImages: a list of nparray images that have been changed to 32x32
    """
    distortedImages = []
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        image = join(path, onlyfiles[n])
        images[n] = cv2.imread(image)
        res = cv2.resize(images[n], (32, 32), cv2.INTER_CUBIC)
        distortedImages.append(res)
    return distortedImages

def makeDatabase(images):
    """
    Creates a .bin file as a dataset for tensorflow processing
    :param images: a list of nparray images
    """
    out1 = np.empty(0, dtype=int)
    for i in images:
        r = i[:, :, 0].flatten()
        g = i[:, :, 1].flatten()
        b = i[:, :, 2].flatten()
        label = [1]
        out2 = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
        out1 = np.concatenate([out1,out2])
    out1.tofile("Roomba.bin")
def main():
    path = 'C:/Users/Malcolm/Documents/GitHub/Roomba-Detection/roomba_photos/roomba'
    images = resize(path)
    makeDatabase(images)
if __name__ == "__main__":
    main()