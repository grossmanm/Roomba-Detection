import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

path = 'C:/Users/Malcolm/Documents/GitHub/Roomba-Detection/roomba_photos/roomba'
onlyfiles = [f for f in listdir(path) if isfile(join(path,f))]
images = np.empty(len(onlyfiles),dtype=object)
for n in range(0, len(onlyfiles)):
    image = join(path,onlyfiles[n])
    images[n] = cv2.imread(image)
    print(onlyfiles[n])
    res = cv2.resize(images[n],(32,32), cv2.INTER_CUBIC)
    cv2.imshow(image,res)
    cv2.destroyAllWindows()
