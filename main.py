import cv2
import homework2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc

IMGDIR = 'Problem2Images'

img1=cv2.imread(f'{IMGDIR}/3_1.jpg')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2=cv2.imread(f'{IMGDIR}/3_2.jpg')
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

