import cv2
import homework2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc

IMGDIR = 'Problem2Images'

img=cv2.imread(f'{IMGDIR}/1_1.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

R=homework2.harris_response(img_gray,0.04,7)
pixels=homework2.corner_selection(R,0.01*np.max(R),5)
for x,y in pixels:
    cv2.circle(img,(y,x),1,(0,0,255),2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()