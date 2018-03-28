import numpy as np
import cv2
from matplotlib import pyplot as plt
     
imgL = cv2.imread('/home/pi/image2.jpg',0)
imgR = cv2.imread('/home/pi/image3.jpg',0)
     
stereo = cv2.StereoBM(1, 16, 15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()
