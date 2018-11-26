import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# mouse callback function
def draw_circle(event,x,y,flags,param):
	if event == cv.EVENT_LBUTTONDBLCLK:
		cv.circle(img,(x,y),100,(255,0,0),-1)

img = cv.imread('1.tif',0)
cv.namedWindow('original', cv.WINDOW_NORMAL)
cv.setMouseCallback('original',draw_circle)

hist = cv.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist)
plt.show()

while 1:
	cv.imshow('original',img) 	
	if cv.waitKey(100) & 0xFF == 27:
		break 

cv.destroyAllWindows()
cv.imwrite('1saved.png',img)