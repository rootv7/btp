import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('2.tif',0)
r = len(img)
c = len(img[0])
# cv.namedWindow('original', cv.WINDOW_NORMAL)
# cv.imshow('original', img)

# cv.waitKey(0)
# cv.destroyWindow('original')

hist = cv.calcHist([img],[0],None,[256],[0,256])
# plt.plot(hist)
# plt.show()


i = 1
maxv = 0
while i < len(hist):
	while hist[i] >= hist[i-1] and i < len(hist):
		if hist[i] > maxv:
			maxv = hist[i] 
		i = i + 1
	while hist[i] <= hist[i-1] and i < len(hist):
		i = i + 1
	thresh = i-1
	if hist[thresh] < maxv/2:
		break

print(thresh)

# for i in range(len(img)):
# 	for j in range((len(img[0]))):
# 		if(img[i][j] < thresh):
# 			img[i][j] = 0
# 		else:
# 			img[i][j] = 255

ret,img1 = cv.threshold(img,thresh,255,cv.THRESH_BINARY)

# cv.namedWindow('changed', cv.WINDOW_NORMAL)
# cv.imshow('changed', img1)

# cv.waitKey(0)
# cv.destroyWindow('changed')

# hist = cv.calcHist([img1],[0],None,[256],[0,256])
# plt.plot(hist)
# plt.show()

output = cv.connectedComponentsWithStats(img1, 4, cv.CV_32S)


max_val = 0
max_label = 1 
for i in range(len(output[2])):
	if i == 0:
		continue
	if output[2][i][cv.CC_STAT_AREA] > max_val:
		max_val = output[2][i][cv.CC_STAT_AREA]
		max_label = i;
print(output[2][i])

mins = [r+c,1+c,r+1,2]
#corners = np.zeros((4,2), dtype=int)
corners1 = np.zeros((4,2), dtype=int)
#corners2 = np.zeros((4,2), dtype=int)

for i in range(len(img1)):
	for j in range((len(img1[0]))):
 		if(output[1][i][j] == max_label):
 			img1[i][j] = 255
 			if i+j < mins[0]:
 				mins[0] = i+j
 				corners1[0][0]=j
 				corners1[0][1]=i
 			elif i+j == mins[0]:
 				corners1[0][0]=j
 				#corners1[0][1]=i
 			if 0-i+j < mins[1]:
 				mins[1] = 0-i+j
 				corners1[1][0]=j
 				corners1[1][1]=i
 			elif 0-i+j == mins[1]:
 				#corners1[1][0]=j
 				corners1[1][1]=i
 			if i-j < mins[2]:
 				mins[2] = i-j
 				corners1[2][0]=j
 				corners1[2][1]=i
 			elif i-j == mins[2]:
 				corners1[2][0]=j
 				#corners2[2][1]=i
 			if 0-i-j < mins[3]:
 				mins[3] = 0-i-j
 				corners1[3][0]=j
 				corners1[3][1]=i
 			elif 0-i-j == mins[3]:
 				#corners2[3][0]=j
 				corners1[3][1]=i
 		else:
 			img1[i][j] = 0

corners = corners1
print(corners)
height = max(corners[1][1]-corners[0][1],corners[3][1]-corners[2][1])
width = max(corners[2][0]-corners[0][0],corners[3][0]-corners[1][0])

pts1 = np.float32(corners)
pts2 = np.float32([[0,0],[0,height],[width,0],[width,height]])


M = cv.getPerspectiveTransform(pts1,pts2)

img2 = cv.warpPerspective(img,M,(width,height))

cv.namedWindow('changed', cv.WINDOW_NORMAL)
cv.imshow('changed', img2)

cv.waitKey(0)
cv.destroyWindow('changed')

rowc = 4
colc = 8

sr = 1
sc = 1

wc = width/colc
hc = height/rowc

cell = img2[(sr-1)*hc:sr*hc,(sc-1)*wc:sc*wc]

cv.namedWindow('changed', cv.WINDOW_NORMAL)
cv.imshow('changed', cell)

cv.waitKey(0)
cv.destroyWindow('changed')

cv.imwrite('2saveda.png',cell)