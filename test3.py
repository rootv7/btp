import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


img = cv.imread('1saveda.png',0)
r = len(img)
c = len(img[0])			

img3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,7,2)


cv.namedWindow('changed', cv.WINDOW_NORMAL)
cv.imshow('changed', img3)

cv.waitKey(0)
cv.destroyWindow('changed')

cv.imwrite('1savedc.png',img3)

votes = np.zeros((c,2),dtype=float)
lh=[0]
lv=[0]

for i in range(r):
	for j in range(c):
		if img3[i][j] == 0:
			votes[j][0]=votes[j][0] + 1
			votes[i][1]=votes[i][1] + 1

for i in range(c):
	if votes[i][0] >= 0.9*r:
		lv.append(i)

lv.append(c)

for i in range(r):
	if votes[i][1] >= 0.9*r:
		lh.append(i)

lh.append(r)

print(lh,lv)

ver = 1
l = lv 
if len(lv) < len(lh):
	ver = 0
	l = lh

imgs = []

cv.namedWindow('subcell', cv.WINDOW_NORMAL)

for i in range(len(l)-1):
	if l[i+1]-l[i] < r/50:
		continue
	if(ver == 1):
		img_tmp = img[:,l[i]:l[i+1]]
	else:
		img_tmp = img[l[i]:l[i+1],:1]
	imgs.append(img_tmp)
	cv.imshow('subcell', img_tmp)
	cv.waitKey(0)

cv.destroyWindow('subcell')

cv.namedWindow('subcell', cv.WINDOW_NORMAL)

# order = 9
# lap_ord = 3
# lap_mat = np.ones((lap_ord,lap_ord),float)
# lap_mat[1][1] = -8

imgs2 = []
for img in imgs:
	r = len(img)
	c = len(img[0])	
	# img_pad = np.zeros((r+order-1,c+order-1),dtype=float)
	# img_pad[order/2:r+order/2,order/2:c+order/2] = img
	# img_tmp = np.zeros(img.shape)
	# img_tmp = img_tmp.astype(np.uint8)

	# for i in range(r):
	# 	for j in range(c):
	# 		img_tmp[i][j] = np.median(img_pad[i:i+order,j:j+order])

	#img_tmp = cv.medianBlur(img,5)	
	# img2_pad = np.zeros((r+lap_ord-1,c+lap_ord-1),dtype=float)
	# img2_pad[lap_ord/2:r+lap_ord/2,lap_ord/2:c+lap_ord/2] = img_tmp
	# img3 = np.zeros(img_tmp.shape)

	# for i in range(r):
	# 	for j in range(c):
	# 		img3[i][j] = (lap_mat*img2_pad[i:i+lap_ord,j:j+lap_ord]).sum()
	# 		if img3[i][j] >= 10:
	# 			img3[i][j] = 0
	# 		else:
	# 			img3[i][j] = 255

	# img3 = img3.astype(np.uint8)

	#img3 = cv.medianBlur(img,3) - img_tmp

	imgs2.append(img3)
	cv.imshow('subcell', img3)
	cv.waitKey(0)
	cv.imshow('subcell', img_tmp)
	cv.waitKey(0)

cv.namedWindow('subcell', cv.WINDOW_NORMAL)


