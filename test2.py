import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img2 = cv.imread('2saved.png',0)
height = len(img2)
width = len(img2[0])

print(height,width)

rowc = 4
colc = 8

sr = 1
sc = 1

wc = width/colc
hc = height/rowc

#print(argelmax(hist,order=3))

#hist = np.convolve(hist,[0.5,0,0.5])
order = 9

blur_mat = np.ones((order,order),float)/(order*order)
img3_pad = np.zeros((len(img2)+order-1,len(img2[0])+order-1),dtype=float)
img3_pad[order/2:len(img2)+order/2,order/2:len(img2[0])+order/2] = img2

img4 = img2

for i in range(len(img2)):
	for j in range((len(img2[0]))):
		img4[i][j] = (blur_mat*img3_pad[i:i+order,j:j+order]).sum()

hist = cv.calcHist([img4],[0],None,[256],[0,256])/10
hist = [round(x/10,0)*10 for x in hist]
plt.plot(hist)
plt.show()


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

ret,img3 = cv.threshold(img4,thresh,255,cv.THRESH_BINARY_INV)
#img3 = np.convolve(img3,[[0.125,0.125,0.125],[0.125,0,0.125],[0.125,0.125,0.125]])

output = cv.connectedComponentsWithStats(img3, 4, cv.CV_32S)

print(output[0],output[2],output[3])
# cell = img2[(sr-1)*hc:sr*hc,(sc-1)*wc:sc*wc]

cor = np.zeros((colc+1,rowc+1,2),dtype=int)

for i in range(output[0]):
	if i==0:
		continue
	x = int(round(output[3][i][0]/wc,0))
	y = int(round(output[3][i][1]/hc,0))
	cor[x][y][0] = output[3][i][0]
	cor[x][y][1] = output[3][i][1]
	if x==0:
		cor[x][y][0] = 0
	if y==0:
		cor[x][y][1] = 0
	if x==colc:
		cor[x][y][0] = width-1
	if y==rowc:
		cor[x][y][0] = height-1

print(cor)

cell = img2[cor[sc-1][sr-1][1]:cor[sc][sr][1],cor[sc-1][sr-1][0]:cor[sc][sr][0]]

cv.namedWindow('changed', cv.WINDOW_NORMAL)
cv.imshow('changed', cell)

cv.waitKey(0)
cv.destroyWindow('changed')

cv.imwrite('2saveda.png',cell)