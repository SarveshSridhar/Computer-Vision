import cv2
import numpy as np
import rescale
import matplotlib.pyplot as plt

img = cv2.imread('img/puppy.jpg')
# cv2.imshow("Puppy",img)
rescaled_img = rescale.rescaleFrame(img,scale=0.3)
cv2.imshow("Rescaled_img",rescaled_img)
gray = cv2.cvtColor(rescaled_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray color",gray)


rose = cv2.imread('img/rose.jpg')
rescaled_rose = rescale.rescaleFrame(rose,scale=0.3)
gray_rose = cv2.cvtColor(rescaled_rose, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray rose",gray_rose)

# LAPLACIAN EDGE DETECTION METHOD
lap = cv2.Laplacian(gray, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
cv2.imshow("Laplacian",lap)

lap1 = cv2.Laplacian(gray_rose, cv2.CV_64F)
lap1 = np.uint8(np.absolute(lap1))
cv2.imshow("Rose-Laplacian",lap1)

# SOBEL EDGE DETECTION
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1,0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0,1)
sobelxy = cv2.bitwise_or(sobelx, sobely)

cv2.imshow("Sobelx",sobelx)
cv2.imshow("Sobely",sobely)
cv2.imshow("Sobelxy",sobelxy)

#CANNY
canny = cv2.Canny(gray, 150, 175)
cv2.imshow("Canny",canny)

cv2.waitKey(0)