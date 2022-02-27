import cv2
import numpy as np
import rescale
import matplotlib.pyplot as plt

img = cv2.imread('img/puppy.jpg')
cv2.imshow("Puppy",img)

rescaled_img = rescale.rescaleFrame(img,scale=0.3)
cv2.imshow("Rescaled_img",rescaled_img)

gray = cv2.cvtColor(rescaled_img, cv2.COLOR_RGB2GRAY)
cv2.imshow("Gray", gray)

# Grayscale histogram
blank = np.zeros(rescaled_img.shape[:2],dtype='uint8')

circle = cv2.circle(blank, (rescaled_img.shape[1]//2, rescaled_img.shape[0]//2), 100, 255,-1)

mask = cv2.bitwise_and(rescaled_img,rescaled_img, mask= circle)

cv2.imshow("Mask",mask)

gray_hist1 = cv2.calcHist([mask], [0], mask, [256], [0,256])

plt.figure()
plt.title("Grayscale histogram")
plt.xlabel("Bins")
plt.ylabel("Number of pixels")
plt.xlim([0,256])
plt.plot(gray_hist1)
plt.show()

# COLOR HISTOGRAM
colors = ('b','g','r')

plt.figure()
plt.title("Grayscale histogram")
plt.xlabel("Bins")
plt.ylabel("Number of pixels")

for i,col in enumerate(colors):
    hist = cv2.calcHist([rescaled_img],[i],mask,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])

plt.show()

cv2.waitKey(0)