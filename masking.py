import cv2
import numpy as np
import rescale

img = cv2.imread('img/puppy.jpg')
cv2.imshow("Puppy",img)

rescaled_img = rescale.rescaleFrame(img,scale=0.3)
cv2.imshow("Rescaled_img",rescaled_img)

# CREATE BLANK IMAGE
blank = np.zeros(rescaled_img.shape[:2], dtype='uint8')

mask = cv2.circle(blank, (rescaled_img.shape[1]//2, rescaled_img.shape[0]//2), 100, 255, -1)
cv2.imshow("Mask", mask)

masked_img = cv2.bitwise_and(rescaled_img, rescaled_img, mask=mask)
cv2.imshow("Masked image", masked_img)

cv2.waitKey(0)