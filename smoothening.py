import cv2
import numpy as np
import rescale

img = cv2.imread('img/puppy.jpg')
cv2.imshow("Puppy",img)

rescaled_img = rescale.rescaleFrame(img,scale=0.3)
cv2.imshow("Rescaled_img",rescaled_img)

# AVERAGING BLUR
average = cv2.blur(rescaled_img, (3,3))
cv2.imshow("Average blur", average)

# GAUSSIAN BLUR
gaussian = cv2.GaussianBlur(rescaled_img, (3,3), 0)
cv2.imshow("Gaussian Blur",gaussian)

# MEDIAN BLUR - remove salt and pepper noise
median = cv2.medianBlur(rescaled_img, 3)
cv2.imshow("Median Blur",median)

# BILATERAL BLUR - RETAINS THE EDGES
bilateral = cv2.bilateralFilter(rescaled_img, 15, 45, 30)
cv2.imshow("Bilateral blur",bilateral)

cv2.waitKey(0)