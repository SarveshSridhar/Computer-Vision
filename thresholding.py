import cv2
import numpy as np
import rescale
import matplotlib.pyplot as plt

img = cv2.imread('img/puppy.jpg')
# cv2.imshow("Puppy",img)

rescaled_img = rescale.rescaleFrame(img,scale=0.3)
cv2.imshow("Rescaled_img",rescaled_img)

gray = cv2.cvtColor(rescaled_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",gray)
# SIMPLE THRESHOLDING
threshold,thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
cv2.imshow("Thresholded image",thresh)

threshold,thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Thresholded-Inverse image",thresh)

# ADAPTIVE THRESHOLD
adaptive_thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 10)
guassian_adaptive_thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 10)
cv2.imshow("Adaptive Threshold", adaptive_thres)
cv2.imshow("Adaptive Gaussian Threshold", guassian_adaptive_thres)

cv2.waitKey(0)