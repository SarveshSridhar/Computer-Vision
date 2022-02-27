import cv2
import numpy as np
import rescale

img = cv2.imread('img/puppy.jpg')
cv2.imshow("Puppy",img)

rescaled_img = rescale.rescaleFrame(img,scale=0.3)
cv2.imshow("Rescaled_img",rescaled_img)

# RGB TO GRAYSCALE
gray = cv2.cvtColor(rescaled_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",gray)

'''
RGB TO HSV
cv2.COLOR_RGB2HSV_FULL - 0 to 360
cv2.COLOR_RGB2HSV - 0 to 180
'''
hsv1 = cv2.cvtColor(rescaled_img, cv2.COLOR_RGB2HSV_FULL)
hsv2 = cv2.cvtColor(rescaled_img, cv2.COLOR_RGB2HSV)
cv2.imshow("HSV1",hsv1)
cv2.imshow("HSV2",hsv2)

# RGB TO LAB
lab = cv2.cvtColor(rescaled_img, cv2.COLOR_RGB2LAB)
cv2.imshow("LAB",lab)

'''
HSV TO BGR
converting from cv2_COLOR_RGB2HSV_FULL to cv2.COLOR_HSV2RGB is not possible
converting from cv2_COLOR_RGB2HSV to cv2.COLOR_HSV2RGB is possible
'''
hsv_bgr = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)
cv2.imshow("hsv to bgr",hsv_bgr)

cv2.waitKey(0)