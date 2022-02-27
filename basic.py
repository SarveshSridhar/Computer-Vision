import cv2
import rescale

img = cv2.imread('img/puppy.jpg')
cv2.imshow('puppy',rescale.rescaleFrame(img, scale = 0.3))

# RESIZED IMG
resized_img = rescale.rescaleFrame(img,scale = 0.3)

# CONVERTING TO GRAYSCALE
gray_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray_img',gray_img)

# BLUR IMAGE
blur_img = cv2.GaussianBlur(resized_img, (3,3), cv2.BORDER_DEFAULT)
cv2.imshow("Blur_img",blur_img)

# Edge Cascade - Canny edge detection
## BLURRING AND EDGE DETECTING CAN BE DONE TO REDUCE NUMEROUS NUMBER OF EDGES
canny1 = cv2.Canny(blur_img, 125,175)
canny2 = cv2.Canny(resized_img, 125, 175)
cv2.imshow("Edges from Canny blur", canny1)
cv2.imshow("Edges from Canny", canny2)

# Dilate the image
dilate1 = cv2.dilate(canny1, (3,3), iterations = 3)
dilate2 = cv2.dilate(canny2, (3,3), iterations = 3)
cv2.imshow("Dilated img-blur",dilate1)
cv2.imshow("Dilated img-canny",dilate2)

# Erode the image
erode1 = cv2.erode(dilate1, (3,3), iterations = 3)
cv2.imshow("Eroded image", erode1)

'''
Resizing the image:
1. To shrink - Use interpolation = cv2.INTER_AREA
2. To enlarge - Use interpolation = cv2.INTER_LINEAR or cv2.INTER_CUBIC
'''
resized = cv2.resize(img, (1000,1000), interpolation = cv2.INTER_AREA)
cv2.imshow("Resized img",resized)

# CROPPING
crop = resized_img[100:300,200:400]
cv2.imshow("Cropped img",crop)

cv2.waitKey(7000)
