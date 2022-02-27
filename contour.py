import cv2
import numpy as np
import rescale

img = cv2.imread('img/puppy.jpg')
resized_img = rescale.rescaleFrame(img, scale = 0.3)

cv2.imshow("Image", resized_img)

gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
cv2.imshow("Gray", gray)

# BLUR
blur = cv2.blur(gray, (3,3), cv2.BORDER_DEFAULT)
cv2.imshow("Blur", blur)

# EDGES
canny = cv2.Canny(blur,125,175)
cv2.imshow("Canny",canny)

'''
CONTOURS looks at the structuring element from the 
edges of the image and returns two values:
1. contours - python list of all coordinates of the contours found in the image
2. heirarchies - heirarchical representation of contours. eg. if a square is inside a rectange, so these information are stored in heirarchies and "opencv" uses it to find contours

mode: 
1. cv2.RETR_LIST -> it retrieves all the contours
2. cv2.RETR_EXTERNAL -> it retrieves all the external contours
3. cv2.RETR_TREE -> returns all the heirarchical contours 

contour approximation method:
1. cv2.CHAIN_APPROX_NONE -> it returns the contours coordinates as it is
2. cv2.CHAIN_APPROX_SIMPLE -> it returns the contours coordinates in a simplified manner. 
Eg. A line is returned as a contour with all the coordinates with cv2.CHAIN_APPROX_NONE 
but in cv2.CHAIN_APPROX_SIMPLE, only 2 endpoints of the line is returned
'''
# THRESHOLDING
ret, thresh_img = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold",thresh_img)

# FINDING CONTOURS
contours_nothresh,heirarchies_nothresh = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours_thresh, heirarchies_thresh = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# NUMBER OF CONTOURS IN EACH CASE
print("Number of contours without thresholding the image: ",len(contours_nothresh))
print("Number of contours with thresholding the image: ",len(contours_thresh))

# MASKED CONTOURS
blank1 = np.zeros(resized_img.shape, dtype='uint8')
blank2 = np.zeros(resized_img.shape, dtype='uint8')

# DRAW THE CONTOURS IN MASKS
cv2.drawContours(blank1, contours_thresh, -1, (0,0,255), thickness = 1)
cv2.drawContours(blank2, contours_nothresh, -1, (0,255,0), thickness = 1)

# SHOW
cv2.imshow("Blank with thresholding", blank1)
cv2.imshow("Blank without thresholding",blank2)

cv2.waitKey(0)