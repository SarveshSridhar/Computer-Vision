import cv2
import numpy as np
import rescale

img = cv2.imread('img/puppy.jpg')
cv2.imshow("Puppy",img)

rescaled_img = rescale.rescaleFrame(img,scale=0.3)
cv2.imshow("Rescaled_img",rescaled_img)

blank = np.zeros((400,400), dtype= 'uint8')
rectangle = cv2.rectangle(blank.copy(), (30,30), (300,300), 255, -1)
circle = cv2.circle(blank.copy(), (200,200), 200, 255,-1)

cv2.imshow("Rectangle",rectangle)
cv2.imshow("Circle",circle)

# bitwise AND  - intersecting regions
bit_and = cv2.bitwise_and(rectangle, circle)
cv2.imshow("AND",bit_and)

# bitwise OR - non-intersecting and intersecting regions
bit_or = cv2.bitwise_or(rectangle, circle)
cv2.imshow("OR",bit_or)

# bitwise XOR - good for getting non-intersecting regions
bit_xor = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("XOR",bit_xor)

# bitwise NOT
bit_not = cv2.bitwise_not(rectangle)
cv2.imshow("NOT", bit_not)

cv2.waitKey(0)