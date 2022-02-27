import cv2
import rescale
import numpy as np

img = cv2.imread('img/puppy.jpg')
resized_img = rescale.rescaleFrame(img, scale = 0.3)

cv2.imshow("Image", resized_img)

# TRANSFORMATION FUNTION 
def translate(img, x, y):
    transmat = np.float32([[1,0,x], [0,1,y]])
    dimension = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, transmat, dimension)

'''
Transformation direction
-x --> left
+y --> right
-y --> up
+y --> down
'''
translated_img1 = translate(resized_img, 100, 100)
translated_img2 = translate(resized_img, -100, 100)
translated_img3 = translate(resized_img, -100, -100)
translated_img4 = translate(resized_img, 100, -100)

cv2.imshow("Translated image1",translated_img1)
cv2.imshow("Translated image2",translated_img2)
cv2.imshow("Translated image3",translated_img3)
cv2.imshow("Translated image4",translated_img4)

# ROTATIONS
def rotateimg(img, angle, rotPoint=None):
    (height,width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width//2, height//2)
    rotmat = cv2.getRotationMatrix2D(rotPoint, angle, 1)
    dimensions = (width,height)
    rot_img = cv2.warpAffine(img, rotmat, dimensions)
    return rot_img

rot_img1 = rotateimg(resized_img, 20,rotPoint=(200,200))
cv2.imshow("Rotated img", rot_img1)

'''
FLIPPING
parameter :-
greater than 0 and equal to zero --> horizontal flip (left to right)
less than 0 --> vertical flip (top to bottom)
'''
flip = cv2.flip(resized_img, -1)
cv2.imshow("flip",flip)

cv2.waitKey(0)