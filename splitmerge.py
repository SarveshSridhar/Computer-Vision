import cv2
import numpy as np
import rescale

img = cv2.imread('img/puppy.jpg')
cv2.imshow("Puppy",img)

rescaled_img = rescale.rescaleFrame(img,scale=0.3)
cv2.imshow("Rescaled_img",rescaled_img)

# SPLIT THE RGB CHANNELS
b,g,r = cv2.split(rescaled_img)
cv2.imshow("Green",g)
cv2.imshow("Blue",b)
cv2.imshow("Red",r)

print(rescaled_img.shape,b.shape, g.shape, r.shape)

# MERGE THE CHANNELS
merged = cv2.merge([b,g,r])
cv2.imshow("Merged image",merged)

# SHOW RED, GREEN, BLUE COLORED IMAGES
blank = np.zeros(rescaled_img.shape[:2], dtype='uint8')

blue = cv2.merge([blank,blank,b])
red = cv2.merge([r,blank,blank])
green = cv2.merge([blank,g,blank])

cv2.imshow("Pure red",red)
cv2.imshow("Pure green",green)
cv2.imshow("Pure blue",blue)



cv2.waitKey(0)