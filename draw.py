import cv2
import numpy as np

blank = np.zeros((500,500,3),dtype = 'uint8')
# cv2.imshow("Blank",blank)

blank[300:400,200:300] = 0,255,0
# cv2.imshow("Green",blank)

# RECTANGLE
cv2.rectangle(blank,(0,0),(200,200),(0,255,0),thickness = cv2.FILLED)
# cv2.imshow("Rect",blank)

# CIRCLE
cv2.circle(blank, (250,250), 50, (0,0,255),thickness=-1)
# cv2.imshow("Circle",blank)

# LINE
cv2.line(blank, (0,0), (200,200), (255,0,0),thickness=3 )
cv2.imshow("Line",blank)

# TEXT
cv2.putText(blank, "Hi! This is my first text!!", (0,300), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
cv2.imshow("Text", blank)

# original image
img = cv2.imread('img/rose.jpg')
# cv2.imshow("Rose",img)
cv2.waitKey(0)