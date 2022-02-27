import cv2
import numpy as np
import rescale

haar_cas = cv2.CascadeClassifier('haar_cascade_frontface.xml')

# --------------------------------------------------------
img = rescale.rescaleFrame(cv2.imread('img/face.jpg'), scale=0.3)
cv2.imshow("Person",img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",gray)

faces_region = haar_cas.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)

print("Number of faces found = ",len(faces_region))

for (x,y,w,h) in faces_region:
    cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),thickness=2)

cv2.imshow("Detected Faces",img)
# ---------------------------------------------------------
# ---------------------------------------------------------
group_img = cv2.imread('img/group_people.jpg')
gray_group = cv2.cvtColor(group_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray group",gray_group)

faces_region_group = haar_cas.detectMultiScale(gray_group, scaleFactor = 1.1, minNeighbors = 3)

for (x,y,w,h) in faces_region_group:
    cv2.rectangle(group_img, (x,y),(x+w,y+h),(0,255,0),thickness=2)

print("Number of faces found = ",len(faces_region_group))
cv2.imshow("Detected group Faces",group_img)
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)
while True:
    istrue, frame = cap.read()
    gray_cap = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_region_group = haar_cas.detectMultiScale(gray_cap, scaleFactor = 1.1, minNeighbors = 3)
    for (x,y,w,h) in faces_region_group:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),thickness=2)
    cv2.imshow("Processed",frame)
    if cv2.waitKey(27) & 0xFF == ord('d'):
        break

cv2.waitKey(0)