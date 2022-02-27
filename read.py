import cv2

# images

# img = cv2.imread('img/puppy.jpg')
# cv2.imshow("Image",img)
# cv2.waitKey(0)

'''
VideoCapture(0) --> Webcam
VideoCapture("path") --> plays a video

'''
video = cv2.VideoCapture('video/doggie.mp4')
# video = cv2.VideoCapture(0)
while True:
    istrue, frame = video.read()
    cv2.imshow('Video',frame)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break
video.release()
cv2.destroyAllWindows()