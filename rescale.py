import cv2 as cv



def rescaleFrame(frame, scale = 0.75):
    # works for images, videos
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width, height)

    return cv.resize(frame,dimensions, interpolation = cv.INTER_AREA)

def changeRes(width, height):
    # works onyl for LIVE VIDEO ONLY
    capture.set(3,width)
    capture.set(4,height)

# resized img
# img = cv.imread('img/puppy.jpg')
# resized_img = rescaleFrame(img)
# cv.imshow('puppy', img)
# cv.imshow('puppy_resized', resized_img)
# cv.waitKey(0)

# RESCALING VIDEOS
# video = cv.VideoCapture('video/doggie.mp4')

# while True:
#     istrue, frame = video.read()
#     frame_resized = rescaleFrame(frame, scale = 0.2)
#     cv.imshow('Video',frame)
#     cv.imshow('Video resized', frame_resized)

#     if cv.waitKey(20) & 0xFF == ord('d'):
#         break
# video.release()
# cv.destroyAllWindows()
# cv.waitKey(0)