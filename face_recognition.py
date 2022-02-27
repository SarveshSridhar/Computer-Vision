import cv2
import numpy as np
import os

people = ['vijay','rdj','chris_evans']

dir = r'D:\college\6TH SEM\Computer Vision\project\practice\people_faces\train'

haar_cas = cv2.CascadeClassifier('haar_cascade_frontface.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(dir,person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path)
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            
            rect = haar_cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in rect:
                roi = gray[y:y+h,x:x+w]
                features.append(roi)
                labels.append(label)

create_train()

faces_recognizer = cv2.face.LBPHFaceRecognizer_create()

features = np.array(features, dtype='object')
labels = np.array(labels)

print(features.shape, labels.shape)

faces_recognizer.train(features, labels)

faces_recognizer.save('face_trained.yml')
# np.save('features.npy',features)
# np.save('labels.npy',labels)
# faces_recognizer.read('face_trained.yml')

test_img1 = cv2.imread('people_faces/val/rdj/3.jpg')
sample_img1 = cv2.cvtColor(test_img1, cv2.COLOR_BGR2GRAY)
# sample_img2 = cv2.cvtColor(cv2.imread('/people_faces/val/vijay/2.webp'),cv2.COLOR_BGR2GRAY)
# sample_img3 = cv2.cvtColor(cv2.imread('/people_faces/val/vijay/3.jpg'),cv2.COLOR_BGR2GRAY)

def show_predict_picture(sample_img1, test_img1):
    face_rect = haar_cas.detectMultiScale(sample_img1,3, 6)

    for (x,y,w,h) in face_rect:
        roi = sample_img1[y:y+h,x:x+w]

        label,confidence = faces_recognizer.predict(roi)
        print(f'Label: str{people[label]}, with confidence of {confidence}')

        cv2.putText(test_img1, str(people[label]), (20,20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), thickness=2)
        cv2.rectangle(test_img1, (x,y), (x+w,y+h), (0,255,0),thickness=2)

    cv2.imshow("Predicted result",test_img1)

    cv2.waitKey(0)

# show_predict_picture(sample_img1, test_img1)

test_img = cv2.imread('people_faces/val/chris_evans/1.jpg')
sample_img = cv2.cvtColor(test_img1, cv2.COLOR_BGR2GRAY)

show_predict_picture(sample_img, test_img)