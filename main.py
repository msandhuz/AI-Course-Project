import cv2
import numpy as np
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

faceClassify = cv2.CascadeClassifier(r'/Users/eeshafarrukh/PycharmProjects/AIproj/haarcascade_frontalface_default.xml')
classifier = load_model(r'/Users/eeshafarrukh/PycharmProjects/AIproj/model.h5')
emotionArr = ['ANGRY', 'DISGUST', 'FEAR', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISE']
camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    arr = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgArr = faceClassify.detectMultiScale(gray)

    for (x, y, w, h) in imgArr:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 0, 128), 2)  # Change square color here

        roiGray = gray[y:y+h, x:x+w]
        roiGray = cv2.resize(roiGray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roiGray]) != 0:
            roi = roiGray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            prediction = classifier.predict(roi)[0]
            label = emotionArr[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # Change text color here
        else:
            cv2.putText(frame, 'No faces found', (30, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # Change text color here

    cv2.imshow('Prediction System of Micro-Expressions Face Using the Convolutional Neural Network', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
