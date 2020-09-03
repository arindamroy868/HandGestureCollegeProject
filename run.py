import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import cv2
import os

lookup = {'Fist': 0, 'Nothing': 1, 'Palm': 2, 'Swing': 3, 'Thumb': 4, 'Yo': 5}

reverselookup = {0: 'Fist', 1: 'Nothing', 2: 'Palm', 3: 'Swing', 4: 'Thumb', 5: 'Yo'}

def model_func(reverselookup,model):
    cap  = cv2.VideoCapture(0)

    _, first_frame = cap.read()
    cv2.rectangle(first_frame,(0,100),(300,400),(0,255,0))
    roi = first_frame[100:400,0:300]
    first_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    first_roi = cv2.GaussianBlur(first_roi, (5, 5), 0)

    while True:
        _,frame = cap.read()
        #To flip the image if needed
        #frame = cv2.flip(frame,1)
        
        cv2.rectangle(frame,(0,100),(300,400),(0,255,0))
        roi = frame[100:400,0:300]
        
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        
        difference = cv2.absdiff(first_roi, gray_roi)
        _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
        
        difference_resized = cv2.resize(difference,(400,400))
        difference_reshaped = difference_resized.reshape(1,400,400,1)/255.0
        cv2.putText(frame, reverselookup[np.argmax(model.predict(difference_reshaped))], (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("first_frame",first_frame)
        cv2.imshow("frame",frame)
        cv2.imshow("difference",difference)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
model_path = os.path.abspath('3rd_model.h5')
model = load_model(model_path)

model_func(reverselookup,model)
