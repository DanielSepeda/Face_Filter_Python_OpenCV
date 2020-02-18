import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_alt.xml')
dwight = cv2.imread(r'dwight_photo.jpg')

def detect_face1(img):
    
    face_img1 = img.copy()
    
    face_rects1 = face_cascade.detectMultiScale(face_img1)
    
    for (x,y,w,h) in face_rects1:
        
        cv2.rectangle(face_img1,(x,y),(x+w,y+h),(0,255,0,0))
        face_detected1 = face_img1[y:y+h,x:x+w]
        face_detected1 = cv2.resize(dwight,(h,w))
        face_img1[y:y+h,x:x+w] = face_detected1
        
    return face_img1
    
    while True:
        cap = cv2.VideoCapture(0)

        ret,frame = cap.read(0)
        
        frame = detect_face1(frame)
    
        cv2.imshow("Dwight", frame)

        k = cv2.waitKey(1)
        if k ==27:
            break

  
cap.release()
cv2.destroyAllWindows()
