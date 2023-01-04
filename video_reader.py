import cv2
import numpy as np
from keras.utils import img_to_array
from keras.models import load_model


faceClassif = cv2.CascadeClassifier('modelo/haarcascade_frontalface_default.xml')

classifier = load_model('EmotionDetectionModel.h5')
class_labels=['Angry','Fear','Happy','Neutral','Sad','Surprise']

captura = cv2.VideoCapture(0) 

while captura.isOpened():
    ret, imagen = captura.read()
    gray=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)

    if ret:
        faces = faceClassif.detectMultiScale(imagen, minNeighbors=5, )

        for (x,y,w,h) in faces:
            cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,255,0), 1)

            face = gray[y:y+h+1, x:x+w+1]

            mini_face=cv2.resize(face,(48,48),interpolation=cv2.INTER_AREA)
            mini_face=mini_face.astype('float')/255.0
            mini_face=img_to_array(mini_face)
            mini_face=np.expand_dims(mini_face,axis=0)
            print(mini_face.shape)

            cv2.imshow("Face", face)

            preds = classifier.predict(mini_face)
            print('preds:',preds)
            label=class_labels[preds.argmax()]
            label_position=(x,y)
            cv2.putText(imagen,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            
        
        cv2.imshow("Video",imagen)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
captura.release()
cv2.destroyAllWindows()