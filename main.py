import cv2
from matplotlib.pyplot import gray

face_cascade = cv2.CascadeClassifier(r"classifier/hearcascade_frontalface_default.xml")
img = cv2.imread(r"photos/resim1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

faces = face_cascade.detectMultiScale(gray,1.1,4)

for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),3)

cv2.imshow('img',img)
cv2.waitKey(0)