import os
import pickle
import cv2
import numpy as np
from skimage.transform import resize

with open('model.p', 'rb') as model:
    svc_model = pickle.load(model)


try:
    with open('Park_spots', 'rb') as spots:
        posList = pickle.load(spots)
except:
    posList= []

path = os.path.join('.','Images','Park.jpg')
width = 48
height = 20
#print(posList[0])
def mouseClick(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONUP:
      posList.append((x, y))
    elif event == cv2.EVENT_RBUTTONUP:
       for i, pos in enumerate(posList):
          old_x, old_y = pos
          if old_x < x < old_x + width and old_y < y < old_y + height:
            posList.pop(i)
    with open('Park_spots', 'wb') as file:
       pickle.dump(posList,file)

def feature_extracted(image):
    img_resized = resize(image, (15,15,3))
    img_resized = img_resized.flatten()
    return img_resized
while True:
    emptySpot = 0
    notEmptySpot = 0
    img = cv2.imread(path)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for pos in posList:
        x,y = pos
        #print('kkkk',pos)
        crope_img = imgGray[y:y + height, x: x + width]

        features = feature_extracted(crope_img)
        # print(type(features))
        features =np.array([features])

        prediction = svc_model.predict(features)[0]
        # print('prediction: ', prediction[0])
        if prediction == 1:
            color = (0,0,255)
            notEmptySpot +=1
        else:
            color = (0,255,0)
            emptySpot +=1

        cv2.rectangle(img,pos,(pos[0] + width, pos[1] + height),color,2)

    cv2.putText(img, f' Empty slots: {emptySpot}/{notEmptySpot}',(10,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,2559),2)
    cv2.imshow('img',img)
    cv2.setMouseCallback('img',mouseClick)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break