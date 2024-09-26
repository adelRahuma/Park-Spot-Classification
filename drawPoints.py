import os
import cv2
import pickle


try:
   with open('Park_spots','rb') as f:
      posList = pickle.load(f)
except:
   print('File not exist')
   posList = []
path = os.path.join('.','Images','Park.jpg')
width =48
height = 20

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
while True:
    img = cv2.imread(path)
    for pos in posList:
       cv2.rectangle(img,pos, (pos[0] + width, pos[1] + height),(0,255,0),1)
    cv2.imshow('img', img)
    cv2.setMouseCallback('img',mouseClick)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

