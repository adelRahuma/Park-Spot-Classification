import os
import cv2
import pickle

try:
     with open('CarParkPoints_', 'rb') as f:
          posList = pickle.load(f)    
except:
     posList = []

width, height = 48,21
def mouseClick(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        posList.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
         for i, pos in enumerate(posList):
              x1, y1 = pos
              if x1 < x < x1+width and y1 < y < y1 + height:
                   posList.pop(i)

#     with open('CarParkPoints', 'wb') as f:
#          pickle.dump(posList,f)
         
# read image
image_path = os.path.join(".","Images","Park.jpg")

while True:
    img = cv2.imread(image_path)
    for pos in posList:
        cv2.rectangle(img, pos,(pos[0]+ width, pos[1] + height),(0,0,255),1)
    cv2.imshow('img',img)
    cv2.setMouseCallback('img',mouseClick)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  