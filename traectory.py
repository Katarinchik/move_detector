import cv2
import numpy as np
import math
color = np.random.randint(0, 255, (100, 3))
def are_cross(contour1,contour2):
    p=20
    betweenx = (contour1[0][0]<=contour2[0][0]+p and contour1[1][0]+p>=contour2[0][0]) or (contour1[0][0]<=contour2[1][0]+p and contour1[1][0]+p>=contour2[1][0])
    betweeny =(contour1[0][1]<=contour2[0][1]+p and contour1[1][1]+p>=contour2[0][1]) or (contour1[0][1]<=contour2[1][1]+p and contour1[1][1]+p>=contour2[1][1])
    inx =contour1[0][0]<=contour2[0][0]+p  and  contour1[1][0]+p>=contour2[1][0] or contour1[0][0]+p>=contour2[0][0] and  contour1[1][0]<=contour2[1][0]+p
    iny =contour1[0][1]<=contour2[0][1]+p and  contour1[1][1]+p>=contour2[1][1] or contour1[0][1]+p>=contour2[0][1]  and contour1[1][1]<=contour2[1][1]+p
    return (betweenx and betweeny) or (inx and betweeny) or (iny and betweenx) or (inx and iny)
 

def clean_conturs(contours):
 f =True
 
 while f:
  
   f= False
   res_contours =[]
   for contour1 in contours:
     new_contour = contour1.copy()
     for contour2 in contours:
       if contour1 == contour2:
           continue
       if are_cross(new_contour,contour2):
         new_contour[0][1]=min(contour2[0][1],new_contour[0][1]) 
         new_contour[0][0]=min(contour2[0][0],new_contour[0][0])
         new_contour[1][1]=max(contour2[1][1],new_contour[1][1])
         new_contour[1][0]=max(contour2[1][0],new_contour[1][0])
         contours.remove(contour2)
         f =True
   
     res_contours.append(new_contour)
     if(contour1 in contours):
       contours.remove(contour1)
   contours=res_contours.copy()
   
 return contours


def find_centres(contours):
  centres =[]
  for contour in contours:
    centres.append([int((contour[1][1]+contour[0][1])/2),int((contour[0][0]+contour[1][0])/2)])
  return(centres)

def traectory(centers, traectories,marks, maxtr =3):
  i=0
  for traectory in traectories:
    last_point = traectory[-1]
    if len(centers)!=0:
         ct =centers[0]

         min_dist =abs( last_point[0] - centers[0][0] + last_point[1] - centers[0][1])
         for center in centers:
           if abs( last_point[0] - center[0] + last_point[1] - center[1])< min_dist:
             min_dist =abs( last_point[0] - center[0] + last_point[1] - center[1])
             ct =center
         if min_dist<200:
           traectory.append(ct)
           centers.remove(ct)
         else:
             marks[i]+=1
    i+=1
    
  for c in centers:
    if len(traectories)>=maxtr:
        break
    traectories.append([c])
    
  return traectories, marks

def contr(img):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(5,5))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    
    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img2

cap = cv2.VideoCapture( "C:/Users/bryzg/OneDrive/Desktop/232-video.mp4") #   'http://192.168.217.103/mjpg/video.mjpg'  "/home/ebryzgalova/Downloads/232-video.mp4"
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
  
success, frame1 = cap.read()
success, frame2 = cap.read()
frame1 = contr(frame1)
marks = [0,0,0]
tracks =[]
while success: 
  frame2 = contr(frame2)
  diff = cv2.absdiff(frame1, frame2) 

  gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) 

  blur = cv2.GaussianBlur(gray, (5, 5), 0) 

  _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY) 

  dilated = cv2.dilate(thresh, None, iterations = 8) 

  contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

  cords =[]
  for contour in contours:
    if cv2.contourArea(contour) < 1500: 
      continue
    (x, y, w, h) = cv2.boundingRect(contour) 
    cords.append([[x,y],[x+w,y+h]])
   

  final_contours = clean_conturs(cords)
  for contor in final_contours:
    cv2.rectangle(frame1, (contor[0][0],contor[0][1]), (contor[1][0],contor[1][1]), (0, 255, 0), 2) # получение прямоугольника из точек кортежа
    cv2.putText(frame1, "Status: {}".format("Dvigenie"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA) # вставляем текст
  centers =find_centres(final_contours)
  tracks, marks = traectory(centers, tracks, marks)
  for i in range(len(marks)):
      if marks[i]>90 and len(tracks)>i:
          tracks.pop(i)
          marks[i] =0
  j=-1
  for track in tracks:
      for i in range(1,len(track)-1 ):
          track[i][0] = int((track[i][0]+track[i-1][0]*0.3+track[i][0]*0.3)/1.6)
          track[i][1] = int((track[i][1]+track[i-1][1]*0.3+track[i][1]*0.3)/1.6)
  for track in tracks:
    j+=1
    for i in range(len(track)-1):
        cv2.line(frame1,(track[i][1],track[i][0]),(track[i+1][1],track[i+1][0]), color[j].tolist(),thickness= 9) # (track[i][1],track[i][0]),(track[i+1][1],track[i+1][0])   track[i],track[i+1]

  cv2.imshow('Frame',cv2.resize(frame1,(650,500)))
  frame1 = frame2  #
  success, frame2 = cap.read() #
  
  if cv2.waitKey(40) == 27:
    break

cap.release()
cv2.destroyAllWindows()