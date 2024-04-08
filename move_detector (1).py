from xmlrpc.client import INVALID_XMLRPC
import cv2
import numpy as np
import math
def are_cross(contour1,contour2):
    p=0
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
         new_contour[0][1]=min(contour2[0][1],new_contour[0][1]) #min max
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

cap = cv2.VideoCapture( "C:/Users/bryzg/OneDrive/Desktop/230-video.mp4") #   'http://192.168.217.103/mjpg/video.mjpg'  "/home/ebryzgalova/Downloads/232-video.mp4"
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
success, frame1 = cap.read()
success, frame2 = cap.read()

while success: 

  diff = cv2.absdiff(frame1, frame2) 

  gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) 

  blur = cv2.GaussianBlur(gray, (7, 7), 0) 

  _, thresh = cv2.threshold(blur, 8, 255, cv2.THRESH_BINARY) 

  dilated = cv2.dilate(thresh, None, iterations = 2) 

  contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

  cords =[]
  for contour in contours:
    if cv2.contourArea(contour) < 500: 
      continue
    (x, y, w, h) = cv2.boundingRect(contour) 
    cords.append([[x,y],[x+w,y+h]])
   

  final_contours = clean_conturs(cords)
  for contor in final_contours:
    cv2.rectangle(frame1, (contor[0][0],contor[0][1]), (contor[1][0],contor[1][1]), (0, 255, 0), 2) 
    cv2.putText(frame1, "Status: {}".format("Dvigenie"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA) 

  cv2.imshow('Frame',cv2.resize(frame1,(650,500)))


  frame1 = frame2  #
  success, frame2 = cap.read() #
  #success, frame2 = cap.read()
  if cv2.waitKey(40) == 27:
    break

cap.release()
cv2.destroyAllWindows()