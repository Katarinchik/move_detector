import cv2

import math
from skimage.io import imread, imsave, imshow
from math import pi, exp
import numpy as np
from numpy import array
from scipy.signal import convolve2d
import skimage
from skimage  import img_as_ubyte
from numpy import clip



def gauss(x, y):
    s =  10
    return 1 / (2 * pi * s ** 2) * exp((-x ** 2 - y ** 2) / (2 * s ** 2))

def gaussian(imgage):
    k = 3
    gauss_kernel = np.array([[gauss(i-k//2, j-k//2) for j in range(k)] for i in range(k)])
    gauss_kernel = (gauss_kernel / gauss_kernel.sum())

    img = imgage

    img_new = convolve2d(img, gauss_kernel / gauss_kernel.sum(), mode='valid')

    return img_new

def make_lines_bigger(image,n):

    for i in range(n):
        image[1:-1] = (image[1:-1] + image[:-2]  + image[2:] )
        image[:, 1:-1] = (image[:, 1:-1]  + image[:, :-2] + image[:, 2:])
    
    return clip(image,0,1)

def list_of_objects_points(image): 
  objects_points=[]
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      if image[i][j] == 1:
        new_points = []
        object_points =[]
        new_points.append([i,j])

        while (len(new_points)>0):
          temp = new_points[0];

          if temp[0]>0 and image[temp[0]-1][temp[1]]==1:
            new_points.append([temp[0]-1,temp[1]])
            image[temp[0]-1][temp[1]] = 0


          if temp[1]>0 and image[temp[0]][temp[1]-1]==1 :
            new_points.append([temp[0],temp[1]-1])
            image[temp[0]][temp[1]-1] = 0

          if temp[0]<image.shape[0]-1 and image[temp[0]+1][temp[1]]==1 :
            new_points.append([temp[0]+1,temp[1]])
            image[temp[0]+1][temp[1]] = 0


          if temp[1]<image.shape[1]-1 and image[temp[0]][temp[1]+1]==1 :
            new_points.append([temp[0],temp[1]+1])
            image[temp[0]][temp[1]+1] = 0

          object_points.append(temp)
          image[temp[0]][temp[1]] = 0
          new_points.pop(0)


        objects_points.append(object_points)
  return objects_points

def find_max(point_list,cord):
  max_mean = point_list[0][cord]
  for i in range(1,len(point_list)):
    if(point_list[i][cord]>max_mean):
      max_mean = point_list[i][cord]
  return max_mean

def find_min(point_list,cord):
  min_mean = point_list[0][cord]
  for i in range(1,len(point_list)):
    if(point_list[i][cord]<min_mean):
      min_mean = point_list[i][cord]
  return min_mean

def draw_contur(frame,cords):
  for cord in cords:

      frame[cord[2]:cord[0],cord[1]-1:cord[1]+1]=[0,1,0]
      frame[cord[2]:cord[0],cord[3]-1:cord[3]+1]=[0,1,0]
      frame[cord[0]-1:cord[0]+1,cord[3]:cord[1]]=[0,1,0]
      frame[cord[2]-1:cord[2]+1,cord[3]:cord[1]]=[0,1,0]

  return frame

cap = cv2.VideoCapture("C:/Users/bryzg/OneDrive/Desktop/232-video.mp4")# /home/ebryzgalova/Downloads/232-video.mp4 "C:/Users/bryzg/OneDrive/Desktop/230-video.mp4"
success, frame1 = cap.read()
success, frame2 = cap.read()

frame2 = np.asarray(cv2.resize(frame2,(650,500)))
 
frame2 = frame2 /255 
while success:
    frame1 = np.asarray(cv2.resize(frame1,(650,500)) )/255
    work_frame = abs(frame1 - frame2)

    gray_frame = (work_frame[:,:,0]*0.2126+work_frame[:,:,1]*0.7152+work_frame[:,:,2]*0.0722)
    #gauss_frame = gaussian(gray_frame)
    frame_with_white_conture = (gray_frame > 0.05) * (gray_frame )

    big_white_conture = make_lines_bigger(frame_with_white_conture,3)
    objects_points = list_of_objects_points(big_white_conture)
 
    rectangle_cords =[]
    if(len(objects_points)==0):
        final_frame = frame2
    else:
      for object_points in objects_points:
        if len(object_points)>700:
          x1 = find_max(object_points,0)#функция
          x2 = find_min(object_points,0)
          y1 = find_max(object_points,1)
          y2 = find_min(object_points,1)
          rectangle_cords.append([x1,y1,x2,y2])
       #print(rectangle_cords)
      final_frame = draw_contur(frame2,rectangle_cords)
    cv2.imshow('Frame',cv2.resize(skimage.img_as_ubyte(final_frame),(650,500)))
    
    frame2 = frame1.copy()
    success, frame1 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()