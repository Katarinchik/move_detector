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
def clean_conturs(contours):
   return contours
cap = cv2.VideoCapture("C:/Users/bryzg/OneDrive/Desktop/230-video.mp4")
success, frame1 = cap.read()
success, frame2 = cap.read()
cords =[]
diff = cv2.absdiff(frame1, frame2) 
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) 
blur = cv2.GaussianBlur(gray, (5, 5), 0) 
_, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
dilated = cv2.dilate(thresh, None, iterations = 3) 
contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
  if cv2.contourArea(contour) < 700: 
    continue
  (x, y, w, h) = cv2.boundingRect(contour) 
  cords.append([np.float32(x+w/2),np.float32(y+h/2)])
final_contours = clean_conturs(contours)
for contor in final_contours:
  cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2) 
  cv2.putText(frame1, "Status: {}".format("Dvigenie"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA) 



cords =np.asarray(cords)




lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create a random color mask for drawing optical flow
color = np.random.randint(0, 255, (100, 3))
times = 0
success, frame1 = cap.read()
success, frame2 = cap.read()
old_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)    
old_points = cords
track =[]
while True:

    success, frame = cap.read()
    if frame is None:
        break

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the optical flow using the Lucas-Kanade method
    
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, None, **lk_params)

    # Select the good points (those that have been successfully tracked)
    
    good_new = new_points[status.squeeze() == 1]
    good_old = old_points[status.squeeze() == 1]

    # Draw the optical flow on the frame
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = map(int, new.ravel())
        c, d = map(int, old.ravel())
        if len(track)<=i:
            track.append([[c,d]])
        else:
            track[i].append([c,d])

        for j in range(1,len(track[i])-1 ):
          track[i][j][0] = int((track[i][j][0]+track[i][j-1][0]*0.3+track[i][j][0]*0.3)/1.6)
          track[i][j][1] = int((track[i][j][1]+track[i][j-1][1]*0.3+track[i][j][1]*0.3)/1.6)

        for j in range(len(track[i])-1):
            cv2.line(frame,track[i][j],track[i][j+1], color[i].tolist(),thickness= 3)
        #frame = cv2.line(frame, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 10, color[i].tolist(), -1)

   
    cv2.imshow('Frame',cv2.resize(frame, (680, 460)))

    # Update the previous frame and points
    old_gray = frame_gray.copy()
    old_points = good_new.reshape(-1, 1, 2)
    if cv2.waitKey(40) == 27:
       break
cap.release()
cv2.destroyAllWindows()