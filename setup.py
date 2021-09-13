import math
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def ImageToCoord(path_dir):
  img_2 = cv.imread(path_dir) # read image file
  fig = plt.figure(figsize=(20,20))
  p = fig.add_subplot(1,2,1)
  p.imshow(img_2)
  plt.title('Input Image')

  edges_2 = cv.Canny(img_2,80,300) # convert grayscale image into an edge map (binary)
  p = fig.add_subplot(1,2,2)
  p.imshow(edges_2)
  plt.title('Edge Map')
  plt.show()
  WIDTH = edges_2.shape[0]
  HEIGHT = edges_2.shape[1]
  coord_x = []
  coord_y = []
  for i in range(WIDTH):
    for j in range(HEIGHT):
      if edges_2[i][j]!=0:
        coord_x.append(i)
        coord_y.append(j)
  return coord_x, coord_y


# Hough Transform for line
def HoughTransformLine(*coord_x, *coord_y):
  THETA = 179
  RADIUS = 2000
  H = [[0 for k in range(THETA)] for j in range(RADIUS)]
  for i in range(len(coord_x)):
    for x in range(THETA):
      r = round((coord_x[i])*math.cos(x*math.pi/180) + (coord_y[i])*math.sin(x*math.pi/180)) # r = xcos(angle) + ysin(angle)
      if r<RADIUS//2 - 2 and r>-RADIUS//2 + 2:
        H[r-1+RADIUS//2][x]+=1
        H[r+RADIUS//2][x]+=1
        H[r+1+RADIUS//2][x]+=1
  return H # accumulator function
      
def HoughTransformCircle(*coord_x, *coord_y):
  HEIGHT = 1000
  WIDTH = 1000
  H = [[[0 for k in range(410)] for j in range(HEIGHT)] for i in range(WIDTH)]
  for i in range(len(coord_x)):
    for x in range(WIDTH):
      for y in range(HEIGHT):
        r = math.floor(math.sqrt((coord_x[i]-x)*(coord_x[i]-x)+(coord_y[i]-y)*(coord_y[i]-y)))
        if r<405 and r>=2 and WIDTH-x > 3 and x > 3 and y > 3 and HEIGHT - y > 3:
          H[x][y][r+1]+=1
          H[x][y][r]+=1
          H[x][y][r-1]+=1
  return H

def FindArgMax(*H):
  plt.imshow(np.array(H), cmap = 'gray', aspect= 180/250)
  h = np.array(H)
  argmax = [np.unravel_index(np.argmax(h), h.shape) for r in h]
  xmax = argmax[0][0]
  ymax = argmax[0][1]
  return xmax, ymax

#output
def CoordToImage( *coord_x, *coord_y):
  display = []
  sub = []
  for i in range(WIDTH):
    sub = []
    for j in range(HEIGHT):
      cou = 0
      for run in range(len(coord_x)):
        if i==coord_x[run] and j==coord_y[run]:
          cou+=1
      if cou!=0:
        sub.append(1)
      else:
        sub.append(0)
    display.append(sub)
  return display
