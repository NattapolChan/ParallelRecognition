import math
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
img_2 = cv.imread('/content/drive/MyDrive/')
fig = plt.figure(figsize=(20,20))
p = fig.add_subplot(1,2,1)
p.imshow(img_2)
plt.title('Input Image')

edges_2 = cv.Canny(img_2,80,300)
p = fig.add_subplot(1,2,2)
p.imshow(edges_2)
plt.title('Edge Map')
plt.show()

#read image(2D-array)
WIDTH = edges_2.shape[0]
HEIGHT = edges_2.shape[1]
coord_x = []
coord_y = []
for i in range(WIDTH):
  for j in range(HEIGHT):
    if edges_2[i][j]!=0:
      coord_x.append(i)
      coord_y.append(j)


#line
THETA = 179
RADIUS = 2000
H = [[0 for k in range(THETA)] for j in range(RADIUS)]
for i in range(len(coord_x)):
  for x in range(THETA):
    r = round((coord_x[i])*math.cos(x*math.pi/180) + (coord_y[i])*math.sin(x*math.pi/180))
    if r<RADIUS//2 - 2 and r>-RADIUS//2 + 2:
      H[r-1+RADIUS//2][x]+=1
      H[r+RADIUS//2][x]+=1
      H[r+1+RADIUS//2][x]+=1

import matplotlib.pyplot as plt

plt.imshow(np.array(H), cmap = 'gray', aspect= 180/250)
h = np.array(H)
argmax = [np.unravel_index(np.argmax(h), h.shape) for r in h]
xmax = argmax[0][0]
ymax = argmax[0][1]

#output
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

arr = []
for i in range(h.shape[1]):
  sub = 0
  for j in range(h.shape[0]):
    sub += h[j,i]
  arr.append(sub)
arr = np.array(arr)
plt.plot(np.arange(arr.shape[0]), arr)
arr = []
for i in range(h.shape[1]):
  sub = 0
  for j in range(h.shape[0]):
    sub += h[j,i]*h[j,i]
  arr.append(sub)
arr = np.array(arr)
plt.plot(np.arange(arr.shape[0]), arr)
import scipy
import scipy.signal
themax = 0
for i in range(arr.shape[0]):
  if arr[i] >= arr[themax]:
    themax = i

peak = scipy.signal.find_peaks(h[:, themax], prominence = 9, distance = 10)

rmax = 0
for j in range(h.shape[0]):
  if h[j,themax] > h[rmax,themax]:
    rmax = j

fig = plt.figure(figsize=(20,20))
p = fig.add_subplot(1, 3, 2)
p.imshow(display, cmap = 'binary')
x = np.arange(start = -500, stop = 900, step = 1)
ceta = ymax*math.pi/180
r = xmax - RADIUS//2

y = (r - x*math.sin(ceta))/math.cos(ceta)
p.plot(x,y,zorder = 2, color = 'red', linewidth = 3)
plt.xlim(0,np.array(display).shape[1])
plt.ylim(0,np.array(display).shape[0])

p = fig.add_subplot(1,3, 3)
p.imshow(display, cmap = 'binary')
x = np.arange(start = -5000, stop = 9000, step = 1)

r = peak[0]
ceta = themax*math.pi/180
for j in range(len(r)):
  y = (r[j]- RADIUS//2 - x*math.sin(ceta))/math.cos(ceta)
  p.plot(x,y,zorder = 2, color = 'red', linewidth = 3)

plt.xlim(0,np.array(display).shape[1])
plt.ylim(0,np.array(display).shape[0])

p = fig.add_subplot(1,3,1)
p.imshow(img_2, cmap = 'binary')
plt.xlim(0,np.array(display).shape[1])
plt.ylim(0,np.array(display).shape[0])
