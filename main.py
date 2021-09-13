import setup
import numpy as np
import matplotlib.pyplot as plt

path_dir = ''
coord_x, coord_y = ImageToCoord(path_dir)
H = HoughTransformLine(coord_x, coord_y)
h = np.array(H)
xmax, ymax = FindArgMax(H)
display = CoordToImage(coord_x, coord_y)

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
