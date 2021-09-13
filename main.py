import setup
import numpy as np
import matplotlib.pyplot as plt

path_dir = ''
coord_x, coord_y = ImageToCoord(path_dir)
H = HoughTransformLine(coord_x, coord_y)
h = np.array(H)
xmax, ymax = FindArgMax(H)
display = CoordToImage(coord_x, coord_y)



# for line
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




# for circle
argmax_CM = [np.unravel_index(np.argmax(h), h.shape) for r in h]
print(argmax_CM)

arr_1 = [[0 for j in range(HEIGHT)] for i in range(WIDTH)]
arr_3 = [[0 for j in range(HEIGHT)] for i in range(WIDTH)]
arr_1 = np.array(arr_1)
arr_3 = np.array(arr_3)
for i in range(HEIGHT):
  for j in range(WIDTH):
    for k in range(400):
      arr_1[i,j] += h[i,j,k+1] * h[i,j,k+1]
      arr_3[i,j] += h[i,j,k+1]

plt.imshow(edges_2)
plt.show()

a = np.sum(h*h, axis=2)[400:400 + height,400:400+width]
argmax = [np.unravel_index(np.argmax(a), a.shape) for r in a]

print(argmax)

#circle
plt.imshow(np.sum(np.array(H)*np.array(H), axis = 2)[400:,400:], cmap = 'Reds')
plt.xlim(0,width)
plt.ylim(height,0)
plt.colorbar()
plt.show()

plt.imshow(np.array(edges_2), cmap = 'binary')
plt.xlim(0,width)
plt.ylim(height,0)
ax = plt.gca()

max_radius = 0
for i in range(100):
  if h[400+argmax[0][0],400+argmax[0][1],i] > h[400+argmax[0][0],400+argmax[0][1],max_radius]:
    max_radius = i

circle1 = plt.Circle((argmax[0][1], argmax[0][0]), max_radius, color = 'r', fill=False)
ax.add_patch(circle1)
