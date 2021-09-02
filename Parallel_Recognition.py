import math
import matplotlib.pyplot as plt
import numpy as np

WIDTH = 1000
HEIGHT = 1000
maximum_radius = 100
radius = 30
offset1 = 50
offset2 = 50
offset3 = 60
offset4 = 70
rad2 = 50
rad3 = 40
rad4 = 8


#################
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from google.colab import drive
drive.mount('/content/drive')

#################
def window(image, min_in, max_in):
    image = np.array(list(map(lambda x: 50+(x-min_in)*500/(max_in-min_in), image)))
    image = image.astype("float32")
    return image


#################
film_depth = window(film_range, 43, 174)

edges = edges[:2000,:2000]

plt.imshow(edges, cmap = 'gray')

edges = edges[::10, ::10]
plt.imshow(edges, cmap = 'gray')

for i in range

img_2[::3, ::3].shape

#################
img = cv.imread('/content/drive/MyDrive/Colab Notebooks/Fourior Transform/FT.jpg', 0)
img = cv.Canny(img,50,100)
plt.imshow(img)
plt.show()
thresh = 0
#get threshold image
ret,thresh_img = cv.threshold(img, thresh, 255, cv.THRESH_BINARY)
#find contours
contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#################

#create an empty image for contours
img_contours = np.zeros(img.shape)
# draw the contours on the empty image
cv.drawContours(img_contours, contours, -1, (0,255,0), 3)
plt.imshow(img_contours)

#################
img_2 = cv.imread('/content/drive/MyDrive/Colab Notebooks/IMG_2038.PNG',0)
img_2 = img_2[::3,::3]
img_2 = img_2[200:500,100:400]
edges_2 = cv.Canny(img_2,100,100)
plt.imshow(img_2,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(edges_2,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

img.shape

k = np.array(img_2)[::3,::3]
k.shape

for i in range(k.shape[0]):
  for j in range(k.shape[1]):
    if k[i,j] != 255:
      img[100+i][150+j]=k[i,j]

#################
plt.imshow(img, cmap ='gray')

edges = cv.Canny(img,100,100)
plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

edges.shape

edges = edges[:,60:-60]

import scipy.ndimage

edges = scipy.ndimage.interpolation.zoom(edges, [100/360, 100/360], order=1)
print(edges.shape)

#################
#input picture(2D-array)
WIDTH = edges.shape[0]
HEIGHT = edges.shape[1]
coord_x = []
coord_y = []
for i in range(WIDTH):
  for j in range(HEIGHT):
    if edges[i][j]!=0:
      coord_x.append(i)
      coord_y.append(j)

#################
#circle
coord_x = [ 10, 10, 18, 26, 20, 36, 36, 42, 42, 50, 43, 50, 57, 50, 58, 74, 78, 82, 82, 84, 84, 90, 90, 88, 90, 90]
coord_y = [ 50, 80, 26, 18, 10, 52, 88, 46, 50, 10, 46, 42, 46, 58, 50, 18, 46, 26, 74, 52, 88, 20, 50, 72, 70, 80]

#line 
coord_x = [55,60,65,15,20,25,85]
coord_y = [10,20,30, 5,15,35,70]

#elliptic curve
coord_x = [50,  5, 84,     11, 3, 64, 90]
coord_y = [36, 35, 37,      5, 2, 16, 20]

#parabola
#a=1, h=10, k=10
#a=4, h= 7, k= 5
coord_x = [10, 14, 14, 35, 59, 91, 5, 41, 69]
coord_y = [10, 12, 8 , 15, 17, 19, 7, 10,  3]


#################
for i in range(len(coord_x)):
  coord_x[i] += 400
  coord_y[i] += 400

len(coord_x)

print(coord_x)

print(coord_y)

#output circle
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

len(coord_x)

from matplotlib.pyplot import figure

figure(figsize=(7, 7), dpi=90)
show = edges[:, :380]
show = np.flip(show, axis =0)
plt.imshow(show, cmap = 'binary')

show.shape

len(coord_x)

#circle
n = 100
H = [[[0 for k in range(300)] for j in range(HEIGHT)] for i in range(WIDTH)]
for i in range(len(coord_x)):
  for x in range(WIDTH):
    for y in range(HEIGHT):
      r = math.floor(math.sqrt((coord_x[i]-x)*(coord_x[i]-x)+(coord_y[i]-y)*(coord_y[i]-y)))
      if r<299 and r>=0:
        H[x][y][r]+=1
        H[x][y][r+1]+=1
        H[x][y][r-1]+=1

#H[r][theta]
#line
THETA = 180
RADIUS = 1200
H = [[0 for k in range(THETA)] for j in range(RADIUS)]
for i in range(len(coord_x)):
  for x in range(THETA):
    r = math.floor((coord_x[i])*math.cos(x*math.pi/180) + (coord_y[i])*math.sin(x*math.pi/180))
    if r<RADIUS//2 and r>-RADIUS//2:
      H[r+RADIUS//2][x]+=1
    print(r," ", x, " ",i)

#line
figure(figsize = (10,7), dpi = 100)

plt.imshow(np.array(H), cmap = 'gray', zorder = 1 , aspect= 0.1)

h = np.array(H)
# h.shape = 300 x 360
#extract r and theta from argmax

from numpy import savetxt
savetxt('/content/drive/MyDrive/Colab Notebooks/circle_hough.csv', h, delimiter=',')

h = np.array(H)
argmax = [np.unravel_index(np.argmax(h), h.shape) for r in h]
print(argmax)

print(h.shape)

fig = plt.figure(figsize=(20,20))
p = fig.add_subplot(1, 3, 3)
p.imshow(display, cmap = 'binary', zorder = 1)
x = np.arange(start = -500, stop = 500, step = 1)

ceta = 60*math.pi/180
r = 830 - RADIUS//2
y = (r - x*math.sin(ceta))/math.cos(ceta)
p.plot(x,y,zorder = 2, color = 'red', linewidth = 3)

ceta = 100*math.pi/180
r = 810 - RADIUS//2
y = (r - x*math.sin(ceta))/math.cos(ceta)
p.plot(x,y,zorder = 4, color = 'red',linewidth = 3)
plt.xlim(0, 380)
plt.ylim(0, 360)

p = fig.add_subplot(1, 3, 2)
p.imshow(np.flip(show, axis = 0), cmap = 'binary')
plt.xlim(0, 380)
plt.ylim(0, 360)
ceta = 174*math.pi/180
r = 462 - RADIUS//2
y = (r - x*math.sin(ceta))/math.cos(ceta)
p.plot(x,y,zorder = 3, color = 'red', linewidth = 3)

p = fig.add_subplot(1, 3, 1)
p.imshow(img[:,:380], cmap='Greens_r')
plt.xlim(0, 380)
plt.ylim(0, 360)

#################

def arg_max(H): # O(n^3)
  maxi = 0
  maxx = 0
  maxy = 0
  for i in range(WIDTH):
    for x in range(HEIGHT):
      for y in range(200):
        if H[i][x][y] > H[maxi][maxx][maxy]:
          maxi = i
          maxx = x
          maxy = y
  return maxi, maxx, maxy


#################
plt.imshow(display, cmap = 'binary')
plt.xlim(400,500)
plt.ylim(400,500)

np.argmax(h[450,450,:])

plt.scatter(np.arange(60),h[450,450,:60])


plt.imshow(np.sum(h*h, axis=2))

np.sum(display)


#################
#circle
plt.imshow(np.sum(np.array(H)*np.array(H), axis = 2)[400:,400:], cmap = 'Reds')
plt.xlim(0,100)
plt.ylim(0,100)
plt.colorbar()
plt.show()

plt.imshow(np.array(display)[400:,400:], cmap = 'binary')
plt.xlim(0,100)
plt.ylim(0,100)
ax = plt.gca()
circle1 = plt.Circle((50, 50), 39, color = 'r', fill=False)
circle2 = plt.Circle((50, 50), 50, color = 'r', fill=False)
circle3 = plt.Circle((50, 50), 8, color = 'r', fill=False)
circle4 = plt.Circle((69, 60), 29, color = 'r', fill=False)
#ax.add_patch(circle1)
#ax.add_patch(circle2)
#ax.add_patch(circle3)
ax.add_patch(circle4)

#################

k = np.sum(h*h, axis = 0, keepdims=True)
a = np.arange(np.squeeze(k).shape[0])
plt.plot(a,np.squeeze(k))
plt.xlim(0,180)

plt.xlabel('Theta')
plt.ylabel('Accumulator')

from scipy.interpolate import interp1d
k = np.sum(h*h, axis = 0, keepdims=True)
print(k.shape)
a = np.arange(0,180)
print(a.shape)
f = interp1d(a, np.squeeze(k))
b = f(a)
plt.plot(a,b)
plt.xlim(0,180)
plt.xlabel('Theta')
plt.ylabel('Accumulator')


#################
#elliptic curve
#H[a][b]
#y^2=x^3+ax+b
a = 1
#fix a=1
#b  = -130000 to 130000
A = 20
n = 1000000

H = [[0 for k in range(2*n)] for j in range(A)]
print(len(H))
print(len(H[0]))

for i in range(len(coord_x)):
  for a in range(A):    
    for b in range(2*n-1):
      if coord_x[i]*coord_x[i]*coord_x[i]+a*coord_x[i]+b-n >= 0:
        y = math.floor(math.sqrt(coord_x[i]*coord_x[i]*coord_x[i]+a*coord_x[i]+b-n))
        if coord_y[i] == y:
          H[a][b]+=1
          H[a][b+1]+=1


#################
#H[a][h][k]
#parabola
A = 50000
# a = [-10,10] ###
step = 1e-1

X = 100
Y = 100
H = [[[0 for k in range(Y)] for j in range(X)] for k in range(A)]
for index in range(len(coord_x)):
  for i in range(X):
    for j in range(Y):
      if coord_y[index] != i:
        a = math.floor((j+coord_x[index])/(step*(coord_y[index]-i)*(coord_y[index]-i)))
        if a<25000:
          H[a+25000][i][j]+=1


#################

SumALongA = []
for j in range(A):
  sub = 0
  for i in range(X):
    for k in range(X):
      sub += H[j][i][k]
  SumALongA.append(sub)

#################
SumALongHK = [[0 for i in range(100)] for j in range(100)]
for j in range(100):
  for i in range(100):
    sum = 0
    for k in range(A):
      sum += H[k][j][i] * H[k][j][i]
    SumALongHK[i][j] += sum

#################
plt.imshow(SumALongHK, cmap = 'gray')
plt.colorbar()
plt.show()
plt.imshow(display)

#################
SumALongA = np.array(SumALongA)

y = np.arange(20000)
plt.plot(y,SumALongA)

plt.imshow(h)


#################
p = []
for i in range(200):
  sub = 0
  for j in range(360):
    sub+=h[i,j]
  p.append(sub)
  

#################
p = []
for i in range(A):
  sub = 0
  for j in range(n):
    sub+=h[i,j]*h[i,j]
  p.append(sub)


#################
p = np.array(p)
y = np.arange(start = 0, stop = 20, step = 1)
plt.plot(y,p)


#################
plt.plot(np.arange(start = -n, stop = n, step = 1),h[1,:])

#################
h_r = []
for k in range(200):
  sum = 0
  for i in range(WIDTH):
    for j in range(HEIGHT):
      sum+=h[i,j,k]
  h_r.append(sum)


#################
nparr = np.array(H)

result = np.sum(np.array(H)*np.array(H), axis = 2)
results = np.sum(np.array(H)*np.array(H)*np.array(H), axis = 2)
print(result[60,70]/result[50,50])
print(results[60,70]/results[50,50])


#################

plt.imshow(re)
plt.colorbar()
plt.show()
plt.imshow(res)
plt.colorbar()

a.shape


#################

from matplotlib.pyplot import figure
figure(figsize=(8, 8), dpi=80)
plt.imshow(a)
plt.colorbar()
plt.show()


figure(figsize=(7, 7), dpi=80)
plt.imshow(display)


#################
maxi = 0
maxx = 0
maxy = 0
for i in range(WIDTH):
  for x in range(HEIGHT):
    for y in range(200):
      if H[i][x][y] > H[maxi][maxx][maxy]:
        maxi = i
        maxx = x
        maxy = y

#################
display2 = []

# create circle
for w in range(WIDTH):
  # append [x,y]:
  sub_y = []
  for h in range(HEIGHT):
    if abs(maxy*maxy-((w-maxi)*(w-maxi)+(h-maxx)*(h-maxx)))< 10:
      sub_y.append(1)
    else :
      sub_y.append(0)
  display2.append(sub_y)


#################
def mixup(segmentation1,segmentation2):
    seggg = segmentation1 * 0.6 + segmentation2 * 0.4
    return seggg

merge = mixup(Display2,Display)


#################
from matplotlib.pyplot import figure

figure(figsize=(7, 7), dpi=80)
plt.imshow(merge, cmap = 'gist_heat')

from matplotlib.pyplot import figure

figure(figsize=(7, 7), dpi=80)
plt.imshow(display2)
figure(figsize=(7, 7), dpi=80)
plt.imshow(display)



#################
def picture_to_array(input_pic): # O(n^2)
  #input picture(2D-array)
  coord_x = []
  coord_y = []
  for i in range(WIDTH):
    for j in range(HEIGHT):
      if input_pic[i][j]!=0:
        coord_x.append(i)
        coord_y.append(j)
  return coord_x, coord_y


#################
def find_H(coord_x , coord_y): # O(n^3)
  H = [[[0 for k in range(200)] for j in range(HEIGHT)] for i in range(WIDTH)]
  for i in range(len(coord_x)):
    for x in range(WIDTH):
      for y in range(HEIGHT):
        r = math.floor(math.sqrt((coord_x[i]-x)*(coord_x[i]-x)+(coord_y[i]-y)*(coord_y[i]-y)))
        if r<200 and r>=0:
          H[x][y][r]+=1
          H[x][y][r+1]+=1
          H[x][y][r-1]+=1
  return H


#################
def arg_max(H): # O(n^3)
  maxi = 0
  maxx = 0
  maxy = 0
  for i in range(WIDTH):
    for x in range(HEIGHT):
      for y in range(200):
        if H[i][x][y] > H[maxi][maxx][maxy]:
          maxi = i
          maxx = x
          maxy = y
  return maxi, maxx, maxy