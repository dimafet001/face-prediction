import numpy as numpy
import matplotlib.pyplot as plt
# %matplotlib inline


# get a DB of the faces images. Has 10 of one face from different angles
# 400 pics overall
from sklearn.datasets import fetch_olivetti_faces
data = fetch_olivetti_faces().images



# making a huge graph out of small graphs
# plt.subplot(num of cells in height, num of cells in width, the # id of the cell)

# Shows a face from 4 angles a bit inclined from straight
# for i in range(4):
	# plt.subplot(2, 2, i+1)
	# plt.imshow(data[i], cmap='gray')

# plt.show()

# cutting the left and right halves of the images
X = data[:,:, : len(data[0][0])/2 ]
Y = data[:,:,   len(data[0][0])/2 : ]

plt.subplot(1, 2, 1)
plt.imshow(X[0], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(Y[0], cmap='gray')
plt.show()