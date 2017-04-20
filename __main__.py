import numpy as np
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


# shows the example of divided face
# plt.subplot(1, 2, 1)
# plt.imshow(X[0], cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(Y[0], cmap='gray')
# plt.show()

# join two pics in one
def glue(left_half, right_half):
	left_half = left_half.reshape([-1, 64, 32])
	right_half = right_half.reshape([-1, 64, 32])
	return np.concatenate([left_half, right_half], axis=-1)

plt.imshow(glue(X,Y)[99], cmap='gray')
plt.show()