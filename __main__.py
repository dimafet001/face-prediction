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
	# the types are numpy.ndarray
	left_half = left_half.reshape([-1, 64, 32])
	right_half = right_half.reshape([-1, 64, 32])
	return np.concatenate([left_half, right_half], axis=-1)

plt.imshow(glue(X,Y)[99], cmap='gray')
plt.show()

# in old version (0.8)
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split


# dividing the collection for test and train
X_train, X_test, Y_train, Y_test = train_test_split(X.reshape([len(X),-1]),
													Y.reshape([len(Y),-1]),
													test_size=.05, random_state=42)
# TODO: clarify the meaning of those reshapes

print(X_test.shape)

# training the algorithm
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)


# getting the error
from sklearn.metrics import mean_squared_error

print(mean_squared_error(Y_train, model.predict(X_train)))
print(mean_squared_error(Y_test, model.predict(X_test)))

# pictures on train (obviously the same)
pics = glue(X_train, model.predict(X_train))
plt.figure(figsize=[16, 12])
for i in range(20):
	plt.subplot(4, 5, i+1)
	plt.imshow(pics[i], cmap='gray')
plt.show()

# pictures on test
pics = glue(X_test, model.predict(X_test))
plt.figure(figsize=[16,12])
for i in range(20):
	plt.subplot(4, 5, i+1)
	plt.imshow(pics[i], cmap='gray')
plt.show()