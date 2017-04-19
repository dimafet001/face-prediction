import numpy as numpy
import matplotlib.pyplot as plt
%matplotlib inline


# get a DB of the faces images
from sklearn.datasets import fetch_olivetti_faces
data = fetch_olivetti_faces().images



# making a huge graph out of small graphs
# plt.subplot()