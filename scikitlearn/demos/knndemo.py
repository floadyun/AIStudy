from sklearn.neighbors import NearestNeighbors
import numpy as np

X = np.array([[2, 1], [3, 5], [4, 6]])
model = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = model.kneighbors(X)
print('distances=', distances)
print('indices=', indices)

