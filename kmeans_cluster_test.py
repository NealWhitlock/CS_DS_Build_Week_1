import numpy as np
from numpy import loadtxt
# from sklearn.datasets import make_blobs
from kmeans_cluster import KMeans
import matplotlib.pyplot as plt

X = loadtxt('X.csv', delimiter=',')
y = loadtxt('y.csv', delimiter=',')
# X, y = make_blobs(centers = 4, n_samples=500, n_features=2, shuffle=True, random_state=7)
print(X.shape)

X_test = X[:50]

clusters = len(np.unique(y))
print(clusters)

k = KMeans(k_clusters=clusters)
results = k.fit(X)
y_pred = k.predict(X_test)

# k.plot()

print('After K-Means:', results[:50])
print('Passing in 50 coordinates:', y_pred)

