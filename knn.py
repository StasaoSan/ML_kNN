import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter


def uniform_kernel(distances):
    return np.ones_like(distances)


def gaussian_kernel(distances, bandwidth=1.0):
    return np.exp(-0.5 * (distances / bandwidth) ** 2)


def polynomial_kernel(distances, a=2, b=1):
    return np.power((1 - np.power(np.abs(distances), a)), b)


def triangular_kernel(distances):
    return np.maximum(1 - distances, 0)


KERNELS = {
    'uniform': uniform_kernel,
    'gaussian': gaussian_kernel,
    'triangular': triangular_kernel,
    'polynomial': polynomial_kernel
}


class KNN:
    def __init__(self, n_neighbors=5, radius=None, metric='euclidean', kernel='uniform', weights=None, p=2):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.metric = metric
        self.kernel = KERNELS.get(kernel, uniform_kernel)
        self.weights = weights if weights is not None else np.ones(1)
        self.p = p
        if radius is None:
            if self.metric == 'minkowski':
                self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric, p=self.p)
            else:
                self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric)
        else:
            if self.metric == 'minkowski':
                self.model = NearestNeighbors(radius=radius, metric=self.metric, p=self.p)
            else:
                self.model = NearestNeighbors(radius=radius, metric=self.metric)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.model.fit(X)

    def predict(self, X):
        predictions = []
        if self.radius is None:
            distances, indices = self.model.kneighbors(X)
            for i, neighbors in enumerate(indices):
                neighbor_labels = self.y_train[neighbors]
                neighbor_distances = distances[i]
                weights = self.kernel(neighbor_distances)
                if self.weights.size > 1:
                    weights *= self.weights[neighbors]
                weighted_votes = Counter()
                for label, weight in zip(neighbor_labels, weights):
                    weighted_votes[label] += weight
                most_common = weighted_votes.most_common(1)[0][0]
                predictions.append(most_common)
        else:
            neighbors = self.model.radius_neighbors(X, return_distance=True)
            for i in range(len(X)):
                neighbor_indices = neighbors[1][i]
                neighbor_distances = neighbors[0][i]
                if len(neighbor_indices) == 0:
                    distances, indices = self.model.kneighbors(X[i].reshape(1, -1))
                    neighbor_indices = indices[0]
                    neighbor_distances = distances[0]
                neighbor_labels = self.y_train[neighbor_indices]
                weights = self.kernel(neighbor_distances)
                if self.weights.size > 1:
                    weights *= self.weights[neighbor_indices]
                weighted_votes = Counter()
                for label, weight in zip(neighbor_labels, weights):
                    weighted_votes[label] += weight
                most_common = weighted_votes.most_common(1)[0][0]
                predictions.append(most_common)
        return np.array(predictions)
