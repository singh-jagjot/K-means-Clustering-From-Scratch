import numpy as np


class KMeans:
    def __init__(self, K: int, X:np.ndarray) -> None:
        self.K = K
        self.X = X
        self.centroids = self._init_centroids()

    def _init_centroids(self):
        m, _ = self.X.shape

        # Need this instead of randint() because this allows indices to be 
        # selected at random without the risk of getting the same element twice.
        rand_idx = np.random.permutation(m)
        
        # Selecting K elements from X using rand_idx
        return self.X[rand_idx[:self.K]]
    
    def _find_closest_centroid(self):
        centroid_idx = np.zeros(self.X.shape[0], dtype=int)
        for i in range(self.X.shape[0]):
            distance = float('inf')
            for j in range(self.centroids.shape):
                l2norm = np.linalg.norm(self.X[i] - self.centroid[j])
                if l2norm < distance:
                    distance = l2norm
                    centroid_idx[i] = j

        return centroid_idx
    
    

