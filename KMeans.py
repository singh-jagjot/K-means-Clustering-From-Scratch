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
            for j in range(self.centroids.shape[0]):
                l2norm = np.linalg.norm(self.X[i] - self.centroid[j])
                if l2norm < distance:
                    distance = l2norm
                    centroid_idx[i] = j

        return centroid_idx
    

    def _compute_centroids(self):
        centroid_idx = self._compute_centroids()
        centroid_sum = np.zeros_like(self.centroids)
        centroid_total_ele = np.zeros(self.centroids.shape[0])
        for i in range(self.X.shape[0]):
            centroid_sum[centroid_idx[i]] += self.X[i] 
            centroid_total_ele[i] += 1
        return centroid_sum / centroid_total_ele
    
    
