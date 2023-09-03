import numpy as np


class KMeans:
    def __init__(self, K: int, X) -> None:
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
    
    

