import numpy as np
import time


class KMeans:
    def __init__(self, X: np.ndarray, K: int, max_iters: int, iters: int = 1) -> None:
        """X: Vector for using K-means.
            K: Number of clusters.
            max_iters: Maximum number of iterations for a single run.
            iters: Maximum number of times to run K-means. Final result
                  will have best(have low distortion) of 'iters' runs.
                  Default = 1.
        """
        self.X = X
        self.K = K
        self.max_iters = max_iters
        self.iters = iters

    def _init_centroids(self):
        m, _ = self.X.shape

        # Need this instead of randint() because this allows indices to be
        # selected at random without the risk of getting the same element twice.
        rand_idx = np.random.permutation(m)

        # Selecting K elements from X using rand_idx
        return self.X[rand_idx[:self.K]]

    def _init_centroids_kmeans_pp(self):
        centroids = np.zeros((self.K, ))
        pass

    def _find_closest_centroid(self, centroids):
        K = centroids.shape[0]
        # centroid_idx = np.zeros(self.X.shape[0], dtype=int)
        # centroid_idx2 = np.zeros(self.X.shape[0], dtype=int)
        centroid_idx3 = np.zeros(self.X.shape[0], dtype=int)
        # t = time.time()
        # for i in range(self.X.shape[0]):
        #     min_distance = float('inf')
        #     for j in range(K):
        #         l2norm = np.linalg.norm(self.X[i] - centroids[j])
        #         if l2norm < min_distance:
        #             min_distance = l2norm
        #             centroid_idx[i] = j
        # print("logic 1: ", time.time()-t)

        # t = time.time()
        # for i in range(self.X.shape[0]):
        #     centroid_idx2[i] = np.argmin(np.linalg.norm(centroids - self.X[i], axis=1))
        # print("logic 2: ", time.time()-t)

        # t = time.time()
        ci = np.zeros((K, self.X.shape[0]))
        for i in range(K):
            ci[i] = np.linalg.norm(self.X - centroids[i], axis=1)
        centroid_idx3 = np.argmin(ci, axis=0)
        # print("logic 3: ", time.time()-t)

        # print("1 == 2: ", np.array_equal(centroid_idx, centroid_idx2))
        # print("1 == 3: ", np.array_equal(centroid_idx, centroid_idx3))
        # print("2 == 3: ", np.array_equal(centroid_idx2, centroid_idx3))
        # print("     Orphan Centroids: ", K - np.unique(centroid_idx3).shape[0])
        return centroid_idx3

    def _compute_centroids(self, centroid_idx: np.ndarray, centroids: np.ndarray):
        # Slow Logic as loop runs for every X.shape[0]
        # t = time.time()
        # centroid_sum = np.zeros_like(centroids)
        # centroid_total_ele = np.zeros(centroids.shape[0])
        # for i in range(self.X.shape[0]):
        #     centroid_sum[centroid_idx[i]] += self.X[i]
        #     centroid_total_ele[centroid_idx[i]] += 1
        # centroids = centroid_sum / centroid_total_ele.reshape((centroid_total_ele.shape[0], 1))
        # print("compute_centroids logic 1: ", time.time()-t)

        # Fast Logic as loop runs for every K. Assuming K << X.shape[0].
        # t = time.time()
        centroid_sum_ = np.zeros_like(centroids)
        centroid_total_ele_ = np.zeros(centroids.shape[0])
        for i in range(self.K):
            centroid_sum_[i] = self.X[centroid_idx == i].sum(axis=0)
            centroid_total_ele_[i] = self.X[centroid_idx == i].shape[0]
        centroids_ = centroid_sum_ / \
            centroid_total_ele_.reshape((centroid_total_ele_.shape[0], 1))
        # print("compute_centroids logic 2: ", time.time()-t)

        return np.nan_to_num(centroids_)

    # This method computes 'distortion'
    def _compute_cost(self, centroids: np.ndarray, centroid_idx: np.ndarray):
        distances = np.linalg.norm(self.X - centroids[centroid_idx], axis=1)
        return np.mean(distances)

    def optimize(self):
        min_distortion = np.inf
        best_vals = None
        best_iter = -1
        for iter in range(self.iters):
            centroid_idx = None
            centroids = self._init_centroids()
            cost = np.inf
            for i in range(self.max_iters):
                # print(" Iteration: ", i+1)
                centroid_idx = self._find_closest_centroid(centroids)
                centroids = self._compute_centroids(centroid_idx, centroids)
                current_cost = self._compute_cost(centroids, centroid_idx)
                # print(" Distortion: ", cost)
                if current_cost >= cost:
                    break
                cost = current_cost
            if min_distortion > cost:
                min_distortion = cost
                best_vals = (centroids, centroid_idx, min_distortion)
                best_iter = iter + 1
            print("K-means Iteration: {}, Distortion: {}, Best: ({}, {})".format(iter +
                  1, cost, best_iter, min_distortion))
        return best_vals


