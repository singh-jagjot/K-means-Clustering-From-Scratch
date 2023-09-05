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

    # Initializing centroids randomly.
    def _init_centroids(self):
        m, _ = self.X.shape

        # Need this instead of randint() because this allows indices to be
        # selected at random without the risk of getting the same element twice.
        rand_idx = np.random.permutation(m)

        # Selecting K elements from X using rand_idx
        return self.X[rand_idx[: self.K]]

    # Computes a ndarray of indices of closest centroid for every X[i].
    # Closest centroid for X[i] is centroid_idx_[i].
    def _find_closest_centroid(self, centroids):
        K = centroids.shape[0]
        # centroid_idx = np.zeros(self.X.shape[0], dtype=int)
        # centroid_idx2 = np.zeros(self.X.shape[0], dtype=int)
        centroid_idx_ = np.zeros(self.X.shape[0], dtype=int)
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
        centroid_distances = np.zeros((K, self.X.shape[0]))
        for i in range(K):
            centroid_distances[i] = np.linalg.norm(self.X - centroids[i], axis=1)
        centroid_idx_ = np.argmin(centroid_distances, axis=0)
        # print("logic 3: ", time.time()-t)

        # print("     Orphan Centroids: ", K - np.unique(centroid_idx_).shape[0])
        return centroid_idx_

    # Computes a ndarray of new/better centroids.
    #
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

        # Stores the sum of every X[i] closest to centroid C[j].
        # i < X.shape[0], j < K
        centroid_sum_ = np.zeros_like(centroids)
        # Stores total number of X[i] closest to centroid C[j].
        # i < X.shape[0], j < K
        centroid_total_ele_ = np.zeros(centroids.shape[0])
        for i in range(self.K):
            temp = self.X[centroid_idx == i]
            centroid_sum_[i] = temp.sum(axis=0)
            centroid_total_ele_[i] = temp.shape[0]

        # Computing average to get updated centroid C[j].
        centroids_ = centroid_sum_ / centroid_total_ele_.reshape(
            (centroid_total_ele_.shape[0], 1)
        )
        # print("compute_centroids logic 2: ", time.time()-t)

        return np.nan_to_num(centroids_)

    # This method computes 'Distortion' or 'Cost'.
    def _compute_cost(self, centroids: np.ndarray, centroid_idx: np.ndarray):
        distances = np.linalg.norm(self.X - centroids[centroid_idx], axis=1)
        return np.mean(distances)

    # Starts the K-means algorithm.
    def optimize(self):
        min_distortion = np.inf
        best_vals = None
        best_iter = -1
        for iter in range(self.iters):
            centroid_idx = None
            # Initializing centroids randomly at start.
            centroids = self._init_centroids()
            cost = np.inf
            for itr in range(self.max_iters):
                centroid_idx = self._find_closest_centroid(centroids)
                centroids = self._compute_centroids(centroid_idx, centroids)
                current_cost = self._compute_cost(centroids, centroid_idx)
                print(
                    "K-means Iteration: {}, Iteration: {} Current Distortion: {}".format(
                        iter + 1, itr + 1, current_cost
                    )
                )

                # Breaking loop if converged early.
                if current_cost >= cost:
                    break
                cost = current_cost

            # Updating the variables if new best is found.
            if min_distortion > cost:
                min_distortion = cost
                best_vals = (centroids, centroid_idx, min_distortion)
                best_iter = iter + 1
            print(
                "K-means Iteration: {}, Distortion: {}, Best: ({}, {})\n".format(
                    iter + 1, cost, best_iter, min_distortion
                )
            )

        # Returning the best values.
        return best_vals
