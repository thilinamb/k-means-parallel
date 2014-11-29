import numpy as np
from KMeansBase import KMeansBase
from KMeansPP import KMeansPP

class ScalableKMeansPP(KMeansBase):
    def __init__(self, data, k, l, r):
        KMeansBase.__init__(self, data, k)
        self.l = l
        self.r = r

    def _initial_centroids(self):
        # pick the initial centroid randomly
        centroids = self.data[np.random.choice(range(self.data.shape[0]),1), :]
        data_ex = self.data[:, np.newaxis, :]

        passes = 0
        while passes < self.r:
            euclidean_dist = (data_ex - centroids) ** 2
            distance_arr = np.sum(euclidean_dist, axis=2)
            # find the minimum distance, this will be the weight
            min = np.min(distance_arr, axis=1).reshape(-1, 1)
            # let's use weighted reservoir sampling algorithm to select l centroids
            random_numbers = np.random.rand(min.shape[0], min.shape[1])
            # replace zeros in min if available with the lowest positive float in Python
            min[np.where(min==0)] = np.nextafter(0,1)
            # take the n^th root of random numbers where n is the weights
            with np.errstate(all='ignore'):
                random_numbers = random_numbers ** (1.0/min)
            # pick the highest l
            cent = self.data[np.argsort(random_numbers, axis=0)[:, 0]][::-1][:self.l, :]
            # combine the new set of centroids with the previous set
            centroids = np.vstack((centroids, cent))
            passes += 1
        # now we have the initial set of centroids which is higher than k.
        # we should reduce this to k using scalable K-Means++
        euclidean_dist = (data_ex - centroids) ** 2
        distance_arr = np.sum(euclidean_dist, axis=2)
        min_location = np.zeros(distance_arr.shape)
        min_location[range(distance_arr.shape[0]), np.argmin(distance_arr, axis=1)] = 1
        weights = np.array([np.count_nonzero(min_location[:, col]) for col in range(centroids.shape[0])]).reshape(-1,1)
        # cluster these r*l + 1 points with K-Means++ to get K points
        kmeans_pp = KMeansPP(weights, self.k)
        _, _, _, min_locations = kmeans_pp.cluster()
        # calculates the new centroids
        new_centroids = np.empty((self.k, self.data.shape[1]))
        for col in range(0, self.k):
            new_centroids[col] = np.mean(centroids[min_locations[:, col] == True, :], axis=0)
        return new_centroids

