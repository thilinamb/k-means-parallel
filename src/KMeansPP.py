import numpy as np
from KMeansBase import KMeansBase

class KMeansPP(KMeansBase):
    def __init__(self, data, k):
        KMeansBase.__init__(self, data, k)

    def _initial_centroids(self):
        # pick the initial centroid randomly
        centroids = self.data[np.random.choice(range(self.data.shape[0]),1), :]
        data_ex = self.data[:, np.newaxis, :]

        # run k - 1 passes through the data set to select the initial centroids
        while centroids.shape[0] < self.k :
            print (centroids)
            euclidean_dist = (data_ex - centroids) ** 2
            distance_arr = np.sum(euclidean_dist, axis=2)
            min_location = np.zeros(distance_arr.shape)
            min_location[range(distance_arr.shape[0]), np.argmin(distance_arr, axis=1)] = 1
            # calculate J
            j_val = np.sum(distance_arr[min_location == True])
            # calculate the probability distribution
            prob_dist = np.min(distance_arr, axis=1)/j_val
            # select the next centroid using the probability distribution calculated before
            centroids = np.vstack([centroids, self.data[np.random.choice(range(self.data.shape[0]),1, p = prob_dist), :]])
            print(centroids.shape[0])
        return centroids

