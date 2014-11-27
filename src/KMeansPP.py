import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from KMeansBase import KMeansBase

class KMeansPP(KMeansBase):
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
            centroids = np.vstack([centroids, data[np.random.choice(range(self.data.shape[0]),1, p = prob_dist), :]])
            print(centroids.shape[0])
        return centroids


if __name__ == '__main__':
    k = 3
    #data = np.random.randn(100,2)
    data = np.array([[1.1,2],[1,2],[0.9,1.9],[1,2.1],[4,4],[4,4.1],[4.2,4.3],[4.3,4],[9,9],[8.9,9],[8.7,9.2],[9.1,9]])
    kmeans = KMeansPP(data, k)
    _, _, centroids, min_location = kmeans.cluster()
    # plotting code
    colors = iter(cm.rainbow(np.linspace(0, 1, k + 1)))
    plt.figure()
    for col in range (0,k):
            plt.scatter(data[min_location[:,col] == True, :][:,0], data[min_location[:,col] == True, :][:,1], color=next(colors))

    centroid_leg = plt.scatter(centroids[:,0], centroids[:,1], color=next(colors), marker='x')
    plt.legend([centroid_leg], ['Centroids'], scatterpoints=1, loc='best')
    plt.show()
