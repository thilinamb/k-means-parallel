import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class KMeansBase:
    def __init__(self, data, k):
        self.data = data
        self.k = k

    def cluster(self):
        return self._lloyds_iterations()

    def _initial_centroids(self):
        # get the initial set of centroids
        # get k random numbers between 0 and the number of rows in the data set
        centroid_indexes = np.random.choice(range(self.data.shape[0]), self.k, replace=False)
        # get the corresponding data points
        return self.data[centroid_indexes, :]

    def _lloyds_iterations(self):
        centroids = self._initial_centroids()
        print('Initial Centroids:', centroids)

        stabilized = False

        j_values = []
        iterations = 0
        while not stabilized:
            iterations += 1
            if iterations%1000 == 0:
                print ('Iteration: ', iterations)
            # find the Euclidean distance between a center and a data point
            # centroids array shape = k x m
            # data array shape = n x m
            # In order to broadcast it, we have to introduce a third dimension into data
            # data array becomes n x 1 x m
            # now as a result of broadcasting, both array sizes will be n x k x m
            data_ex = self.data[:, np.newaxis, :]
            euclidean_dist = (data_ex - centroids) ** 2
            # now take the summation of all distances along the 3rd axis(length of the dimension is m).
            # This will be the total distance from each centroid for each data point.
            # resulting vector will be of size n x k
            distance_arr = np.sum(euclidean_dist, axis=2)

            # now we need to find out to which cluster each data point belongs.
            # Use a matrix of n x k where [i,j] = 1 if the ith data point belongs
            # to cluster j.
            min_location = np.zeros(distance_arr.shape)
            min_location[range(distance_arr.shape[0]), np.argmin(distance_arr, axis=1)] = 1

            # calculate J
            j_val = np.sum(distance_arr[min_location == True])
            j_values.append(j_val)

            # calculates the new centroids
            new_centroids = np.empty(centroids.shape)
            for col in range(0, self.k):
                new_centroids[col] = np.mean(self.data[min_location[:, col] == True, :], axis=0)

            # compare centroids to see if they are equal or not
            if self._compare_centroids(centroids, new_centroids):
                # it has resulted in the same centroids.
                stabilized = True
            else:
                centroids = new_centroids


        print ('Required ', iterations, ' iterations to stabilize.')
        return iterations, j_values, centroids, min_location

    def _compare_centroids(self, old_centroids, new_centroids, precision=-1):
        if precision == -1:
            return np.array_equal(old_centroids, new_centroids)
        else:
            diff = np.sum((new_centroids - old_centroids)**2, axis=1)
            if np.max(diff) <= precision:
                return True
            else:
                return False

if __name__ == '__main__':
    k = 3
    data = np.random.randn(100,2)
    kmeans = KMeansBase(data, k)
    _, _, centroids, min_location = kmeans.cluster()
    # plotting code
    colors = iter(cm.rainbow(np.linspace(0, 1, k + 1)))
    plt.figure()
    for col in range (0,k):
            plt.scatter(data[min_location[:,col] == True, :][:,0], data[min_location[:,col] == True, :][:,1], color=next(colors))

    centroid_leg = plt.scatter(centroids[:,0], centroids[:,1], color=next(colors), marker='x')
    plt.legend([centroid_leg], ['Centroids'], scatterpoints=1, loc='best')
    plt.show()

