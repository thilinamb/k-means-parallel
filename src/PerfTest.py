from KMeansBase import KMeansBase
from KMeansPP import KMeansPP
from ScalableKMeansPP import ScalableKMeansPP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas

def samplesize_initcost():
    k = 10
    sampleSizes = np.linspace(100000, 1000000, 10)
    # sampleSizes = [1000, 2000, 3000, 4000, 5000, 6000]
    kmeans_perf = []
    kmeanspp_perf = []
    kmeansscalble_perf = []
    for n in sampleSizes:
        data = np.random.randn(n, 2)
        print('Sample Size:', n)
        print('KMeans')
        kmeans = KMeansBase(data, k)
        kmeans_perf.append(kmeans.initCost() * 100)
        print('KMeans++')
        kmeans_pp = KMeansPP(data, k)
        kmeanspp_perf.append(kmeans_pp.initCost() * 100)
        print('Scalable KMeans++')
        kmeans_scalable = ScalableKMeansPP(data, k, 4, 3)
        kmeansscalble_perf.append(kmeans_scalable.initCost() * 100)

    # plot
    plt.figure(figsize=(10, 10))
    plt.plot(sampleSizes, kmeans_perf, '-o', lw=3, markersize=10)
    plt.plot(sampleSizes, kmeanspp_perf, '-o', lw=3, markersize=10)
    plt.plot(sampleSizes, kmeansscalble_perf, '-o', lw=3, markersize=10)
    plt.legend(('KMeans', 'KMeans++', 'Scalable KMeans++'), prop={'size': 18}, loc=0)
    plt.xlabel('Sample Size', fontsize=18)
    plt.ylabel('Initialization Cost (ms)', fontsize=18)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=16)
    ax.xaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.yaxis.set_tick_params(labelsize=16)
    plt.savefig('samples-initcost.png')
    plt.close()


def clusters_initcost():
    n = 500000
    clusters = np.linspace(10, 50, 5)
    # sampleSizes = [1000, 2000, 3000, 4000, 5000, 6000]
    kmeans_perf = []
    kmeanspp_perf = []
    kmeansscalble_perf = []
    for k1 in clusters:
        k = int(k1)
        data = np.random.randn(n, 2)
        print('k:', k)
        print('KMeans')
        kmeans = KMeansBase(data, k)
        kmeans_perf.append(kmeans.initCost() * 100)
        print('KMeans++')
        kmeans_pp = KMeansPP(data, k)
        kmeanspp_perf.append(kmeans_pp.initCost() * 100)
        print('Scalable KMeans++')
        kmeans_scalable = ScalableKMeansPP(data, k, 11, 5)
        kmeansscalble_perf.append(kmeans_scalable.initCost() * 100)

    # plot
    plt.figure(figsize=(10, 10))
    plt.plot(clusters, kmeans_perf, '-o', lw=3, markersize=10)
    plt.plot(clusters, kmeanspp_perf, '-o', lw=3, markersize=10)
    plt.plot(clusters, kmeansscalble_perf, '-o', lw=3, markersize=10)
    plt.legend(('KMeans', 'KMeans++', 'Scalable KMeans++'), prop={'size': 18}, loc=0)
    plt.xlabel('Number of Clusters (K)', fontsize=18)
    plt.ylabel('Initialization Cost (ms)', fontsize=18)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=16)
    #ax.xaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.yaxis.set_tick_params(labelsize=16)
    plt.savefig('k-initcost.png')
    plt.close()


def no_of_iterations(n):
    mean = [0, 1, 2]
    cov = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
    data0 = np.random.multivariate_normal(mean, cov, n)
    data0 = np.hstack((data0, np.ones((data0.shape[0],1))))

    mean1 = [6, 8, 9]
    cov1 = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
    data1 = np.random.multivariate_normal(mean1, cov1, n)
    data1 = np.hstack((data1, np.ones((data1.shape[0],1)) * 2))

    mean2 = [15, 18, 19]
    cov2 = [[1, 0.5,0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
    data2 = np.random.multivariate_normal(mean2, cov2, n)
    data2 = np.hstack((data2, np.ones((data2.shape[0],1)) * 3))

    mean3 = [25, 26, 27]
    cov3 = [[1, 0.5,0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
    data3 = np.random.multivariate_normal(mean3, cov3, n)
    data3 = np.hstack((data3, np.ones((data3.shape[0],1)) * 4))

    data = np.vstack((data0, data1, data2, data3))
    np.random.shuffle(data)
    print (data.shape)
    return data


def itr_vs_sampels():
    k = 4
    r = 3
    n_from_each_cluster = np.linspace(10000, 100000, 10)
    kmeans_mean = []
    kmeanspp_mean = []
    kmeansscalable_mean1 = []
    kmeansscalable_mean2 = []
    kmeansscalable_mean3 = []
    for e in n_from_each_cluster:
        n = int(e)
        kmeans_itr = []
        kmeanspp_itr = []
        kmeansscalable_itr1 = []
        kmeansscalable_itr2 = []
        kmeansscalable_itr3 = []
        for i in range(20):
            print ("n: ",n, ' i:', i)
            data = no_of_iterations(n)

            kmeans = KMeansBase(data[:,:4], k)
            iterations, j_values, centroids, min_location = kmeans.cluster()
            kmeans_itr.append(iterations)

            kmeans = KMeansPP(data, k)
            iterations, j_values, centroids, min_location = kmeans.cluster()
            kmeanspp_itr.append(iterations)

            kmeans = ScalableKMeansPP(data, k, 2, r)
            iterations, j_values, centroids, min_location = kmeans.cluster()
            kmeansscalable_itr1.append(iterations)

            kmeans = ScalableKMeansPP(data, k, 4, r)
            iterations, j_values, centroids, min_location = kmeans.cluster()
            kmeansscalable_itr2.append(iterations)

            kmeans = ScalableKMeansPP(data, k, 8, r)
            iterations, j_values, centroids, min_location = kmeans.cluster()
            kmeansscalable_itr3.append(iterations)
        kmeans_mean.append(np.mean(kmeans_itr))
        kmeanspp_mean.append(np.mean(kmeanspp_itr))
        kmeansscalable_mean1.append(np.mean(kmeansscalable_itr1))
        kmeansscalable_mean2.append(np.mean(kmeansscalable_itr2))
        kmeansscalable_mean3.append(np.mean(kmeansscalable_itr3))

    # plot
    plt.figure(figsize=(10, 10))
    plt.plot(n_from_each_cluster * 4, kmeans_mean, '-o', lw=3, markersize=10)
    plt.plot(n_from_each_cluster * 4, kmeanspp_mean, '-o', lw=3, markersize=10)
    plt.plot(n_from_each_cluster * 4, kmeansscalable_mean1, '-o', lw=3, markersize=10)
    plt.plot(n_from_each_cluster * 4, kmeansscalable_mean2, '-o', lw=3, markersize=10)
    plt.plot(n_from_each_cluster * 4, kmeansscalable_mean3, '-o', lw=3, markersize=10)
    plt.legend(('KMeans', 'KMeans++', 'Scalable KMeans++ (l = 0.5k)', 'Scalable KMeans++ (l = 1k)', 'Scalable KMeans++ (l = 2k)'), prop={'size': 18}, loc=0)
    plt.xlabel('Sample Size', fontsize=18)
    plt.ylabel('Number of iterations', fontsize=18)
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=16)
    #ax.xaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.yaxis.set_tick_params(labelsize=16)
    plt.savefig('itr-samples.png')
    plt.close()

def accuracy_1():
    d = pandas.read_csv('../data/kddcup.data_10_percent_corrected')
    d_clean = d[d.isnull().any(axis=1)==False]
    data_full = d_clean.iloc[:,:].values
    for col in [1,2,3,41]:
        unique_labels = np.unique(data_full[:,col])
        for label in range(len(unique_labels)):
            data_full[np.where(data_full[:,col]==unique_labels[label])[0], col] = label

    k = 23
    r_count = 20000
    data = data_full[:r_count,:41]

    kmeans_ppv = []
    kmeanspp_ppv = []
    kmeansppscalable_ppv = []
    kmeansppscalable1_ppv = []
    kmeansppscalable2_ppv = []
    for i in range (1):
        print ('iteration: ', i)
        kmeans = KMeansBase(data, k)
        kmeans_ppv.append(gather_ppv(kmeans, data_full[:r_count,41]))

        kmeans = KMeansPP(data, k)
        kmeanspp_ppv.append(gather_ppv(kmeans, data_full[:r_count,41]))

        kmeans = ScalableKMeansPP(data, k, 12, 3)
        kmeansppscalable_ppv.append(gather_ppv(kmeans, data_full[:r_count,41]))

        kmeans = ScalableKMeansPP(data, k, 23, 3)
        kmeansppscalable1_ppv.append(gather_ppv(kmeans, data_full[:r_count,41]))

        kmeans = ScalableKMeansPP(data, k, 46, 3)
        kmeansppscalable2_ppv.append(gather_ppv(kmeans, data_full[:r_count,41]))
    ppv = np.array((np.mean(kmeans_ppv), np.mean(kmeanspp_ppv),
                          np.mean(kmeansppscalable_ppv), np.mean(kmeansppscalable1_ppv), np.mean(kmeansppscalable2_ppv)))
    std = np.array((np.std(kmeans_ppv), np.std(kmeanspp_ppv),
                          np.std(kmeansppscalable_ppv), np.std(kmeansppscalable1_ppv), np.std(kmeansppscalable2_ppv)))
    ind = np.arange(len(ppv))
    width = 0.35
    plt.bar(ind, ppv, width=width, yerr=std, color='y')
    plt.ylabel('Mean PPV')
    plt.xlabel('Algorithm')
    plt.xticks(ind+width/2., ('KMeans', 'KMeans++', 'Sc. KMeans++\n(l=0.5)', 'Sc. KMeans++\n(l=1)', 'Sc. KMeans++\n(l=2)'))
    plt.savefig('ppv-exp.png')
    plt.close()

def accuracy_spam():
    d = pandas.read_csv('../data/spambase.data')
    d_clean = d[d.isnull().any(axis=1)==False]
    data_full = d_clean.iloc[:,:].values
    k = 2
    data = data_full[:,:57]

    kmeans_ppv = []
    kmeanspp_ppv = []
    kmeansppscalable_ppv = []
    kmeansppscalable1_ppv = []
    kmeansppscalable2_ppv = []
    for i in range (100):
        print ('iteration: ', i)
        kmeans = KMeansBase(data, k)
        kmeans_ppv.append(gather_ppv(kmeans, data_full[:,57]))

        kmeans = KMeansPP(data, k)
        kmeanspp_ppv.append(gather_ppv(kmeans, data_full[:,57]))

        kmeans = ScalableKMeansPP(data, k, 1, 3)
        kmeansppscalable_ppv.append(gather_ppv(kmeans, data_full[:,57]))

        kmeans = ScalableKMeansPP(data, k, 2, 3)
        kmeansppscalable1_ppv.append(gather_ppv(kmeans, data_full[:,57]))

        kmeans = ScalableKMeansPP(data, k, 4, 3)
        kmeansppscalable2_ppv.append(gather_ppv(kmeans, data_full[:,57]))
    ppv = np.array((np.mean(kmeans_ppv), np.mean(kmeanspp_ppv),
                          np.mean(kmeansppscalable_ppv), np.mean(kmeansppscalable1_ppv), np.mean(kmeansppscalable2_ppv)))
    std = np.array((np.std(kmeans_ppv), np.std(kmeanspp_ppv),
                          np.std(kmeansppscalable_ppv), np.std(kmeansppscalable1_ppv), np.std(kmeansppscalable2_ppv)))
    ind = np.arange(len(ppv))
    width = 0.35
    plt.bar(ind, ppv, width=width, yerr=std, color='y')
    plt.ylabel('Mean PPV')
    plt.xticks(ind+width/2., ('KMeans', 'KMeans++', 'Sc. KMeans++\n(l=0.5)', 'Sc. KMeans++\n(l=1)', 'Sc. KMeans++\n(l=2)'))
    plt.savefig('ppv-exp.png')
    plt.close()

def calc_ppv(cluster_assignment, initial_cluster_assignment):
    cluster_index = []
    for i in np.unique(initial_cluster_assignment):
        cluster_index.append(np.where(initial_cluster_assignment == i))
    assigned_cluster_index = []
    for i in np.unique(cluster_assignment):
        assigned_cluster_index.append(np.where(cluster_assignment == i))
    correspondance = []
    for index in cluster_index:
        overlap = []
        for assigned_i in assigned_cluster_index:
            overlap.append(np.intersect1d(index, assigned_i).shape[0])
        correspondance.append(np.argmax(overlap))

    # now calculate the PPV
    # get the true positives
    ttp = 0
    tfp = 0
    for i in range(len(cluster_index)):
        tp = np.intersect1d(cluster_index[i], assigned_cluster_index[correspondance[i]]).shape[0]
        fp = len(cluster_index[i][0]) - tp
        print ('**********************', tp, fp)
        ttp += tp
        tfp += fp
    return ttp/float(ttp + tfp)

def gather_ppv(kmeans, initial_cluster_assignment):
    iterations, j_values, centroids, min_location = kmeans.cluster()
    cluster_assignment = np.argmax(min_location, axis=1)
    return calc_ppv(cluster_assignment, initial_cluster_assignment)

def accuracy_synthetic():
    k = 4
    kmeans_ppv = []
    kmeanspp_ppv = []
    kmeansppscalable_ppv = []
    kmeansppscalable1_ppv = []
    kmeansppscalable2_ppv = []
    for i in range (20):
        data = no_of_iterations(100000)
        kmeans = KMeansBase(data[:,:3], k)
        kmeans_ppv.append(gather_ppv(kmeans, data[:,3]))

        kmeans = KMeansPP(data[:,:3], k)
        kmeanspp_ppv.append(gather_ppv(kmeans, data[:,3]))

        kmeans = ScalableKMeansPP(data[:,:3], k, 2, 3)
        kmeansppscalable_ppv.append(gather_ppv(kmeans, data[:,3]))

        kmeans = ScalableKMeansPP(data[:,:3], k, 4, 3)
        kmeansppscalable1_ppv.append(gather_ppv(kmeans, data[:,3]))

        kmeans = ScalableKMeansPP(data[:,:3], k, 8, 3)
        kmeansppscalable2_ppv.append(gather_ppv(kmeans, data[:,3]))

    ppv = np.array((np.mean(kmeans_ppv), np.mean(kmeanspp_ppv),
                          np.mean(kmeansppscalable_ppv), np.mean(kmeansppscalable1_ppv), np.mean(kmeansppscalable2_ppv)))
    std = np.array((np.std(kmeans_ppv), np.std(kmeanspp_ppv),
                          np.std(kmeansppscalable_ppv), np.std(kmeansppscalable1_ppv), np.std(kmeansppscalable2_ppv)))
    ind = np.arange(len(ppv))
    width = 0.35
    plt.bar(ind, ppv, width=width, yerr=std, color='y')
    plt.ylabel('Mean PPV')
    plt.xticks(ind+width/2., ('KMeans', 'KMeans++', 'Sc. KMeans++\n(l=0.5)', 'Sc. KMeans++\n(l=1)', 'Sc. KMeans++\n(l=2)'))
    plt.savefig('ppv.png')
    plt.close()


if __name__ == '__main__':
    #print(np.where(data[:,3] == (np.argmax(min_location, axis=1) + 1))[0].shape[0])
    #samplesize_initcost()
    #clusters_initcost()
    #itr_vs_sampels()
    #accuracy_synthetic()
    accuracy_spam()