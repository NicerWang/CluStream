import os
import pickle
import time
import random
import numpy as np

from sklearn.cluster import KMeans


def subtract(snapshot_left, snapshot_right):
    """
    subtract between snapshots
    """
    mapper = {}
    for cluster in snapshot_right:
        for _id in cluster.id_list:
            mapper[_id] = cluster
    for cluster in snapshot_left:
        id_list = cluster.id_list.copy()
        for _id in id_list:
            if _id in mapper and mapper[_id] is not None:
                cluster_right = mapper[_id]
                cluster.points -= cluster_right.points
                cluster.linear_sum -= cluster_right.linear_sum
                cluster.squared_sum -= cluster_right.squared_sum
                cluster.id_list.remove(_id)
                for _removed_id in cluster_right.id_list:
                    mapper[_removed_id] = None
    return list(filter(lambda _cluster: _cluster.points != 0, snapshot_left))


def select_center(clusters, n):
    """
    select centroids by probability
    """
    centroids = []
    used = set()
    weight_sum = 0
    for cluster in clusters:
        weight_sum += cluster.get_weight()
    for i in range(n):
        t = random.randint(0, weight_sum - 1)
        for cluster in clusters:
            t -= cluster.get_weight()
            if t < 0:
                used.add(cluster)
                centroids.append(cluster.get_center())
                break
    return centroids


def query(h, n_cluster, path="./clustream", cur=int(time.time())):
    """
    user query function
    Parameters
    ----------
    h time: interval (h in paper)
    n_cluster: num of clusters
    path: path to load snapshots
    cur: query timestamp (in seconds)

    Returns (clusters, types)
    -------

    """

    # Load snapshots
    snapshots = sorted(os.listdir(path), reverse=True)
    start_snapshot_file = None
    end_snapshot_file = None
    for snapshot in snapshots:
        if int(snapshot.split('.')[0]) <= cur:
            end_snapshot_file = snapshot
            break
    for snapshot in snapshots:
        if int(snapshot.split('.')[0]) <= cur - h:
            start_snapshot_file = snapshot
            break
    with open(path + "/" + start_snapshot_file, "rb") as start:
        start_snapshot = pickle.load(start)
    with open(path + "/" + end_snapshot_file, "rb") as end:
        end_snapshot = pickle.load(end)

    # Process snapshots
    result = subtract(end_snapshot, start_snapshot)
    centroids = select_center(result, n_cluster)
    x = [cluster.get_center() for cluster in result]
    kmeans = KMeans(n_clusters=n_cluster, init=np.array(centroids), n_init=1)
    return result, kmeans.fit_predict(x)
