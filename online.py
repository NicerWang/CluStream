import os
import pickle

from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np
import threading
import time
from micro_cluster import MicroCluster


def distance_to_cluster(x, cluster):
    """
    calculate euclidean distance
    """
    return distance.euclidean(x, cluster.get_center())


class CluStream:
    def __init__(self, dimension, time_window=1000, timestamp=0, micro_cluster_cnt=100, clusters=None):
        """
        Parameters
        ----------
        dimension: dimension of data points
        time_window: number of data points which will never be removed
        timestamp: not real timestamp, like IDs of data points
        micro_cluster_cnt: number of micro cluster
        clusters: clusters in clustream (init without call fit)
        """
        if clusters is None:
            self.clusters = []
        else:
            self.clusters = clusters
        self.timestamp = timestamp
        self.time_window = time_window
        self.micro_cluster_cnt = micro_cluster_cnt
        # for unique id
        self.created_clusters = 0
        self.dimension = dimension

    def fit(self, x):
        """
        initialize

        Parameters
        ----------
        x: an array like (number of data point, dimension)
        -------
        """
        assert x.shape[1] == self.dimension
        kmeans = KMeans(n_clusters=self.micro_cluster_cnt, random_state=1)
        labels = kmeans.fit_predict(x, None)
        x = np.column_stack((labels, x))
        initial_clusters = [x[x[:, 0] == label][:, 1:] for label in set(labels) if label != -1]
        # initial data is not saved
        for cluster in initial_clusters:
            self._create_micro_cluster(cluster)

    def insert_new(self, new):
        """
        insert a new data point

        Parameters
        ----------
        new: an array like (dimension,)
        -------
        """
        self.timestamp += 1
        assert new.shape == (self.dimension,)
        closest_cluster = self._find_closest_cluster(new)
        can_insert = self._check_insertion(new, closest_cluster)
        if can_insert:
            closest_cluster.insert(new, self.timestamp)
        else:
            oldest = self._oldest_updated_cluster()
            if oldest is not None:
                self.clusters.remove(oldest)
            else:
                self._merge_closest_clusters()
            self._create_micro_cluster([new])

    def get_snapshot(self):
        """
        return current status
        """
        return self.clusters

    def _create_micro_cluster(self, cluster):
        linear_sum = np.zeros(self.dimension)
        squared_sum = np.zeros(self.dimension)
        new_micro_cluster = MicroCluster(identifier=self.created_clusters, points=0, linear_sum=linear_sum,
                                         squared_sum=squared_sum)
        self.created_clusters += 1
        for point in cluster:
            new_micro_cluster.insert(point, self.timestamp)
        self.clusters.append(new_micro_cluster)

    def _find_closest_cluster(self, x, exception=None):
        min_distance = float('inf')
        closest_cluster = None
        for cluster in self.clusters:
            if cluster == exception:
                continue
            distance_cluster = distance_to_cluster(x, cluster)
            if distance_cluster < min_distance:
                min_distance = distance_cluster
                closest_cluster = cluster
        return closest_cluster

    def _check_insertion(self, x, cluster):
        # cluster have only one point, use distance to the closest cluster as radius
        if cluster.get_weight() == 1:
            radius = float('inf')
            next_cluster = self._find_closest_cluster(x, cluster)
            dist = distance.euclidean(next_cluster.get_center(), cluster.get_center())
            radius = min(dist, radius)
        else:
            radius = cluster.get_radius()
        if distance_to_cluster(x, cluster) < radius:
            return True
        else:
            return False

    def _oldest_updated_cluster(self):
        threshold = self.timestamp - self.time_window
        min_relevance_stamp = float('inf')
        oldest_cluster = None
        for cluster in self.clusters:
            relevance_stamp = cluster.get_relevance_stamp()
            if (relevance_stamp < threshold) and (relevance_stamp < min_relevance_stamp):
                min_relevance_stamp = relevance_stamp
                oldest_cluster = cluster
        return oldest_cluster

    def _merge_closest_clusters(self):
        min_distance = float('inf')
        cluster_1 = None
        cluster_2 = None
        for i, cluster in enumerate(self.clusters):
            center = cluster.get_center()
            for another_idx in range(i + 1, len(self.clusters)):
                dist = distance.euclidean(center, self.clusters[another_idx].get_center())
                if dist < min_distance:
                    min_distance = dist
                    cluster_1 = cluster
                    cluster_2 = self.clusters[another_idx]
        MicroCluster.merge(cluster_1, cluster_2)
        self.clusters.remove(cluster_2)


class Snapshot:
    def __init__(self, path, links, snapshot):
        self.path = path
        self.links = links
        with open(path, 'wb') as pkl:
            pickle.dump(snapshot, pkl)

    def remove(self):
        self.links -= 1
        if self.links <= 0:
            os.remove(self.path)


class SnapshotManager:
    def __init__(self, clustream, alpha=2, l=2, path="./clustream", log=False):
        """
        Parameters
        ----------
        clustream: an instance of CluStream
        alpha: alpha in paper
        l: l in paper
        path: path to save snapshots
        log: set True to enable log in stdin
        """
        self.clustream = clustream
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)
        self.log = log
        # base snapshots save interval
        self.save_interval = alpha
        self.alpha = alpha
        # each order has (alpha ^ l + 1) snapshots at most
        self.l = l
        self.current = 0
        self.orders = []

        self.timer = threading.Timer(self.save_interval, self._save_snapshot)
        self.timer.start()

    def stop(self):
        try:
            self.timer.cancel()
        except:
            pass

    def _save_snapshot(self):
        self.current += self.save_interval
        orders = self._get_valid_orders()
        path = self.path + '/{}.clustream'.format(int(round(time.time())))
        snapshot = Snapshot(path, len(orders), self.clustream.get_snapshot())
        for order in orders:
            self._try_add(order, snapshot)
        if self.log:
            print("[Snapshot] For orders [{}] Saved to {}".format(",".join(map(str, orders)), path))

        # if running too slow, actual save_interval will > save_interval
        del self.timer
        self.timer = threading.Timer(self.save_interval, self._save_snapshot)
        self.timer.start()

    def _get_valid_orders(self):
        t = 0
        valid_orders = []
        while pow(self.alpha, t) <= self.current:
            if self.current % pow(self.alpha, t) == 0:
                valid_orders.append(t)
            t += 1
        return valid_orders

    def _try_add(self, order, snapshot):
        while len(self.orders) <= order:
            self.orders.append([])

        target_order = self.orders[order]
        target_order.append(snapshot)
        while len(target_order) > pow(self.alpha, self.l) + 1:
            oldest = target_order.pop(0)
            oldest.remove()
            if self.log:
                print("[Snapshot] Remove order [{}] (links left: {}) in {}".format(order, oldest.links, oldest.path))


if __name__ == '__main__':
    clustream = CluStream(5, time_window=1000, micro_cluster_cnt=10)
    # manager = SnapshotManager(clustream=clustream, log=True)
    inputs = np.random.random((200, 5))
    clustream.fit(inputs)
    for i in range(10000):
        new = np.random.random((5,))
        clustream.insert_new(new)
        # time.sleep(1.0)
    # manager.stop()
    print("OK")
