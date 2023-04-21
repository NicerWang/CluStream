import math as math
from scipy import special


class MicroCluster:
    """
    the MicroCluster data structure
    Parameters
    ----------
    :parameter points: the number of points in the cluster
    :parameter identifier: the identifier of the cluster
    :parameter linear_sum: the same as in paper
    :parameter squared_sum: the same as in paper
    :parameter linear_time_sum: the same as in paper
    :parameter squared_time_sum: the same as in paper
    :parameter m: the number of points considered to determine the relevance stamp of a cluster (the same as in paper)
    :parameter t: radius factor
    """

    def __init__(self, points=0, identifier=None, linear_sum=None, squared_sum=None, m=100, t=1.8):
        self.points = points
        self.id_list = [identifier]
        self.linear_sum = linear_sum
        self.squared_sum = squared_sum
        self.linear_time_sum = 0
        self.squared_time_sum = 0
        self.update_timestamp = 0
        self.m = m
        self.t = t

    def get_center(self):
        center = [self.linear_sum[i] / self.points for i in range(len(self.linear_sum))]
        return center

    def get_weight(self):
        return self.points

    def insert(self, new_point, current_timestamp):
        self.points += 1
        self.update_timestamp = current_timestamp
        for i in range(len(new_point)):
            self.linear_sum[i] += new_point[i]
            self.squared_sum[i] += math.pow(new_point[i], 2)
        self.linear_time_sum += current_timestamp
        self.squared_time_sum += math.pow(current_timestamp, 2)

    @classmethod
    def merge(cls, micro_cluster1, micro_cluster2):
        micro_cluster1.points += micro_cluster2.points
        micro_cluster1.linear_sum += micro_cluster2.linear_sum
        micro_cluster1.squared_sum += micro_cluster2.squared_sum
        micro_cluster1.linear_time_sum += micro_cluster2.linear_time_sum
        micro_cluster1.squared_time_sum += micro_cluster2.squared_time_sum
        micro_cluster1.id_list.extend(micro_cluster2.id_list)
        return micro_cluster1

    def get_relevance_stamp(self):
        if self.points < 2 * self.m:
            return self._get_average_stamp()
        x = self.m / (2 * self.points)
        y = self._get_average_stamp() + 2 * self._get_sigma_of_time() * special.erfinv(1 - x)
        return y

    def _get_average_stamp(self):
        return self.linear_time_sum / self.points

    def _get_sigma_of_time(self):
        return self._get_deviation([self.linear_time_sum], [self.squared_time_sum])

    def get_radius(self):
        if self.points == 1:
            # distance to the closest cluster
            return 0
        # unbiased estimator, so use standard deviation directly
        return self._get_deviation(self.linear_sum, self.squared_sum) * self.t

    def _get_deviation(self, linear, squared):
        variances = 0.0
        for i in range(len(linear)):
            mean_1 = linear[i] / self.points
            mean_2 = squared[i] / self.points
            # D(x) = E(x^2) - (E(x))^2
            variances += mean_2 - math.pow(mean_1, 2)
        # sum of sqrt(D(x))
        return math.sqrt(variances)
