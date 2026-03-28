import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose, PoseArray
import numpy as np
from sklearn.cluster import DBSCAN

import lidar.helper as helper


QOS_PROFILE = 10


class PersonDetectNode(Node):
    def __init__(self):
        super().__init__('detect')

        # Create a DBSCAN object to do clustering for humans 2D points
        self.dbscan = DBSCAN(eps=0.3, min_samples=5)

        # Subscriber to read lidar scan and process data
        self.subscription_scan = self.create_subscription(
            msg_type=LaserScan,
            topic='/scan',
            qos_profile=QOS_PROFILE,
            callback=self.read_scan,
        )

        # Publisher to publish detected human centroids
        self.centroids_publisher = self.create_publisher(
            msg_type=PoseArray,
            topic='/person_detections',
            qos_profile=QOS_PROFILE
        )

        self.is_first_scan = True
        self.ranges_static = None
        self.angles = None


    def clustering(self, xy_coordinates):
        """
        Receive a list of [[x1, y1], [x2, y2],...] of humans points from lidar.
        Return:
            [Pose, Pose,...]: list of human cluster centroids
        """

        if xy_coordinates is None or xy_coordinates.shape[0] == 0:
            return []

        self.dbscan.fit(xy_coordinates)
        labels = self.dbscan.labels_

        cluster_centroids = []
        labels_set = set(labels) - {-1}
        for label in labels_set:
            cluster_points = xy_coordinates[labels == label]
            x, y = np.mean(cluster_points, axis=0)

            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = 0.0
            cluster_centroids.append(pose)

        return cluster_centroids


    def read_scan(self, msg):

        ranges = np.array(msg.ranges, dtype=np.float32)

        # Replace infinite value with range max from lidar
        ranges[np.isinf(ranges)] = msg.range_max

        # For nan values, use linear intepolation from numpy to get their values
        nans = np.isnan(ranges)
        if np.any(nans):
            idx = np.arange(len(ranges))
            ranges[nans] = np.interp(x=idx[nans], xp=idx[~nans], fp=ranges[~nans])

        # If this is the first scan, store the lidar data as the data of static objects.
        # As humans tend to appear in later scans.
        if self.is_first_scan:
            self.ranges_static = ranges.copy()
            self.angles = np.linspace(
                msg.angle_min,
                msg.angle_min + msg.angle_increment * (len(ranges) - 1),
                len(ranges)
            )
            self.is_first_scan = False
            return


        # If range is smaller than ranges_static (the farthest) at the same angle,
        # so the farthest is the static object, and the closer range is human.
        # This can also return some points belong to static objects, but this can be solved after clustering step.
        mask = self.ranges_static - ranges > helper.THRESHOLD_DETECT_WALL
        ranges_human = ranges[mask]
        angles_human = self.angles[mask]

        if ranges_human.size == 0:
            pose_array = PoseArray()
            pose_array.header = msg.header
            self.centroids_publisher.publish(pose_array)
            return

        xy_coordinates = helper.ranges_to_xy(ranges_human, angles_human)
        cluster_centroids = self.clustering(xy_coordinates)

        pose_array = PoseArray()
        pose_array.header = msg.header
        pose_array.poses = cluster_centroids

        # Publish centroids right after reading scan
        self.centroids_publisher.publish(pose_array)


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()