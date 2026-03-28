import rclpy
from rclpy.node import Node
import visualization_msgs.msg
import geometry_msgs.msg

import numpy as np
import scipy

import lidar.helper as helper

class Test(Node):
    def __init__(self):
        super().__init__("test")
        
        self.markers_publisher = self.create_publisher(
            msg_type=visualization_msgs.msg.MarkerArray,
            topic="/person_markers",
            qos_profile=helper.QOS_PROFILE,
        )

        self.markers_timer = self.create_timer(
            timer_period_sec=helper.PUBLISH_MARKER_TIME_PERIOD,
            callback=self.publish_marker
        )


        self.subscription_next_centroids = self.create_subscription(
            msg_type=geometry_msgs.msg.PoseArray,
            topic="/centroids_pose",
            qos_profile=helper.QOS_PROFILE,
            callback=self.get_next_pose_centroids
        )

        self.cluster_centroids = None


    def get_next_pose_centroids(self, msg):
        """
        return [N, 2, 1] [[[x], [y]], [[x], [y]],...]
        """
        observed_centroids = []
        for state in msg.poses:
            centroid = [state.x, state.y]
            observed_centroids.append(centroid)
        
        self.cluster_centroids = np.array(observed_centroids).reshape(-1, 2, 1)
    

    def publish_marker(self):
        if self.cluster_centroids is None:
            return
        
        marker_array = visualization_msgs.msg.MarkerArray()
        
        # Clear previous markers if necessary (optional depending on your RViz setup)
        # We use the index 'i' as a unique ID for each marker
        for i, centroid in enumerate(self.cluster_centroids):
            marker = visualization_msgs.msg.Marker()
            marker.header.frame_id = "laser" # Or "laser_frame" / "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "human_clusters"
            marker.id = i
            marker.type = visualization_msgs.msg.Marker.SPHERE
            marker.action = visualization_msgs.msg.Marker.ADD
            
            # Set the position from the centroid pose
            marker.pose.position.x = centroid.position.x
            marker.pose.position.y = centroid.position.y
            marker.pose.position.z = 0.5  # Slightly elevated so it's visible
            
            # Set the scale (size) of the sphere
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            
            # Set color (Green for detected human clusters)
            marker.color.a = 1.0 # Alpha must be non-zero!
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
                        
            marker_array.markers.append(marker)
        
        print(len(marker_array))
        self.marker_publisher.publish(marker_array)
        


    def hungarian_match(self, centroids, estimate):
        num_mu = estimate.shape[0]
        num_measurement = centroids.shape[0]

        cost_matrix = np.zeros((num_mu, num_measurement))

        for i, mu in enumerate(estimate):
            estimate_pose = mu[:2]
            for j, centroid in enumerate(centroids):
                cost_matrix[i, j] = np.linalg.norm(estimate_pose - centroid)
        
        cost_matrix[cost_matrix > helper.HUMAN_SPEED] = 1e3


        # TODO: Update track when identify who is person, which is obstacles motionless
        track_idx, measurement_idx = scipy.optimize.linear_sum_assignment(cost_matrix)

        track_idx_corrected, measurement_idx_corrected = [], []

        for idx_track, idx_measurement in zip(track_idx, measurement_idx):
            if cost_matrix[idx_track, idx_measurement] <= helper.HUMAN_SPEED:
                track_idx_corrected.append(idx_track)
                measurement_idx_corrected.append(idx_measurement)
        
        return np.array(track_idx_corrected), np.array(measurement_idx_corrected)


def main(args=None):
    rclpy.init(args=args)
    tracker = Test()
    rclpy.spin(tracker)
    tracker.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()