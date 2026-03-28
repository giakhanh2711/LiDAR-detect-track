import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np
import random
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

import lidar.helper as helper
from lidar.kalman_filter import KalmanFilter



class Track:
    def __init__(self, track_id, x, y, now):
        self.id = track_id

        # Maintain estimate next step for each track
        self.mu = np.array([[x], [y], [helper.HUMAN_SPEED], [helper.HUMAN_SPEED]], dtype=float)
        self.sigma = np.eye(4) * 0.1                                

        self.path = []
        self.last_seen_time = now
        self.in_scene = True

        self.r = random.uniform(0.0, 0.19)
        self.g = random.uniform(0.0, 1.0)
        self.b = random.uniform(0.0, 1.0)


class PersonTrackerNode(Node):
    def __init__(self):
        super().__init__("track")

        self.tracks = {}
        self.next_id = 0

        self.match_distance = 1.5
        self.max_invisible_time = 3.0

        self.kf = KalmanFilter(
            A=helper.A,
            B=helper.B,
            C=helper.C,
            G=helper.G,
            H=helper.H,
            sigma_theta=helper.SIGMA_THETA,
            sigma_psi=helper.SIGMA_PSI
        )

        self.subscription_next_centroids = self.create_subscription(
            msg_type=PoseArray,
            topic="/person_detections",
            callback=self.get_next_centroids,
            qos_profile=10
        )

        self.markers_publisher = self.create_publisher(
            msg_type=MarkerArray,
            topic="/person_markers",
            qos_profile=10
        )

        self.marker_timer = self.create_timer(
            helper.PUBLISH_MARKER_TIME_PERIOD,
            self.publish_markers
        )

        self.last_header = None

    
    def predict_all_tracks(self):
        if len(self.tracks) == 0:
            return np.zeros((0, 2)), []

        track_list = list(self.tracks.values())
        predicted_positions = []

        for track in track_list:
            # Estimate the next state from the current state for the cost matrix.
            # No action
            # mu_next = A @ mu_current
            mu_next_estimate = self.kf.A @ track.mu
            
            predicted_positions.append([
                float(mu_next_estimate[0, 0]),
                float(mu_next_estimate[1, 0])
            ])

        return np.array(predicted_positions), track_list

    
    def compute_cost_matrix(self, centroids, predicted_positions):
        if len(centroids) == 0 or len(predicted_positions) == 0:
            return np.zeros((0,0))
        
        cost = cdist(centroids, predicted_positions, metric='euclidean')

        # track_list = list(self.tracks.values())
    
        # for t_idx, track in enumerate(track_list):
        #     # Give established tracks a wider search area (2.0m) to reconnect segments
        #     # but keep new tracks/ghosts strict (0.6m)
        #     is_established = len(track.path) > 10
        #     limit = 2.0 if (track.in_scene and is_established) else 0.7
            
        #     cost[cost[:, t_idx] > limit, t_idx] = 1e4
        cost[cost > self.match_distance] = 1e4
        return cost


    def assign_tracks(self, cost_matrix):
        if cost_matrix.size == 0:
            return [], set(), set()

        # Do Hungarian to match new centroids and current tracks
        centroid_indices, track_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = set(range(cost_matrix.shape[0]))
        unmatched_tracks = set(range(cost_matrix.shape[1]))

        for c_i, t_i in zip(centroid_indices, track_indices):
            if cost_matrix[c_i, t_i] < 1e4:
                matched.append([c_i, t_i])
                unmatched_dets.discard(c_i)
                unmatched_tracks.discard(t_i)

        return matched, unmatched_dets, unmatched_tracks


    def update_matched_tracks(self, matched, centroids, track_list, now):
        for centroid_idx, track_idx in matched:
            track = track_list[track_idx]
            x, y = centroids[centroid_idx]

            observation = np.array([[[x], [y]]], dtype=float) 
            mu = track.mu.reshape(1, 4, 1)
            sigma = track.sigma.reshape(1, 4, 4)

            # KF to predict next step
            mu_new, sigma_new = self.kf.next_step_predict(
                cluster_centroids=mu,
                sigmas=sigma,
                observations=observation
            )

            # Update estimate next step for each track
            track.mu = mu_new[0]
            track.sigma = sigma_new[0]

            # Add the estimate next step from KF to the path
            point = Point()
            point.x = float(track.mu[0,0])
            point.y = float(track.mu[1,0])
            point.z = 0.0
            track.path.append(point)
            
            track.last_seen_time = now
            track.in_scene = True


    def handle_unmatched_tracks(self, unmatched_tracks, track_list, now):
        for track_idx in unmatched_tracks:
            track = track_list[track_idx]

            # If track doesn't have centroids matched with it, its estimate next step is the intermediate covariance and 
            # updated mu without observation
            track.mu = self.kf.A @ track.mu
            track.sigma = self.kf.A @ track.sigma @ self.kf.A.T + self.kf.G @ self.kf.sigma_theta @ self.kf.G.T

            # Check if we can add this estimate step to its path?
            invisible_duration = (now - track.last_seen_time).nanoseconds / 1e9
            if invisible_duration < 0.5:
                point = Point()
                point.x = float(track.mu[0,0])
                point.y = float(track.mu[1,0])
                point.z = 0.0
                track.path.append(point)

            # Temporarily change the color if it's occluded
            track.in_scene = False


    def create_new_tracks(self, unmatched_centroids, centroids, now):
        """
        Create new tracks for unmatched centroids
        """
        for i in unmatched_centroids:
            x, y = centroids[i]
            new_track = Track(self.next_id, x, y, now)
            self.next_id += 1

            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.0
            new_track.path.append(point)

            self.tracks[new_track.id] = new_track


    def delete_tracks(self, now):
        """
        This function helps delete not-in-scene tracks, so help us handle easier
        """
        ids_to_del = []
        for track_id, track in self.tracks.items():
            duration = (now - track.last_seen_time).nanoseconds / 1e9
            
            # If this is a new track (its path has less than 5 points), delete it after 0.5s
            if len(track.path) < 5:
                if duration > 0.5:
                    ids_to_del.append(track_id)

            # If the track's path is long enough, only delete it if the person is considered to leave the scene (after cannot see for 2 seconds)
            elif duration > self.max_invisible_time:
                ids_to_del.append(track_id)
        
        for track_id in ids_to_del:
            del self.tracks[track_id]


    def get_next_centroids(self, msg):
        if not msg.poses:
            return
        
        self.last_header = msg.header
        now = self.get_clock().now()

        centroids = np.array([[p.position.x, p.position.y] for p in msg.poses])

        # First scan: only create new tracks
        if len(self.tracks) == 0:
            self.create_new_tracks(range(len(centroids)), centroids, now)
            return
            
        # Predict next step all tracks based on their current positition and transition equation
        predicted_positions, track_list = self.predict_all_tracks()

        # Compute the cost matrix between the estimate next steps and observed centroids
        cost = self.compute_cost_matrix(centroids, predicted_positions)

        # Hungarian to match centroids to tracks based on the cost between their estimate next step and observed centroids
        matched, unmatched_centroids, unmatched_tracks = self.assign_tracks(cost)

        # If tracks are matched with observed centroids, use KF to update the estimate next step and add these to the track's path
        self.update_matched_tracks(matched, centroids, track_list, now)

        # If track doesn't have centroids matched with it, its estimate next step is the intermediate covariance and 
        # updated mu without observation
        self.handle_unmatched_tracks(unmatched_tracks, track_list, now)

        # Create new tracks for unmatched centroids
        self.create_new_tracks(unmatched_centroids, centroids, now)

        self.delete_tracks(now)


    def publish_markers(self):
        if self.last_header is None:
            return

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for track in self.tracks.values():
            if len(track.path) < 2:
                continue

            marker = Marker()
            marker.header = self.last_header
            marker.header.stamp = now
            marker.ns = "people_tracks"
            marker.id = track.id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05

            if not track.in_scene:
                track.r = random.uniform(0.81, 1.0)

            marker.color.r = track.r
            marker.color.g = track.g
            marker.color.b = track.b
            marker.color.a = 1.0

            marker.points = track.path
            marker_array.markers.append(marker)

        self.markers_publisher.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = PersonTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()