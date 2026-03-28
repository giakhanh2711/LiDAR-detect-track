import numpy as np


class KalmanFilter:
    def __init__(self, A, B, C, G, H, sigma_theta, sigma_psi):
        self.A = A
        self.B = B
        self.C = C
        self.G = G
        self.H = H
        self.sigma_theta = sigma_theta
        self.sigma_psi = sigma_psi

    
    def next_step_predict(self, cluster_centroids, sigmas, observations, u=None):
        """
        cluster_centroids = [[[x], [y], [v_x], [v_y]], [[x], [y], [v_x], [v_y]],...]
        sigmas = [n_clusters x 4 x 4]
        observations = [[[x],[y]], [[x],[y]],...]

        return:
            mu_next: n centroids estimate next position
            sigma_next: how uncertain for each dimension
        """
        state_dim = cluster_centroids.shape[1]
        n_clusters = cluster_centroids.shape[0]

        if u is None:
            u = np.zeros((n_clusters, state_dim, 1))

        sigma_inter_next = self.A @ sigmas @ self.A.T + self.G @ self.sigma_theta @ self.G.T
        l_next = sigma_inter_next @ self.C.T @ np.linalg.inv(
            self.C @ sigma_inter_next @ self.C.T + self.H @ self.sigma_psi @ self.H.T
        )

        # A @ x + B @ u now both (N,4,1)
        mu_pred = self.A @ cluster_centroids + self.B @ u
        innovation = observations - self.C @ mu_pred
        mu_next = mu_pred + l_next @ innovation

        sigma_next = (np.identity(state_dim) - l_next @ self.C) @ sigma_inter_next

        return mu_next, sigma_next

        # state_dim = cluster_centroids.shape[1]
        # n_clusters = cluster_centroids.shape[0]

        # if u is None:
        #     u = np.zeros((n_clusters, state_dim, state_dim))

        # sigma_inter_next = self.A @ sigmas @ self.A.T + self.G @ self.sigma_theta @ self.G.T
        # l_next = sigma_inter_next @ self.C.T @ np.linalg.inv(self.C @ sigma_inter_next @ self.C.T + self.H @ self.sigma_psi @ self.H)
        # mu_next = self.A @ cluster_centroids + self.B @ u + l_next @ (observations - self.C @ (self.A @ cluster_centroids + self.B @ u))
        # sigma_next = (np.identity(state_dim) - l_next @ self.C) @ sigma_inter_next


        # return mu_next, sigma_next
    




    # def get_next_centroids(self, msg: PoseArray):
    #     self.last_header = msg.header
    #     now = self.get_clock().now()

    #     centroids = []
    #     for pose in msg.poses:
    #         centroids.append([pose.position.x, pose.position.y])
    #     centroids = np.array(centroids, dtype=float)  # N x 2

    #     # build arrays of last points and corresponding track ids
    #     track_ids = []
    #     last_centroids = []
    #     for tid, track in self.tracks.items():
    #         if track.path:
    #             last_pt = track.path[-1]
    #             track_ids.append(tid)
    #             last_centroids.append([last_pt.x, last_pt.y])

    #     if len(centroids) > 0 and len(last_centroids) > 0:
    #         last_centroids = np.array(last_centroids, dtype=float)  # M x 2
    #         distance = cdist(centroids, last_centroids, metric='euclidean')  # N x M
    #         min_dists = np.min(distance, axis=1)
    #         best_tracks_idx = np.argmin(distance, axis=1)  # index into track_ids
    #         centroid_match_indices = np.where(min_dists < self.match_distance)[0]
    #     else:
    #         best_tracks_idx = np.array([], dtype=int)
    #         centroid_match_indices = np.array([], dtype=int)

    #     # assign detections
    #     for i, (x, y) in enumerate(centroids):
    #         if len(last_centroids) == 0 or i not in centroid_match_indices:
    #             # no good match → new track
    #             track = Track(self.next_id)
    #             self.next_id += 1
    #             self.tracks[track.id] = track
    #             best_track = track
    #         else:
    #             # match to existing track
    #             tid = track_ids[best_tracks_idx[i]]
    #             best_track = self.tracks[tid]

    #         pt = Point()
    #         pt.x = float(x)
    #         pt.y = float(y)
    #         pt.z = 0.0
    #         best_track.path.append(pt)
    #         best_track.last_seen_time = now
    #         best_track.in_scene = True

    #     # update in_scene based on time
    #     for track in self.tracks.values():
    #         if track.last_seen_time is None:
    #             continue
    #         dt = (now - track.last_seen_time).nanoseconds * 1e-9
    #         if dt > self.max_invisible_time:
    #             track.in_scene = False

    # def publish_markers(self):
    #     if self.last_header is None:
    #         return

    #     marker_array = MarkerArray()
    #     now = self.get_clock().now().to_msg()

    #     for track in self.tracks.values():
    #         if len(track.path) < 2:
    #             continue

    #         marker = Marker()
    #         marker.header = self.last_header
    #         marker.header.stamp = now
    #         marker.ns = 'people_tracks'
    #         marker.id = track.id
    #         marker.type = Marker.LINE_STRIP
    #         marker.action = Marker.ADD
    #         marker.scale.x = 0.05

    #         if not track.in_scene:
    #             track.r = random.uniform(0.81, 1.0)

    #         marker.color.r = track.r
    #         marker.color.g = track.g
    #         marker.color.b = track.b
    #         marker.color.a = 1.0

    #         marker.points = track.path
    #         marker_array.markers.append(marker)

    #     self.markers_publisher.publish(marker_array)