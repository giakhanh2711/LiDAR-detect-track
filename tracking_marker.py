import rclpy
from rclpy.node import Node

from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Vector3, Point
from sensor_msgs.msg import LaserScan
import std_msgs

import math


QOS_PROFILE = 10
TIMER_PERIOD_SEC = 0.1


class TrackingMarker(Node):
    def __init__(self):
        super().__init__("tracking_marker")

        self.markers_publisher = self.create_publisher(
            msg_type = MarkerArray,
            topic = "/person_marker",
            qos_profile = QOS_PROFILE
        )

        self.reset()

        self.subscription_scan = self.create_subscription(
            msg_type=LaserScan,
            topic="/scan",
            qos_profile=QOS_PROFILE,
            callback=self.read_scan
        )


        self.marker_timer = self.create_timer(
            timer_period_sec = TIMER_PERIOD_SEC,
            callback = self.draw_one_scan
        )

    def range_to_xy(self, range, theta):
        x = range * math.cos(theta)
        y = range * math.sin(theta)
        return x, y

    def read_scan(self, msg):
        self.past_locations = []

        angle_min = msg.angle_min
        for i, r in enumerate(msg.ranges):
            if math.isinf(r) or math.isnan(r):
                continue
            angle = angle_min + i * msg.angle_increment
            self.past_locations.append([self.range_to_xy(r, angle)])


    def draw_one_scan(self):

        markers = MarkerArray()
        path_color = std_msgs.msg.ColorRGBA(r=80/255, g=0.0, b=0.0, a=1.0)

        for i, location_chain in enumerate(self.past_locations):
            path_marker = Marker()
            path_marker.header.frame_id = "laser"
            path_marker.header.stamp = self.get_clock().now().to_msg()
            path_marker.ns = f"person"
            path_marker.scale.x = 0.05
            path_marker.scale.y = 0.05
            path_marker.id = i
            path_marker.type= Marker.POINTS

            path_marker.color.r = 1.0
            path_marker.color.g = 0.0
            path_marker.color.b = 0.0
            path_marker.color.a = 1.0

            for location in location_chain:
                path_marker.points.append(Point(x=location[0], y=location[1], z=0.0))
            markers.markers.append(path_marker)

        self.markers_publisher.publish(markers)


    def reset(self):

        self.past_locations = []
        markers = MarkerArray()
        
        marker = Marker()
        marker.header.frame_id = "laser"
        marker.action = marker.DELETEALL

        markers.markers.append(marker)

        self.markers_publisher.publish(markers)



def main():
    rclpy.init()
    marker = TrackingMarker()
    rclpy.spin(marker)

    marker.destroy_node()
    rclpy.shutdown()



if __name__ == "__main__":
    main()