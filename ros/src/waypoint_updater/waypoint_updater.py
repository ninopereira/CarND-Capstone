#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose = None
        self.waypoints = []

        # Loop
        self.loop()
        rospy.spin()

    def loop(self):
        rate = rospy.Rate(4)  # 4 Hz
        while not rospy.is_shutdown():

            if self.waypoints and self.pose:
                idx = self.get_closest_waypoint_index()

                # Solve wrap around start/end of list
                waypoints_wrap = []
                waypoints_wrap.extend(self.waypoints)
                waypoints_wrap.extend(self.waypoints)
                final_waypoints = waypoints_wrap[idx:idx+LOOKAHEAD_WPS]

                # pos = self.pose.position
                # wp_pos = waypoints_wrap[idx].pose.pose.position

                # rospy.loginfo('Position (x,y): (%s, %s)', pos.x, pos.y)
                # rospy.loginfo('WP Position (x,y): (%s, %s)', wp_pos.x, wp_pos.y)

                # Publish final waypoints
                lane = Lane()
                lane.waypoints = final_waypoints
                self.final_waypoints_pub.publish(lane)

            rate.sleep()

    def get_closest_waypoint_index(self):
        distances = []
        for wp in self.waypoints:
            distance = self.get_linear_distance(wp.pose.pose.position.x, wp.pose.pose.position.y,
                                                self.pose.position.x, self.pose.position.y)
            distances.append(distance)

        index = distances.index(min(distances))
        # TODO: Algorithm to select waypoint in front
        index += 1
        return index

    def get_linear_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

    def pose_cb(self, msg):
        self.pose = msg.pose
        pos = msg.pose.position # Point
        ori = msg.pose.orientation # Quaternion
        rospy.loginfo('Position (x,y): (%s, %s)', pos.x, pos.y)
        rospy.loginfo('Orientation (x,y,z,w): (%s, %s, %s, %s)', ori.x, ori.y, ori.z, ori.w)
        # TODO: Implement
        pass

    def waypoints_cb(self, waypoints):
        # Same waypoints every time?
        self.waypoints = waypoints.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
