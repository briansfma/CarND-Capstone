#!/usr/bin/env python

import numpy as np
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

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
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None

        self.stopline_wp_idx = -1 # Default state of "I don't see any traffic lights"

        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.waypoint_tree:
                # Get closest waypoint to car's current position
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                # Generate updated waypoint list for car based on that nearest waypoint
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]

        # Check if closest is ahead or behind vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest coordinates
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self, closest_idx):
        # lane = Lane()
        # lane.header = self.base_waypoints.header
        # lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx+LOOKAHEAD_WPS]

        lane = self.generate_lane(closest_idx)

        self.final_waypoints_pub.publish(lane)

    def generate_lane(self, closest_idx):
        lane = Lane()
        lane.header = self.base_waypoints.header
        
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        new_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]

        # Case 1: Traffic light is red
        if (0 < self.stopline_wp_idx < farthest_idx):
            # Build new waypoints with speed adjusted for environment
            temp = []

            # Lower speed to stop at the line (model deceleration wanted)
            for i, wp in enumerate(new_waypoints):
                # Calc velocity curve to smoothly stop the car
                # Four points back so car stops at line
                stop_idx = max(self.stopline_wp_idx - closest_idx - 4, 0)

                dist = self.distance(new_waypoints, i, stop_idx)
                vel = 0.4 * dist                       # arbitrary value to work with MAX_DECEL
                if vel < 0.5:
                    vel = 0.

                temp.append(self.generate_waypoint(wp, min(vel, wp.twist.twist.linear.x)))

                lane.waypoints = temp
        # Case 2: Nothing in the way, raise speed to match waypoint command
        else:
            lane.waypoints = new_waypoints

        # rospy.logwarn("VEL_CMD: {}".format(temp[0].twist.twist.linear.x))

        return lane

    def generate_waypoint(self, waypoint, vel_cmd):
        p = Waypoint()
        p.pose = waypoint.pose
        p.twist.twist.linear.x = vel_cmd

        return p

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, msg):
        self.base_waypoints = msg
        if not self.waypoints_2d:
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in msg.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data
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
