#!/usr/bin/env python3
import rospy
import csv
import os
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np

class DataCollector:
    def __init__(self):
        rospy.init_node('data_collector')

        # Prepare output directory and file
        self.output_dir = rospy.get_param('~output_dir', 'data')
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_file = open(os.path.join(self.output_dir, 'dataset.csv'), 'w')
        self.writer = csv.writer(self.csv_file)
        # Header: ranges (360), linear_vel, angular_vel, cmd_lin, cmd_ang
        header = [f'r{i}' for i in range(360)] + ['lin_vel', 'ang_vel', 'cmd_lin', 'cmd_ang']
        self.writer.writerow(header)

        # State storage
        self.scan = None
        self.odom_lin = 0.0
        self.odom_ang = 0.0
        self.cmd_lin = 0.0
        self.cmd_ang = 0.0

        # Subscribers
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        rospy.Subscriber('/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_cb)

        # Main loop
        self.rate = rospy.Rate(10)
        self.run()

    def scan_cb(self, msg):
        self.scan = np.array(msg.ranges)

    def odom_cb(self, msg):
        self.odom_lin = msg.twist.twist.linear.x
        self.odom_ang = msg.twist.twist.angular.z

    def cmd_cb(self, msg):
        self.cmd_lin = msg.linear.x
        self.cmd_ang = msg.angular.z

    def run(self):
        rospy.loginfo("Starting data collection...")
        while not rospy.is_shutdown():
            if self.scan is not None:
                row = list(np.nan_to_num(self.scan, nan=10.0, posinf=10.0, neginf=0.0))
                row += [self.odom_lin, self.odom_ang, self.cmd_lin, self.cmd_ang]
                self.writer.writerow(row)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        DataCollector()
    except rospy.ROSInterruptException:
        pass