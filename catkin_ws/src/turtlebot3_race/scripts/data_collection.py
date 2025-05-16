#!/usr/bin/env python3
import rospy
import csv
import os
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np

class DataCollector:
    def __init__(self):
        rospy.init_node('data_collector')

        # Directorio de salida y archivo
        self.output_dir = rospy.get_param('~output_dir', 'data')
        os.makedirs(self.output_dir, exist_ok=True)
        self.file_path = os.path.join(self.output_dir, 'dataset.csv')
        file_exists = os.path.isfile(self.file_path)
        is_empty = not file_exists or os.path.getsize(self.file_path) == 0

        self.csv_file = open(self.file_path, 'a', newline='')
        self.writer = csv.writer(self.csv_file)

        # Escribir cabecera si el archivo está vacío
        if is_empty:
            header = [f'r{i}' for i in range(360)] + ['lin_vel', 'ang_vel', 'cmd_lin', 'cmd_ang']
            self.writer.writerow(header)

        # Almacenamiento del estado
        self.scan = None
        self.odom_lin = 0.0
        self.odom_ang = 0.0
        self.cmd_lin = 0.0
        self.cmd_ang = 0.0

        # Suscripciones
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        rospy.Subscriber('/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_cb)

        self.rate = rospy.Rate(15)
        self.run()

    def scan_cb(self, msg):
        self.scan = np.array(msg.ranges)
        self.scan[self.scan == 0] = 10.0  # Reemplazo de ceros por valor máximo
        self.scan = np.clip(self.scan, 0, 10.0)

    def odom_cb(self, msg):
        self.odom_lin = msg.twist.twist.linear.x
        self.odom_ang = msg.twist.twist.angular.z

    def cmd_cb(self, msg):
        self.cmd_lin = msg.linear.x
        self.cmd_ang = msg.angular.z

    def run(self):
        rospy.loginfo("Iniciando recolección de datos...")
        while not rospy.is_shutdown():
            if self.scan is not None:
                scan_processed = np.nan_to_num(self.scan, nan=10.0, posinf=10.0, neginf=0.0)
                scan_processed += np.random.normal(0, 0.02, scan_processed.shape)
                scan_processed = np.clip(scan_processed, 0, 10.0)

                row = list(scan_processed) + [self.odom_lin, self.odom_ang, self.cmd_lin, self.cmd_ang]
                self.writer.writerow(row)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        DataCollector()
    except rospy.ROSInterruptException:
        pass
