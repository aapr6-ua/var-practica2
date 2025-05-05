#!/usr/bin/env python3
import rospy
import torch
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from train_cnn import CNNController

class CNNExecutor:
    def __init__(self):
        rospy.init_node('cnn_executor')
        self.model = CNNController()
        model_path = rospy.get_param('~model_path', 'model.pt')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        self.scan = None
        self.rate = rospy.Rate(10)

    def scan_cb(self, msg):
        arr = np.array(msg.ranges)
        arr = np.nan_to_num(arr, nan=10.0, posinf=10.0, neginf=0.0)
        self.scan = torch.tensor(arr, dtype=torch.float32).view(1,1,-1)

    def run(self):
        while not rospy.is_shutdown():
            if self.scan is not None:
                with torch.no_grad():
                    cmd = self.model(self.scan).squeeze().numpy()
                twist = Twist()
                twist.linear.x = float(cmd[0])
                twist.angular.z = float(cmd[1])
                self.pub.publish(twist)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        CNNExecutor().run()
    except rospy.ROSInterruptException:
        pass