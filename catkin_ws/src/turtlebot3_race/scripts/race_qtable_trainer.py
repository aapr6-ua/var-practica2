#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import random

class QLearningAgent:
    def __init__(self):
        rospy.init_node('race_qtable_trainer')

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.scan_data = None
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.rate = rospy.Rate(5)

        # Q-table: 27 estados x 5 acciones
        self.q_table = np.zeros((27, 5))
        self.epsilon = 0.2
        self.alpha = 0.5
        self.gamma = 0.9

        self.actions = {
            0: ('forward', 0.2, 0.0),
            1: ('left', 0.1, 1.0),
            2: ('right', 0.1, -1.0),
            3: ('stop', 0.0, 0.0),
            4: ('reverse', -0.1, 0.0)
        }

        # Para detectar bucles
        self.prev_state = None
        self.stuck_counter = 0

    def lidar_callback(self, data):
        self.scan_data = data

    def odom_callback(self, data):
        self.linear_velocity = data.twist.twist.linear.x
        self.angular_velocity = data.twist.twist.angular.z

    def get_sector_distances(self):
        if self.scan_data is None:
            return 10, 10, 10

        scan = np.array(self.scan_data.ranges)
        scan = np.clip(scan, 0, 10)
        scan[scan < 0.05] = 10

        front = min(list(scan[330:]) + list(scan[:31]))
        left = min(scan[60:121])
        right = min(scan[240:301])
        return front, left, right

    def get_state(self):
        front, left, right = self.get_sector_distances()

        def discretize(dist):
            if dist < 0.4:
                return 2
            elif dist < 1.0:
                return 1
            else:
                return 0

        d_left = discretize(left)
        d_front = discretize(front)
        d_right = discretize(right)

        state = d_left * 9 + d_front * 3 + d_right
        return state

    def choose_action(self, state):
        front, left, right = self.get_sector_distances()

        if front > 0.5:
            return 0  # avanzar
        elif front < 0.3:
            return random.choice([1, 2, 4])  # izquierda, derecha, o marcha atrás

        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 4)
        return np.argmax(self.q_table[state])

    def perform_action(self, action_index):
        action = self.actions[action_index]
        twist = Twist()
        twist.linear.x = action[1]
        twist.angular.z = action[2]
        self.cmd_pub.publish(twist)

    def calculate_reward(self):
        reward = 0
        front, _, _ = self.get_sector_distances()

        if front < 0.3:
            reward -= 100
        elif front < 0.6:
            reward -= 30

        if self.linear_velocity > 0.05:
            reward += 10
        elif abs(self.angular_velocity) > 0.2:
            reward -= 5
        else:
            reward -= 10

        return reward

    def run(self):
        while not rospy.is_shutdown():
            state = self.get_state()
            action = self.choose_action(state)
            self.perform_action(action)
            rospy.sleep(0.2)

            reward = self.calculate_reward()
            next_state = self.get_state()

            # Detectar bucle de castigo
            if state == self.prev_state and reward < -50:
                self.stuck_counter += 1
                if self.stuck_counter >= 5:
                    rospy.logwarn(f"[BLOQUEO] Estado {state} con penalización continua. Considera reiniciar o entrenar más.")
            else:
                self.stuck_counter = 0

            self.prev_state = state

            # Q-learning update
            old_q = self.q_table[state, action]
            next_max = np.max(self.q_table[next_state])
            self.q_table[state, action] = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * next_max)

            rospy.loginfo(f"[S:{state}] A:{self.actions[action][0]} → R:{reward} → S':{next_state}")
            self.rate.sleep()

if __name__ == '__main__':
    try:
        agent = QLearningAgent()
        agent.run()
    except rospy.ROSInterruptException:
        pass
