#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import message_filters


class image_converter:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        # record the begining time
        self.time_trajectory = rospy.get_time()
        # initialize errors
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
        self.time_previous_step2 = np.array([rospy.get_time()], dtype='float64')
        # initialize error and derivative of error for trajectory tracking
        self.error = np.array([0.0, 0.0, 0.0], dtype='float64')
        self.error_d = np.array([0.0, 0.0, 0.0], dtype='float64')
        # initialize a publisher to send estimated joints' position to the robot
        self.joint2_close_pub = rospy.Publisher("/robot/joint2_position_controller/close", Float64, queue_size=10)
        self.joint3_close_pub = rospy.Publisher("/robot/joint3_position_controller/close", Float64, queue_size=10)
        self.joint4_close_pub = rospy.Publisher("/robot/joint4_position_controller/close", Float64, queue_size=10)

        self.end_eff_pos_est_sub = message_filters.Subscriber("/robot/end_effector_position_estimation", Float64MultiArray)
        self.end_eff_pos_sub = message_filters.Subscriber("/robot/end_effector_position", Float64MultiArray)
        self.joint2_angle_est_sub = message_filters.Subscriber("/robot/joint2_position_controller/estimation", Float64)
        self.joint3_angle_est_sub = message_filters.Subscriber("/robot/joint3_position_controller/estimation", Float64)
        self.joint4_angle_est_sub = message_filters.Subscriber("/robot/joint4_position_controller/estimation", Float64)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.end_eff_pos_est_sub, self.end_eff_pos_sub, self.joint2_angle_est_sub, self.joint3_angle_est_sub, self.joint4_angle_est_sub],
            10, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

    def calculate_jacobian(self, theta1, theta2, theta3, theta4):
        dx = np.array([3 * np.cos(theta1) * np.sin(theta2) * np.cos(theta3) * np.cos(theta4)
                       + 3.5 * np.cos(theta1) * np.sin(theta2) * np.cos(theta3)
                       - 3 * np.sin(theta1) * np.sin(theta3) * np.cos(theta4)
                       - 3.5 * np.sin(theta1) * np.sin(theta3)
                       + 3 * np.cos(theta1) * np.cos(theta2) * np.sin(theta4),

                       3 * np.sin(theta1) * np.cos(theta2) * np.cos(theta3) * np.cos(theta4)
                       + 3.5 * np.sin(theta1) * np.cos(theta2) * np.cos(theta3)
                       - 3 * np.sin(theta1) * np.sin(theta2) * np.sin(theta4),

                       -3 * np.sin(theta1) * np.sin(theta2) * np.sin(theta3) * np.cos(theta4)
                       - 3.5 * np.sin(theta1) * np.sin(theta2) * np.sin(theta3)
                       + 3 * np.cos(theta1) * np.cos(theta3) * np.cos(theta4)
                       + 3.5 * np.cos(theta1) * np.cos(theta3),

                       -3 * np.sin(theta1) * np.sin(theta2) * np.cos(theta3) * np.sin(theta4)
                       - 3 * np.cos(theta1) * np.sin(theta3) * np.sin(theta4)
                       + 3 * np.sin(theta1) * np.cos(theta2) * np.cos(theta4)])

        dy = np.array([3 * np.sin(theta1) * np.sin(theta2) * np.cos(theta3) * np.cos(theta4)
                       + 3.5 * np.sin(theta1) * np.sin(theta2) * np.cos(theta3)
                       + 3 * np.cos(theta1) * np.sin(theta3) * np.cos(theta4)
                       + 3.5 * np.cos(theta1) * np.sin(theta3)
                       + 3 * np.sin(theta1) * np.cos(theta2) * np.sin(theta4),

                       -3 * np.cos(theta1) * np.cos(theta2) * np.cos(theta3) * np.cos(theta4)
                       - 3.5 * np.cos(theta1) * np.cos(theta2) * np.cos(theta3)
                       + 3 * np.cos(theta1) * np.sin(theta2) * np.sin(theta4),

                       3 * np.cos(theta1) * np.sin(theta2) * np.sin(theta3) * np.cos(theta4)
                       + 3.5 * np.cos(theta1) * np.sin(theta2) * np.sin(theta3)
                       + 3 * np.sin(theta1) * np.cos(theta3) * np.cos(theta4)
                       + 3.5 * np.sin(theta1) * np.cos(theta3),

                       3 * np.cos(theta1) * np.sin(theta2) * np.cos(theta3) * np.sin(theta4)
                       - 3 * np.sin(theta1) * np.sin(theta3) * np.sin(theta4)
                       - 3 * np.cos(theta1) * np.cos(theta2) * np.cos(theta4)])

        dz = np.array([0,

                       -3 * np.sin(theta2) * np.cos(theta3) * np.cos(theta4)
                       - 3.5 * np.sin(theta2) * np.cos(theta3)
                       - 3 * np.cos(theta2) * np.sin(theta4),

                       -3 * np.cos(theta2) * np.sin(theta3) * np.cos(theta4)
                       - 3.5 * np.cos(theta2) * np.sin(theta3),

                       -3 * np.cos(theta2) * np.cos(theta3) * np.sin(theta4)
                       - 3 * np.sin(theta2) * np.cos(theta4)])

        jacobian = np.array([dx, dy, dz])

        return jacobian
    
    def control_closed(self, end_eff_pos_est, end_eff_pos, theta2_est, theta3_est, theta4_est):
        # P gain
        K_p = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        # D gain
        K_d = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        # estimate time step
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        self.time_previous_step = cur_time
        # robot end-effector position
        pos = end_eff_pos_est.data
        # desired trajectory
        pos_d = end_eff_pos.data
        # estimate derivative of error
        self.error_d = ((pos_d - pos) - self.error) / dt
        # estimate error
        self.error = pos_d - pos
        q = np.array(0, theta2_est.data, theta3_est.data, theta4_est.data)  # estimate initial value of joints'
        J_inv = np.linalg.pinv(self.calculate_jacobian(0, theta2_est.data, theta3_est.data, theta4_est.data))  # calculating the psudeo inverse of Jacobian
        dq_d = np.dot(J_inv, (np.dot(K_d, self.error_d.transpose()) + np.dot(K_p, self.error.transpose())))  # control input (angular velocity of joints)
        q_d = q + (dt * dq_d)  # control input (angular position of joints)
        return q_d

    def callback(self, end_eff_pos_est, end_eff_pos, theta2_est, theta3_est, theta4_est):
        q_d = self.control_closed(end_eff_pos_est, end_eff_pos, theta2_est, theta3_est, theta4_est)
        self.joint1_close = Float64()
        self.joint1_close.data = q_d[0]
        self.joint2_close = Float64()
        self.joint2_close.data = q_d[1]
        self.joint3_close = Float64()
        self.joint3_close.data = q_d[2]
        # Publish the results
        try:
            self.joint2_close_pub.publish(self.joint2_close)
            self.joint3_close_pub.publish(self.joint3_close)
            self.joint4_close_pub.publish(self.joint4_close)
        except CvBridgeError as e:
            print(e)

# call the class
def main(args):
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)