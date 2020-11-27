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
        # initialize a publisher to send estimated joints' position to the robot
        self.end_eff_pos_est_pub = rospy.Publisher("/robot/end_effector_position_estimation", Float64MultiArray,
                                                   queue_size=10)
        self.end_eff_pos_pub = rospy.Publisher("/robot/end_effector_position", Float64MultiArray,
                                                   queue_size=10)
        self.red_pos_sub = message_filters.Subscriber("/robot/red_position_estimation/3d", Float64MultiArray)
        self.joint2_angle_est_sub = message_filters.Subscriber("/robot/joint2_position_controller/estimation", Float64)
        self.joint3_angle_est_sub = message_filters.Subscriber("/robot/joint3_position_controller/estimation", Float64)
        self.joint4_angle_est_sub = message_filters.Subscriber("/robot/joint4_position_controller/estimation", Float64)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.red_pos_sub, self.joint2_angle_est_sub, self.joint3_angle_est_sub, self.joint4_angle_est_sub], 10, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

    def detect_end_effector_fk(self, theta1, theta2, theta3, theta4):
        x = 3 * np.sin(theta1) * np.sin(theta2) * np.cos(theta3) * np.cos(theta4) \
            + 3.5 * np.sin(theta1) * np.sin(theta2) * np.cos(theta3) \
            + 3 * np.cos(theta1) * np.sin(theta3) * np.cos(theta4) \
            + 3.5 * np.cos(theta1) * np.sin(theta3) \
            + 3 * np.sin(theta1) * np.cos(theta2) * np.sin(theta4)

        y = -3 * np.cos(theta1) * np.sin(theta2) * np.cos(theta3) * np.cos(theta4) \
            - 3.5 * np.cos(theta1) * np.sin(theta2) * np.cos(theta3) \
            + 3 * np.sin(theta1) * np.sin(theta3) * np.cos(theta4) \
            + 3.5 * np.sin(theta1) * np.sin(theta3) \
            - 3 * np.cos(theta1) * np.cos(theta2) * np.sin(theta4)

        z = 3 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) \
            + 3.5 * np.cos(theta2) * np.cos(theta3) \
            - 3 * np.sin(theta2) * np.sin(theta4) + 2.5

        end_eff_pos = np.array([x, y, z])
        return end_eff_pos

    def callback(self, end_eff_pos, theta2_est, theta3_est, theta4_est):
        self.end_eff_pos_est = Float64MultiArray()
        self.end_eff_pos_est.data = self.detect_end_effector_fk(0, theta2_est.data, theta3_est.data, theta4_est.data)
        print(self.end_eff_pos_est.data, end_eff_pos.data)
        # Publish the results
        try:
            self.end_eff_pos_est_pub.publish(self.end_eff_pos_est)
            self.end_eff_pos_pub.publish(end_eff_pos)
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
