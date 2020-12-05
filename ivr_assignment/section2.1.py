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
        self.red_posx_pub = rospy.Publisher("/robot/red_position_estimation/x", Float64, queue_size=10)
        self.red_posy_pub = rospy.Publisher("/robot/red_position_estimation/y", Float64, queue_size=10)
        self.red_posz_pub = rospy.Publisher("/robot/red_position_estimation/z", Float64, queue_size=10)
        self.red_pos_pub = rospy.Publisher("/robot/red_position_estimation/3d", Float64MultiArray, queue_size=10)
        self.joint2_angle_pub = rospy.Publisher("/robot/joint2_position_controller/estimation", Float64, queue_size=10)
        self.joint3_angle_pub = rospy.Publisher("/robot/joint3_position_controller/estimation", Float64, queue_size=10)
        self.joint4_angle_pub = rospy.Publisher("/robot/joint4_position_controller/estimation", Float64, queue_size=10)

        self.robot_blue_pos_est_sub1 = message_filters.Subscriber("/robot/blue_position_estimation/cam1",
                                                                  Float64MultiArray)
        self.robot_green_pos_est_sub1 = message_filters.Subscriber("/robot/green_position_estimation/cam1",
                                                                   Float64MultiArray)
        self.robot_red_pos_est_sub1 = message_filters.Subscriber("/robot/red_position_estimation/cam1",
                                                                 Float64MultiArray)
        self.robot_blue_pos_est_sub2 = message_filters.Subscriber("/robot/blue_position_estimation/cam2",
                                                                  Float64MultiArray)
        self.robot_green_pos_est_sub2 = message_filters.Subscriber("/robot/green_position_estimation/cam2",
                                                                   Float64MultiArray)
        self.robot_red_pos_est_sub2 = message_filters.Subscriber("/robot/red_position_estimation/cam2",
                                                                 Float64MultiArray)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.robot_blue_pos_est_sub1, self.robot_green_pos_est_sub1, self.robot_red_pos_est_sub1,
             self.robot_blue_pos_est_sub2, self.robot_green_pos_est_sub2, self.robot_red_pos_est_sub2], 10, 0.1,
            allow_headerless=True)
        self.ts.registerCallback(self.callback)

    def callback(self, blue_pos_cam1, green_pos_cam1, red_pos_cam1,
                 blue_pos_cam2, green_pos_cam2, red_pos_cam2):
        self.blue_pos = np.array([blue_pos_cam2.data[0], blue_pos_cam1.data[0],
                                  np.mean([blue_pos_cam2.data[1], blue_pos_cam1.data[1]])])
        self.green_pos = np.array([green_pos_cam2.data[0], green_pos_cam1.data[0],
                                   np.mean([green_pos_cam2.data[1], green_pos_cam1.data[1]])])
        self.red_pos = np.array([red_pos_cam2.data[0], red_pos_cam1.data[0],
                                 np.mean([red_pos_cam2.data[1], red_pos_cam1.data[1]])])
        self.red_pos_x = Float64()
        self.red_pos_x.data = self.red_pos[0]
        self.red_pos_y = Float64()
        self.red_pos_y.data = self.red_pos[1]
        self.red_pos_z = Float64()
        self.red_pos_z.data = self.red_pos[2]
        self.red_pos_f = Float64MultiArray()
        self.red_pos_f.data = self.red_pos

        # end_of_semicircle_pos_j2 = self.blue_pos + np.array([0, 3.5, 0])
        # dist = np.sqrt(np.sum((end_of_semicircle_pos_j2 - self.green_pos) ** 2))
        # cos_joint2 = (2 * 3.5 ** 2 - dist ** 2) / (2 * 3.5 * 3.5)
        # if cos_joint2 > 1:  cos_joint2 = 1
        # if cos_joint2 < -1: cos_joint2 = -1
        # self.joint2_angle = Float64()
        # self.joint2_angle.data = np.arccos(cos_joint2) - np.pi / 2

        end_of_semicircle_pos_j3 = self.blue_pos + np.array([-3.5, 0, 0])
        dist = np.sqrt(np.sum((end_of_semicircle_pos_j3 - self.green_pos) ** 2))
        cos_joint3 = (2 * 3.5 ** 2 - dist ** 2) / (2 * 3.5 * 3.5)
        if cos_joint3 < 0: cos_joint3 *= 0.8
        if cos_joint3 > 1:  cos_joint3 = 1
        if cos_joint3 < -1: cos_joint3 = -1
        self.joint3_angle = Float64()
        self.joint3_angle.data = np.arccos(cos_joint3) - np.pi / 2

        origin = self.blue_pos + np.array([3.5 * np.cos(np.pi / 2 - self.joint3_angle.data), 0, 0])
        r = 3.5 * np.sin(np.pi / 2 - self.joint3_angle.data)
        midpoint_of_semicircle_pos_j2 = origin + np.array([0, 0, r])
        dist = np.sqrt(np.sum((midpoint_of_semicircle_pos_j2 - self.green_pos) ** 2))
        cos_joint2 = (2 * 3.5 ** 2 - dist ** 2) / (2 * 3.5 * 3.5)
        if cos_joint2 < 0:  cos_joint2 = 0
        self.joint2_angle = Float64()
        self.joint2_angle.data = np.arccos(cos_joint2)

        # if self.green_pos[1] < self.blue_pos[1]:
        #     if self.joint3_angle.data > 0:
        #         self.joint2_angle.data = np.arccos(cos_joint2) * (0.85 + self.joint3_angle.data ** 2 * 0.6)
        #     else:
        #         self.joint2_angle.data = np.arccos(cos_joint2) * (0.85 + self.joint3_angle.data ** 2 * 0.9)
        #
        # else:
        #     if self.joint3_angle.data > 0:
        #         self.joint2_angle.data = -np.arccos(cos_joint2) * (0.85 + self.joint3_angle.data ** 2 * 0.5)
        #     else:
        #         self.joint2_angle.data = -np.arccos(cos_joint2) * (0.85 + self.joint3_angle.data ** 2 * 1.3)
        #
        # if self.joint2_angle.data > np.pi / 2:
        #     self.joint2_angle.data = np.pi / 2
        # if self.joint2_angle.data < -np.pi / 2:
        #     self.joint2_angle.data = -np.pi / 2

        v1 = self.blue_pos - self.green_pos
        v2 = end_of_semicircle_pos_j3 - self.green_pos
        v = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
        end_of_semicircle_pos_j4 = self.green_pos + 3 * v
        # print(self.joint4_pos, end_of_semicircle_pos_j4)
        dist = np.sqrt(np.sum((end_of_semicircle_pos_j4 - self.red_pos) ** 2))
        cos_joint4 = (2 * 3 ** 2 - dist ** 2) / (2 * 3 * 3)
        if cos_joint4 > 1:  cos_joint4 = 1
        if cos_joint4 < -1: cos_joint4 = -1
        self.joint4_angle = Float64()
        self.joint4_angle.data = np.arccos(cos_joint4) - np.pi / 2

        print(self.joint2_angle.data, self.joint3_angle.data, self.joint4_angle.data)

        # Publish the results
        try:
            self.red_posx_pub.publish(self.red_pos_x)
            self.red_posy_pub.publish(self.red_pos_y)
            self.red_posz_pub.publish(self.red_pos_z)
            self.red_pos_pub.publish(self.red_pos_f)
            self.joint2_angle_pub.publish(self.joint2_angle)
            self.joint3_angle_pub.publish(self.joint3_angle)
            self.joint4_angle_pub.publish(self.joint4_angle)
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
