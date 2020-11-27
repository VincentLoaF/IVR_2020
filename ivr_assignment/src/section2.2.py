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
        self.target_posx_pub = rospy.Publisher("/target/x_position_controller/estimation", Float64, queue_size=10)
        self.target_posy_pub = rospy.Publisher("/target/y_position_controller/estimation", Float64, queue_size=10)
        self.target_posz_pub = rospy.Publisher("/target/z_position_controller/estimation", Float64, queue_size=10)

        self.robot_target_pos_est_sub1 = message_filters.Subscriber("/robot/target_position_estimation/cam1",
                                                                    Float64MultiArray)
        self.robot_target_pos_est_sub2 = message_filters.Subscriber("/robot/target_position_estimation/cam2",
                                                                    Float64MultiArray)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.robot_target_pos_est_sub1, self.robot_target_pos_est_sub2], 10, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

    def callback(self, target_pos_cam1, target_pos_cam2):
        self.target_posx = Float64()
        self.target_posx.data = target_pos_cam2.data[0]
        self.target_posy = Float64()
        self.target_posy.data = target_pos_cam1.data[0]
        self.target_posz = Float64()
        self.target_posz.data = np.mean([target_pos_cam2.data[1], target_pos_cam1.data[1]]) + 1

        # Publish the results
        try:
            self.target_posx_pub.publish(self.target_posx)
            self.target_posy_pub.publish(self.target_posy)
            self.target_posz_pub.publish(self.target_posz)
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
