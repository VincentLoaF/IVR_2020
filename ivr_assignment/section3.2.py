#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
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
        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
        self.end_eff_true_x_pub = rospy.Publisher("/robot/end_eff_position/true/x", Float64, queue_size=10)
        self.end_eff_true_y_pub = rospy.Publisher("/robot/end_eff_position/true/y", Float64, queue_size=10)
        self.end_eff_true_z_pub = rospy.Publisher("/robot/end_eff_position/true/z", Float64, queue_size=10)
        self.end_eff_close_x_pub = rospy.Publisher("/robot/end_eff_position/close/x", Float64, queue_size=10)
        self.end_eff_close_y_pub = rospy.Publisher("/robot/end_eff_position/close/y", Float64, queue_size=10)
        self.end_eff_close_z_pub = rospy.Publisher("/robot/end_eff_position/close/z", Float64, queue_size=10)

        # initialize subscribers
        self.joints = rospy.Subscriber("/robot/joint_states", JointState,self.callback)


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


    def trajectory(self):
        # get current time
        curr_time = np.array([rospy.get_time() - self.time_trajectory])
        x_d = float((2.5 * np.cos(curr_time * np.pi / 15)) + 0.5)
        y_d = float(2.5 * np.sin(curr_time * np.pi / 15))
        z_d = float((1 * np.sin(curr_time * np.pi / 15)) + 7)
        return np.array([x_d, y_d, z_d])

    def control_closed(self,input):
        print("control")
        # P gain
        K_p = np.array([[5.5, 0, 0], [0, 5.5, 0], [0, 0, 5.5]])
        # D gain
        K_d = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
        # estimate time step
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        self.time_previous_step = cur_time
        # robot end-effector position
        pos = self.detect_end_effector_fk(input[0],input[1],input[2],input[3])
        # desired trajectory
        pos_d = self.trajectory()
        # estimate derivative of error
        self.error_d = ((pos_d - pos) - self.error) / (dt+0.000000000001)
        # estimate error
        self.error = pos_d - pos
        self.end_eff_true_x_pub.publish(pos_d[0])
        self.end_eff_true_y_pub.publish(pos_d[1])
        self.end_eff_true_z_pub.publish(pos_d[2])
        J_inv = np.linalg.pinv(self.calculate_jacobian(input[0],input[1],input[2],input[3]))  # calculating the psudeo inverse of Jacobian
        dq_d = np.dot(J_inv, (np.dot(K_d, self.error_d.transpose()) + np.dot(K_p, self.error.transpose())))  # control input (angular velocity of joints)
        print("diff")
        print(dq_d)
        q_d = input + (dt * dq_d)  # control input (angular position of joints)
        return q_d

    def detect_end_effector_fk(self, theta1, theta2, theta3, theta4):
        print("detect")
        x = 3 * np.sin(theta1) * np.sin(theta2) * np.cos(theta3) * np.cos(theta4) + 3.5 * np.sin(theta1) * np.sin(theta2) * np.cos(theta3) + 3 * np.cos(theta1) * np.sin(theta3) * np.cos(theta4) + 3.5 * np.cos(theta1) * np.sin(theta3) + 3 * np.sin(theta1) * np.cos(theta2) * np.sin(theta4)

        y = -3 * np.cos(theta1) * np.sin(theta2) * np.cos(theta3) * np.cos(theta4) - 3.5 * np.cos(theta1) * np.sin(theta2) * np.cos(theta3) + 3 * np.sin(theta1) * np.sin(theta3) * np.cos(theta4) + 3.5 * np.sin(theta1) * np.sin(theta3) - 3 * np.cos(theta1) * np.cos(theta2) * np.sin(theta4)

        z = 3 * np.cos(theta2) * np.cos(theta3) * np.cos(theta4) + 3.5 * np.cos(theta2) * np.cos(theta3) - 3 * np.sin(theta2) * np.sin(theta4) + 2.5

        end_eff_pos = np.array([x, y, z])
        return end_eff_pos

    def callback(self,data):
        angles = np.array(data.position)
        print(angles)
        q_d = self.control_closed(angles)
        end_eff_close = self.detect_end_effector_fk(q_d[0], q_d[1], q_d[2], q_d[3])
        self.end_eff_close_x_pub.publish(end_eff_close[0])
        self.end_eff_close_y_pub.publish(end_eff_close[1])
        self.end_eff_close_z_pub.publish(end_eff_close[2])
        print(q_d)
        self.robot_joint2_pub.publish(q_d[0])
        self.robot_joint2_pub.publish(q_d[1])
        self.robot_joint3_pub.publish(q_d[2])
        self.robot_joint4_pub.publish(q_d[3])
        # Publish the results

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
