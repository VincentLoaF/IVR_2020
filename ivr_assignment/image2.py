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


class image_converter:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send images from camera2 to a topic named image_topic2
        self.image_pub2 = rospy.Publisher("image_topic2", Image, queue_size=1)
        # initialize a subscriber to receive messages rom a topic named /robot/camera1/image_raw and use callback
        # function to receive data
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        # initialize a publisher to send estimated joints' position to the robot
        self.robot_blue_pos_est_pub2 = rospy.Publisher("/robot/blue_position_estimation/cam2", Float64MultiArray,
                                                       queue_size=10)
        self.robot_green_pos_est_pub2 = rospy.Publisher("/robot/green_position_estimation/cam2", Float64MultiArray,
                                                        queue_size=10)
        self.robot_red_pos_est_pub2 = rospy.Publisher("/robot/red_position_estimation/cam2", Float64MultiArray,
                                                      queue_size=10)
        self.robot_target_pos_est_pub2 = rospy.Publisher("/robot/target_position_estimation/cam2", Float64MultiArray,
                                                      queue_size=10)
        self.robot_obstacle_pos_est_pub1 = rospy.Publisher("/robot/obstacle_position_estimation/cam1",
                                                           Float64MultiArray,
                                                           queue_size=10)
        # record the beginning time
        self.start_time = rospy.get_time()

    def detect_yellow(self, image):
        img_y = cv2.inRange(image, (0, 100, 100), (5, 255, 255))
        # cv2.imshow('img_y', img_y)
        kernel = np.ones((5, 5), np.uint8)
        img_y = cv2.dilate(img_y, kernel, iterations=3)
        # cv2.imwrite('img_thresh_y.png', img_y)
        M = cv2.moments(img_y)
        if M['m00'] == 0:
            return self.prev_yellow
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centre = np.array([cx, cy])
        self.prev_yellow = centre
        return centre

    def detect_blue(self, image):
        img_b = cv2.inRange(image, (100, 0, 0), (255, 5, 5))
        # cv2.imshow('img_b', img_b)
        kernel = np.ones((5, 5), np.uint8)
        img_b = cv2.dilate(img_b, kernel, iterations=3)
        # cv2.imwrite('img_thresh_b.png', img_b)
        M = cv2.moments(img_b)
        if M['m00'] == 0:
            return self.prev_blue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        ground = self.detect_yellow(image)
        centre = np.array([cx - ground[0], ground[1] - cy])
        self.prev_blue = centre
        return centre

    def detect_green(self, image):
        img_g = cv2.inRange(image, (0, 100, 0), (5, 255, 5))
        # cv2.imshow('img_g', img_g)
        kernel = np.ones((5, 5), np.uint8)
        img_g = cv2.dilate(img_g, kernel, iterations=3)
        # cv2.imwrite('img_thresh_g.png', img_g)
        M = cv2.moments(img_g)
        if M['m00'] == 0:
            return self.prev_green
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        ground = self.detect_yellow(image)
        centre = np.array([cx - ground[0], ground[1] - cy])
        self.prev_green = centre
        return centre

    def detect_red(self, image):
        img_r = cv2.inRange(image, (0, 0, 100), (5, 5, 255))
        # cv2.imshow('img_r', img_r)
        kernel = np.ones((5, 5), np.uint8)
        img_r = cv2.dilate(img_r, kernel, iterations=3)
        # cv2.imwrite('img_thresh_r.png', img_r)
        M = cv2.moments(img_r)
        if M['m00'] == 0:
            return self.prev_red
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        ground = self.detect_yellow(image)
        centre = np.array([cx - ground[0], ground[1] - cy])
        self.prev_red = centre
        return centre

    def detect_target(self, image):
        img_t = cv2.inRange(image, (5, 50, 100), (50, 200, 255))
        # cv2.imshow('img_t', img_t)
        ret, binary = cv2.threshold(img_t, 0, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            if len(contours[i]) > 10:
                # cv2.imwrite('img_thresh_t.png', contours[i])
                M = cv2.moments(contours[i])
                if M['m00'] == 0:
                    return self.prev_target
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                ground = self.detect_yellow(image)
                centre = np.array([cx - ground[0], ground[1] - cy])
                self.prev_target = centre
                return centre

        return self.prev_target

    def detect_obstacle(self, image):
        img_t = cv2.inRange(image, (5, 50, 100), (50, 200, 255))
        # cv2.imshow('img_t', img_t)
        ret, binary = cv2.threshold(img_t, 0, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            if len(contours[i]) < 10:
                # cv2.imwrite('img_thresh_t.png', contours[i])
                M = cv2.moments(contours[i])
                if M['m00'] == 0:
                    return self.prev_obstacle
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                ground = self.detect_yellow(image)
                centre = np.array([cx - ground[0], ground[1] - cy])
                self.prev_obstacle = centre
                return centre

        return self.prev_obstacle

    def pixel2meter(self, image, point):
        centroid_b = np.sqrt(np.sum(self.detect_blue(image) ** 2))
        return 2.5 / centroid_b * point

    # Receive data, process it, and publish
    def callback2(self, data):
        # Receive the image
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Uncomment if you want to save the image
        # cv2.imwrite('image_copy.png', cv_image)

        im2 = cv2.imshow('window2', self.cv_image2)
        cv2.waitKey(1)

        blue_pos = Float64MultiArray()
        blue_pos.data = self.pixel2meter(self.cv_image2, self.detect_blue(self.cv_image2))
        green_pos = Float64MultiArray()
        green_pos.data = self.pixel2meter(self.cv_image2, self.detect_green(self.cv_image2))
        red_pos = Float64MultiArray()
        red_pos.data = self.pixel2meter(self.cv_image2, self.detect_red(self.cv_image2))
        target_pos = Float64MultiArray()
        target_pos.data = self.pixel2meter(self.cv_image2, self.detect_target(self.cv_image2))
        obstacle_pos = Float64MultiArray()
        obstacle_pos.data = self.pixel2meter(self.cv_image2, self.detect_obstacle(self.cv_image2))

        # Publish the results
        try:
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
            self.robot_blue_pos_est_pub2.publish(blue_pos)
            self.robot_green_pos_est_pub2.publish(green_pos)
            self.robot_red_pos_est_pub2.publish(red_pos)
            self.robot_target_pos_est_pub2.publish(target_pos)
            self.robot_obstacle_pos_est_pub1.publish(obstacle_pos)
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
