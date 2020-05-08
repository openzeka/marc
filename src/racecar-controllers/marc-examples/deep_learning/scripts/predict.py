#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import rospkg
import cv2
import matplotlib.pylab as plt
import rospy
import sys, os
import json, time, math
import argparse
from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan, Image, Joy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
import tensorflow as tf
import threading
from cv_bridge import CvBridge, CvBridgeError
rospy.init_node('predict')
K.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
print('GPUs: ', gpus)
if gpus:
    # Limit GPU usages
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

parser = argparse.ArgumentParser()
parser.add_argument('--speed', nargs='?', type=float, help='Speed of the car', default=0.5, const=0.5)
args = parser.parse_args()

class BaseClass(object):
    """
    Base class for autonomous driving. 
    This class reads the pre-trained model, fetches the image from the ZED camera rostopic, and predicts the next steering angle.
    Speed is constant. The car will not move until a successful inference result is produced. 
    
    Methods
    -------
    get_img(img)
        Gets the OpenCV image and crops the top of it
        It returns the processed image.
    
    zed_callback(data)
        ZED rostopic callback. It is fired when the new image is available.
        
    inference()
        Predicts the steering angle. 
        
    pipeline()
        Publish the predicted steering angle and speed
       
    nn_model()
        Loads the Deep Learning model and returns it.
    """
    def __init__(self):
        # Required for converting the ROS images to OpenCV images
        self.bridge = CvBridge()
        
        # This will be our inference result. Assigned to None for now.
        self.out = None
        
        # Car steering angle
        self.angle = 0.0
        
        # Car speed
        self.speed = 0
        
        self.rate = rospy.Rate(20)
        
        # Debug mode. See what the car is seeing
        self.debug = True
        
        # Our ZED Image
        self.image = None
        
        # For removing the irrelevent information from our raw image, crop the top of it. 
        # The same thing wedo when training
        self.cropped_pixel_from_top = 100
        rospack = rospkg.RosPack()
        
        # Models are read from this directory
        self.model_name = os.environ['HOME'] + '/marc_models/model_new'
        
        # Initialize our NN model
        self.model = self.nn_model()
        
        self.finished = False   
           
        # For the smooth motion of the car, run inference operation in other threads since it is blocking process. 
        self.t = threading.Thread(target=self.inference)
        self.t.start()
        
        # Subscriber and Publisher
        rospy.Subscriber('/zedm/zed_node/right/image_rect_color', Image, self.zed_callback, queue_size=1)
        self.pub = rospy.Publisher('/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=1)

    def get_img(self, img):
        img = cv2.resize(img, (320, 180))
        img = img[self.cropped_pixel_from_top:,:,:]
        img = img.reshape(1, 80, 320, 3)
        return img

    def zed_callback(self, data):
        # Convert the image to OpenCV format
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        if self.debug:
            cv2.imshow('Image', self.image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                pass
    
    def inference(self):
        while not self.finished:
            if not self.image is None:
                cv2_img = self.get_img(self.image)
                self.out = self.model.predict(cv2_img, batch_size=1)
                self.angle = self.out[0][0]
                self.speed = args.speed

    def pipeline(self):

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = self.angle
        msg.drive.speed = self.speed
        if self.debug:
            rospy.loginfo('Predicted angle:' + str(self.angle))
        self.pub.publish(msg)
        self.rate.sleep()

    def nn_model(self):
        jstr = json.loads(open(self.model_name + '.json').read())
        model = model_from_json(jstr)
        model.load_weights(self.model_name + '.h5')
        return model
        
drive = BaseClass()
        
if __name__ == '__main__':
    while not rospy.is_shutdown():
        drive.pipeline()
    drive.finished = True
    rospy.spin()
    sys.exit(0)

