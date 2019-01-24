#!/usr/bin/env python
import numpy as np
import argparse
import imutils
import time
import rospy
import sys
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8
import rospkg

class PostmanDetector():
    def __init__(self):
        rospy.loginfo("Start Postman Detector Init process...")
        # get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()

        self.bridge_object = CvBridge()
        rospy.loginfo("Start camera suscriber...")
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.camera_callback)
        self.pub_Postman = rospy.Publisher("/is_a_postman",UInt8, queue_size=10)
        rospy.loginfo("Finished Postman Detector Init process...Ready")

    def camera_callback(self, data):
        self.postmanDetect(data)

    def sum_white(self,img):
        # get total number of pixels in image
        dimensions = img.shape
        total_pix = dimensions[0] * dimensions[1]
        n_white_pix = float(np.sum(img == 255))
        percent = n_white_pix / total_pix * 100
        return percent

    def postScan(self,img):
        YELLOW_MIN = np.array([20, 100, 100])  # (best)yellow min 20, 100, 100
        YELLOW_MAX = np.array([30, 255, 255])  # (best)yellow max 30, 255, 255
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        frame_threshed = cv2.inRange(hsv_img, YELLOW_MIN, YELLOW_MAX)
        percentage = self.sum_white(frame_threshed)
        per_round = round(percentage, 2)
        print('[INFO]Yellow percentage: ', str(per_round) + '%')
        return per_round

    def postmanDetect(self,data):
        # Get a reference to webcam #0 (the default one)
        try:
            # We select bgr8 because its the OpneCV encoding by default
            video_capture = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        # construct the argument parse and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-p", "--prototxt", required=True,
                        help="path to Caffe 'deploy' prototxt file")
        ap.add_argument("-m", "--model", required=True,
                        help="path to Caffe pre-trained model")
        ap.add_argument("-c", "--confidence", type=float, default=0.2,
                        help="minimum probability to filter weak detections")
        args = vars(ap.parse_args(rospy.myargv()[1:]))

        # initialize the list of class labels MobileNet SSD was trained to
        # detect, then generate a set of bounding box colors for each class
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        is_postman = UInt8()
        is_postman.data = 0

        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = cv2.resize(video_capture,  (0, 0), fx=0.5, fy=0.5)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                category = "{}".format(CLASSES[idx])
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                # EDIT START
                if confidence >= 0.999 and category == "person":
                    human = frame[startY:endY, startX:endX]
                    is_postman.data = 1
                    threshold = self.postScan(human)
                    print("human")
                    if threshold > 30:
                        is_postman.data = 2
            self.pub_Postman.publish(is_postman)



def main():
    rospy.init_node('Postman_detector_python_node', argv=sys.argv, anonymous=True)
    sys.argv = rospy.myargv(argv=sys.argv)

    DetectPostMan = PostmanDetector()

    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
