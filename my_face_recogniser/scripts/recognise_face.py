#!/usr/bin/env python
import os
import rospy
import cv2
import face_recognition
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import UInt8
import rospkg



class FaceRecogniser(object):

    def __init__(self):
        rospy.loginfo("Start FaceRecogniser Init process...")
        # get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()
        # get the file path for face_recognition_pkg
        #self.path_to_package = rospack.get_path('face_recognition_pkg')
        self.bridge_object = CvBridge()
        rospy.loginfo("Start camera suscriber...")
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw",Image,self.camera_callback)
        rospy.loginfo("Finished FaceRecogniser Init process...Ready")
        self.pubDoctor = rospy.Publisher('/visitor_detector/is_doctor', UInt8, queue_size=10)

    def camera_callback(self,data):
        
        self.recognise(data)

    def recognise(self,data):
        
        # Get a reference to webcam #0 (the default one)
        try:
            # We select bgr8 because its the OpneCV encoding by default
            video_capture = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        
        
        # Load a sample picture and learn how to recognize it.
        image_path = "/home/ferid/catkin_ws/src/detect/my_face_recogniser/person_img/Ferid.jpg"
        standing_person_image = face_recognition.load_image_file(image_path)
        standing_person_face_encoding = face_recognition.face_encodings(standing_person_image)[0]
        
        
        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        # Resize frame of video to 1/2 size for faster face recognition processing
        # If this is done be aware that you will have to make the recognition nearer.
        # In this case it will work around maximum 1 meter, more it wont work properly
        small_frame = cv2.resize(video_capture, (0, 0), fx=0.5, fy=0.5)
        #cv2.imshow("SMALL Image window", small_frame)
        is_a_doctor = UInt8()
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
    
            if not face_encodings:
                rospy.logwarn("No Faces found, please get closer...")
                is_a_doctor.data = 2
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                match = face_recognition.compare_faces([standing_person_face_encoding], face_encoding)
                name = "Unknown"
                if match[0]:
                    rospy.loginfo("MATCH")
                    name = "StandingPerson"
                    is_a_doctor.data = 1

                else:
                    rospy.logwarn("NO Match")
                    is_a_doctor.data = 0
                self.pubDoctor.publish(is_a_doctor)


def main():
    rospy.init_node('face_recognising_python_node', anonymous=False)
   
    line_follower_object = FaceRecogniser()

    rospy.spin()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    main()
