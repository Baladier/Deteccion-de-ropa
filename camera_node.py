#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import cv2

def main():
	# El indice de instancia depende de como este identificado en el ordenador
	cap = cv2.VideoCapture(0)
	rospy.init_node('camera_frame', anonymous=True)
	pub = rospy.Publisher('/image_output', Image, queue_size=15)
	bridge = CvBridge()
	
	rate = rospy.Rate(10)
	while not rospy.is_shutdown():
		try:
			ret, frame = cap.read()
			if not ret:
				break
			ros_image = bridge.cv2_to_imgmsg(frame, "rgb8")
			pub.publish(ros_image)
			rospy.loginfo("imagen publicada")
		except CvBridgeError as e:
			rospy.logerr(e)
		
		rate.sleep()

if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		pass
