#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ultralytics import YOLO
import torch
from torchvision.ops import nms
import numpy as np
from pynput import keyboard
from hsvHistogramCode import ImageProcessing
from publish_image import ImageProcessing as publish_image
import os
from datetime import datetime
from clothing_detection.msg import Box, BoxArray
from compareFrame import EnterFrame

class YoloVisualization:
	def __init__(self):
		self.bridge = CvBridge()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = YOLO('yolov8n.pt')
		# Suscribe to the topic
		self.pub = rospy.Subscriber('/image_output', Image, self.yoloDetection)
		self.bounding_box = rospy.Subscriber('/clothing_detector/results', BoxArray, self.clothingDetector)
		self.send = rospy.Publisher('/yolo_detection', Image, queue_size = 1)
		self.listener = keyboard.Listener(on_press=self.on_press)
		self.listener.start()
		self.latest_frame = None
		self.roi = None
		self.roi_counter = 0
		self.detection = False
		self.detection_counter = 0
		self.main_folder = None
		self.subfolder_frame = None
		self.subfolder_histogram = None
		self.subfolder_boundingBoxes = None
		self.carpet_created = False
		self.person_rois = []
		self.following_state = False
		self.compare_frame = EnterFrame()
		self.base_path = "/home/alan/imgtest/results"
		self.compare_frame.getMainFolder(self.base_path)
		self.histo = None
	
	# Acceso manual a los distintos tipos de funciones	
	def on_press(self, key):
		try:
			if key.char == 'q':
				rospy.loginfo("Entro")
				self.detection = True
			if key.char == 'd' and self.following_state == False:
				check = self.compare_frame.checkDatabase()
				if check == True:
					if self.carpet_created == False: 
						rospy.loginfo("Creating Database location")
						rospy.loginfo("Para crear una buena base de datos, procura que el muestreo se realice entre 25 a 40 fotografias")
						create_folders = self.createDataBase()
						self.carpet_created = True
					else:
						rospy.logerr("DataBase creado, no se puede crear otro")
				else:
					self.following_state = True
			if key.char == 'f':
				self.following_state = True
				self.compare_frame.getInfo(self.subfolder_boundingBoxes, self.subfolder_histogram)
					
		except AttributeError:
			pass
			
	# Funcion callback cuando se detecta en el tópico '/yolo_detection'	
	def yoloDetection(self, data):
		try:
			# Conversión del mensaje ROS a un frame de OpenCV
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
			height, width, _ = cv_image.shape

			# Procesamiento en la GPU sin preprocesado
			results = self.model(cv_image)
			results = results[0].cpu()
			self.person_rois.clear()
			# Procesa solo detecciones de personas
			if results and hasattr(results, 'boxes'):
				for box in results.boxes:
					for i in range(box.xyxy.shape[0]):
						cls = box.cls[i]
						if(self.model.names[int(cls)] == "person"):
							x1, y1, x2, y2 = map(int, box.xyxy[i])
							conf = box.conf[i]
							label = f"{self.model.names[int(cls)]} {conf:.2f}"
							cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
							cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
							roi = cv_image[y1:y2, x1:x2]
							self.person_rois.append(roi)

					if self.detection:
						for idx, roi in enumerate(self.person_rois):
							histo_create = ImageProcessing(roi, self.roi_counter, self.subfolder_histogram, self.following_state)
							self.histo = histo_create.getHistogramFrame()
							if not self.following_state:
								cv2.imwrite(f"{self.subfolder_frame}/roi_{self.roi_counter}.jpg", roi)
							publish_image(self.roi_counter, self.subfolder_frame, roi, self.following_state)
							self.roi_counter += 1
							self.detection = False
							rospy.loginfo("Guardado")
			image_send = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
			self.send.publish(image_send)
		except CvBridgeError as e:
			print(e)
	
	# Creacion de la base de datos		
	def createDataBase(self):
		roi_folder = "rois"
		histogram_folder = "histograms" 
		boundingBoxes_folder = "bounding_boxes" 
		date = datetime.now()
		#time and date in a String
		timestamp = date.strftime("%Y%m%d_%H%M%S")
		try:
			self.main_folder = os.path.join(self.base_path, f"main_folder_{timestamp}")
			self.subfolder_frame = os.path.join(self.main_folder, f"{roi_folder}")
			self.subfolder_histogram = os.path.join(self.main_folder, f"{histogram_folder}")
			self.subfolder_boundingBoxes = os.path.join(self.main_folder, f"{boundingBoxes_folder}")
			os.makedirs(self.main_folder, exist_ok=True)
			os.makedirs(self.subfolder_frame,exist_ok=True)
			os.makedirs(self.subfolder_histogram,exist_ok=True)
			os.makedirs(self.subfolder_boundingBoxes,exist_ok=True)
			self.compare_frame.getMainFolder(self.main_folder)
		except Exception as e:
			rospy.logerr(f"{e}")
	
	# Callback del nodo "ailia_models"		
	def clothingDetector(self, data):
		try:
			rospy.loginfo(f"{data}")
			if (self.following_state == False):
				num_boxes = len(data.boxes)
				if(num_boxes != 0):
					with open(f'{self.subfolder_boundingBoxes}/box_{self.roi_counter}.txt', 'a') as file:
						for box in data.boxes:
							file.write(f"class id: {box.class_id}\n")
							file.write(f"prob: {box.prob}\n")
							file.write(f"x: {box.x}\n")
							file.write(f"y: {box.y}\n")
							file.write(f"width: {box.width}\n")
							file.write(f"height: {box.height}\n")
							file.write("------\n")
				else:
					rospy.loginfo("NO DETECTA NADA")
			else: 
				self.compare_frame.sendInfoToCompare(self.histo, data)
		except Exception as e:
			rospy.logerr(f"{e}")
		
			
def main():
	# Link with the node
	rospy.init_node('yolo_node', anonymous=True)
	rate = rospy.Rate(30)
	# Instancia de la clase
	test = YoloVisualization()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	
if __name__ == '__main__':
	main()

