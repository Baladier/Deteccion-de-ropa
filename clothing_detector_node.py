#!/usr/bin/env python3
import rospy
import rospkg
from std_msgs.msg import String
from sensor_msgs.msg import Image
from clothing_detection.msg import Box, BoxArray
from cv_bridge import CvBridge, CvBridgeError

import os
import cv2
import numpy as np
import sys
import time
from collections import OrderedDict
import ailia
from PIL import Image as PILImage

import urllib.request
import ssl
import shutil

rospack = rospkg.RosPack()

class CLOTHING_DETECTOR:
    def __init__(self):
        # Parameters
        self.CLASS_IDS = ["bag", 
                          "belt", 
                          "boots", 
                          "footwear", 
                          "outer", 
                          "dress", 
                          "sunglasses",
                          "pants", 
                          "top", 
                          "shorts", 
                          "skirt", 
                          "headwear", 
                          "scarf/tie"]
        self.THRESHOLD = 0.15		
        self.IOU = 0.4
        self.DETECTION_WIDTH = 640

        # Load the clothing detection model
        model_path = os.path.join(rospack.get_path('clothing_detection'),'models','yolov3-modanet.opt.onnx.prototxt')
        weight_path = os.path.join(rospack.get_path('clothing_detection'),'models','yolov3-modanet.opt.onnx')
        self.check_and_download_models(weight_path,model_path)
        self.detector = ailia.Net(model_path, weight_path)
        id_image_shape = self.detector.find_blob_index_by_name("image_shape")
        self.detector.set_input_shape((1, 3, self.DETECTION_WIDTH, self.DETECTION_WIDTH))
        self.detector.set_input_blob_shape((1, 2), id_image_shape)

        # Subscribers and publishers
        #self.image_sub = rospy.Subscriber('/clothing_detector/input_image',Image,self.image_cb)
        self.image_sub = rospy.Subscriber('/ros_input', Image, self.image_cb)
        self.results_pub = rospy.Publisher('/clothing_detector/results', BoxArray, queue_size=10)

        # CV bridge for image conversion
        self.bridge = CvBridge()
    
    def check_and_download_models(self,weight_path, model_path):
        """ 
        Check if the onnx file and prototxt file exists,
        and if necessary, download the files to the given path.

        Parameters
        ----------
        weight_path: string
            The path of onnx file.
        model_path: string
            The path of prototxt file for ailia.
        remote_path: string
            The url where the onnx file and prototxt file are saved.
            ex. "https://storage.googleapis.com/ailia-models/mobilenetv2/"
        """
        def urlretrieve(remote_path, weight_path):                                                                                    
            temp_path = weight_path + ".tmp"
            try:
                #raise ssl.SSLError # test
                urllib.request.urlretrieve(
                    remote_path,
                    temp_path
                )
            except ssl.SSLError as e:
                print('SSLError detected, so try to download without ssl')
                remote_path = remote_path.replace("https","http")
                urllib.request.urlretrieve(
                    remote_path,
                    temp_path
                )
            shutil.move(temp_path, weight_path)
        
        remote_path = 'https://storage.googleapis.com/ailia-models/clothing-detection/'
        if not os.path.exists(weight_path):
            print('Downloading onnx file... (save path:',weight_path,')')
            urlretrieve(
                remote_path + os.path.basename(weight_path),
                weight_path
            )
        
        if model_path!=None and not os.path.exists(model_path):
            print('Downloading prototxt file... (save path:',model_path,')')
            urlretrieve(
                remote_path + os.path.basename(model_path),
                model_path
            )

    def letterbox_image(self,image, size):
        '''resize image with unchanged aspect ratio using padding'''
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), PILImage.BICUBIC)
        new_image = PILImage.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image

    def preprocess(self,img, resize):
        image = PILImage.fromarray(img)
        boxed_image = self.letterbox_image(image, (resize, resize))
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        image_data = np.transpose(image_data, [0, 3, 1, 2])
        return image_data
    
    def post_processing(self,img_shape, all_boxes, all_scores, indices):
        indices = indices.astype(int)

        results = BoxArray()
        for idx_ in indices[0]:
            cls_ind = idx_[1]
            score = all_scores[tuple(idx_)]

            idx_1 = (idx_[0], idx_[2])
            box = all_boxes[idx_1]
            y, x, y2, x2 = box
            w = (x2 - x) / img_shape[1]
            h = (y2 - y) / img_shape[0]
            x /= img_shape[1]
            y /= img_shape[0]

            r = Box()
            r.class_id = self.CLASS_IDS[cls_ind]
            r.prob = float(score)
            r.x = int(x * img_shape[1])
            r.y = int(y * img_shape[0])
            r.width = int(w * img_shape[1])
            r.height = int(h * img_shape[0])
            results.boxes.append(r)

        return results

    def image_cb(self,msg):
        # Convert image to numpy array
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            height, width, _ = cv_image.shape
            rospy.loginfo(f"height: {height}   width: {width}")
        except CvBridgeError as e:
            print(e)

        # Initial preprocesses
        img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_shape = img.shape[:2]
        img = self.preprocess(img, resize=self.DETECTION_WIDTH)

        # feedforward
        all_boxes, all_scores, indices = self.detector.predict({
            'input_1': img,
            'image_shape': np.array([img_shape], np.float32),
            'layer.score_threshold': np.array([self.THRESHOLD], np.float32),
            'iou_threshold': np.array([self.IOU], np.float32),
        })

        # post processes
        boxes = self.post_processing(img_shape, all_boxes, all_scores, indices)

        # Publish the results
        self.results_pub.publish(boxes)

if __name__ == '__main__':
    rospy.init_node('clothing_detector_node')
    cd = CLOTHING_DETECTOR()
    rospy.spin()
