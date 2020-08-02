######## Detecting and capturing groups of objects using tensorflow on picamera #########
#> Author: Tahir Mahmood
#> Date: 15/4/20
#> Description: This code draws upon standard examples in object detection using opencv to: 
#>> Use a TensorFlow classifier to perform live object detection viewing and counting.
#>> Identify and count and keep a tally based on only specified class(es) returned by the tensorflow model. 
#>> Log whenever a specified number of a specified class is returned and also capture an image of each example.

#> Suggested usage: Detecting groups of people not observing social distancing whilst walking through a narrow line of sight.
#> Note: This can be futher enhanced with object tracking to avoid overcounting of very slow moving or stationary groups.

## ## The boilerplate part of the code is copied from:
## https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py
## And draws from ideas in: https://github.com/vineeth2628/Object-counting-using-tensorflow-on-raspberry-pi

# Import packages
import os
import cv2
import numpy as np
import pandas as pd
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import time
import csv

######## BOILERPLATE CODE #######
# Set up camera constants
#IM_WIDTH = 1280
#IM_HEIGHT = 720
IM_WIDTH = 640   # Use smaller resolution for
IM_HEIGHT = 480  # slightly faster framerate

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
parser.add_argument('--video', help='Name of the video file',
                    default='test.mp4')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'
VIDEO_NAME = args.video

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

## Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#Initialises the list for output
output = []

# Initialize Picamera and grab reference to the raw capture
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (1280,720))


# creating a fucntion 
def group_counting():
    
    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    font = cv2.FONT_HERSHEY_SIMPLEX

    while(video.isOpened()): # Uncomment block for recorded video input
    # Acquire frame and resize to expected shape [1xHxWx3]
    # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()
    # On the next loop set the value of these objects as old for comparison
        ret, frame1 = video.read()
        if not ret:
            print('Reached the end of the video!')
            break

    #Standard setup for the live object viewer
    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
        #t1 = cv2.getTickCount()
      
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = frame1.copy()
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Visualizing the results of the detection - 
        # this is only for show and to visually verify results if needed
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4,
            min_score_thresh=0.45)
        
        ####### OBJECT SELECTION AND COUNTING CODE STARTS HERE #######
        # pulling raw output from object detection. Creates a list of dicts 
        # with details of each of the objects meeting the threshold in a given frame.
        Validobj = [category_index.get(value) for index, value in enumerate (classes[0]) if scores [0,index]>0.4]
        
        # Choose your object
        to_detect = 'person' 
        
        # Creates a log if the chosen object has been detected.
        if Validobj:
            data = [i["name"] for i in Validobj]
            # If in the given frame the number of a given object detected meets the condition then a log is made   
            if data.count(to_detect)>1:
                # Writes a line with how many of the object was detected along with a timestamp
                Summary = ["Atleast " + str(data.count(to_detect)) + " people detected"]
                print(Summary, time.ctime())
                
                evidence_stamp = [data.count(to_detect),to_detect,time.ctime()]
                output.append(evidence_stamp)
                cv2.putText(frame,"ALERT: {}".format(Summary),(30,85),font,1,(60,60,255),2,cv2.LINE_AA)
                # Take a picture for authorities
                #cv2.imwrite("evidence.bmp", frame)
                #time.sleep(5) #- alter depending on footfall or replace with object tracking to reduce overcounting
        
        # Used to dispay framerate in live viewer
        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        vidout = cv2.resize(frame,(1280,720))
        out.write(vidout)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
 
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            # This writes the data gathered in the output to a logfile
            with open('output.csv','w',newline = '\n') as file:
                writer = csv.writer(file)
                writer.writerows(output)
            break

    cv2.destroyAllWindows()
    video.release() #for recorded video
    out.release() 

try:
    while True:
        group_counting()
except KeyboardInterrupt:
    sys.exit()
