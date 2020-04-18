# opencv-group-detection
This Repo gives step by step instructions and script to show how build and deploy a mobile (Respberry Pi) object detector that can be used to detect and report groups of objects detected using TensorFlow and your chosen trained model.

Boilerplate object detection code was copied from:https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py
Other suggested reading: https://github.com/vineeth2628/Object-counting-using-tensorflow-on-raspberry-pi

## Main features:
* Object detection
* Custom logging based on your object specific criteria (e.g. each time a group of more than 2 people walk past)
* Capture a photo each time your object specific criteria is triggered 
* Write the log to csv file for further analysis 

To avoid data privacy concerns in live use on people the active object viewer and photo capture should be disabled. The appendix includes instructions how to remotely access the  detection log in the Raspberry Pi for passive monitoring without using video or photo capture.

Suggested usage: Detecting groups of people not observing social distancing whilst walking through a narrow line of sight.
Note: This can be futher enhanced with object tracking to avoid overcounting of very slow moving or stationary groups.

## The main steps are as follows:
1. Set up and install TensorFlow and OpenCV on your Raspberry Pi by following this great guide by Evan https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py 
The guide walks through the following steps:
    1. Update the Raspberry Pi
    1. Install TensorFlow
    1. Install OpenCV
    1. Compile and install Protobuf
    1. Set up TensorFlow directory structure and the PYTHONPATH variable
  
zxcvbnm,  
s
