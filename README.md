# opencv-group-detection
Instructions and script to build and deploy a mobile object detector that can be used to detect and report groups of people
#> Description: This code draws upon standard examples in object detection using opencv to: 
# Uses a TensorFlow classifier to perform object detection and counting.
# Identify and count and keep a tally based on only specified class(es) returned by the tensorflow model. 
# Log whenever a specified number of a specified class is returned and also capture an image of each example.
#> Suggested usage: Detecting groups of people not observing social distancing whilst walking through a narrow line of sight.
#> Note: This can be futher enhanced with object tracking to avoid overcounting of very slow moving or stationary groups.

## Some of the code is copied from:
## https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py
## And draws from ideas in:
## https://github.com/vineeth2628/Object-counting-using-tensorflow-on-raspberry-pi
