# opencv-group-detection
This Repo gives step by step instructions and script to show how build and deploy a mobile (Respberry Pi) object detector that can be used to detect and report groups of objects detected using TensorFlow and your chosen trained model.

Boilerplate object detection code was copied from: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py


## Main features:
* Object detection
* Custom logging based on your object specific criteria (e.g. each time a group of more than 2 people walk past)
* Capture a photo each time your object specific criteria is triggered 
* Write the log to csv file for further analysis 


To avoid data privacy concerns in live use on people the active object viewer and photo capture should be disabled. The appendix includes instructions how to remotely access the  detection log in the Raspberry Pi for passive monitoring without using video or photo capture.


Suggested usage: Detecting groups of people not observing social distancing whilst walking through a narrow line of sight.
Note: This can be futher enhanced with object tracking to avoid overcounting of very slow moving or stationary groups.


## The main steps are as follows:
1. **Set up and install TensorFlow and OpenCV on your Raspberry Pi** by following this great guide by Evan https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi/blob/master/Object_detection_picamera.py 
The guide walks through the following steps:
    1. Update the Raspberry Pi
    1. Install TensorFlow
    1. Install OpenCV
    1. Compile and install Protobuf
    1. Set up TensorFlow directory structure and the PYTHONPATH variable
1. **Make sure your camera is configured** by following these instructions https://www.raspberrypi.org/documentation/configuration/camera.md
1. Download or clone this Repo and put the *open_cv_group_detection.py* in your /object_detection directory
1. (optional) **Customisation**
   * Select a custom model and number of objects 

```# Name of the directory containing the object detection module we're using
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
```

  * Select which objects to include in the log file
```
 # pulling raw output from object detection. Creates a list of dicts 
        # with details of each of the objects meeting the threshold in a given frame.
        Validobj = [category_index.get(value) for index, value in enumerate (classes[0]) if scores [0,index]>0.5]
        
        # Choose your object
        to_detect = 'person' 
```  
  * Select which criteria to apply for logging in the evidence stamp
```
        # Creates a log if the chosen object has been detected.
        if Validobj:
            data = [i["name"] for i in Validobj]
            # If in the given frame the number of a given object detected meets the condition then a log is made   
            if data.count(to_detect)>2:
                # Writes a line with how many of the object was detected along with a timestamp
                Summary = ["There is a group of " + str(data.count(to_detect)) + " people" ,time.ctime()]
                print(Summary)
                
                evidence_stamp = [data.count(to_detect),to_detect,time.ctime()]
                output.append(evidence_stamp)
```
  * Specify the save location for the log file and image captures (by default this is the working directory).

5. **Run** the *open_cv_group_detection.py* from your /object_detection directory

## Appendix: Remote logging (Windows 10 example)
1. Comment out the following sections in *open_cv_group_detection.py*
```
                # Take a picture for authorities
                cv2.imwrite("evidence.bmp", frame)
                time.sleep(5) #- alter depending on footfall or replace with object tracking to reduce overcounting
        
        # Used to dispay framerate in live viewer
        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
```
1. Follow the instructions to set up SSH  here https://www.raspberrypi.org/documentation/remote-access/ssh/windows10.md
1. **Run** the *open_cv_group_detection.py* from your /object_detection directory
