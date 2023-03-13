# Object-Detection
Object detection is a computer vision task that involves identifying and localizing objects within an image or video.
This is a Python code for object detection using YOLO model in the Ultralytics library, with the help of OpenCV and CVZone libraries.
The code reads a video stream from a webcam or a video file and applies the YOLO model to detect objects in each frame.
It then draws bounding boxes and labels around the detected objects using OpenCV and CVZone functions.

The code defines a list of class names for the objects that can be detected by the YOLO model.
It then uses the Ultralytics YOLO model to detect objects in each frame of the video stream.

The code loops over the detected boxes and draws bounding boxes using both OpenCV and CVZone functions. 
It also calculates the confidence value for each box and labels the box with the class name and
the confidence value using the putTextRect() function from CVZone library.
