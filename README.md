# YOLOv3
An implementation of YOLOv3 from scratch 

### Structure Of Project
```
📦YOLOv3
 ┣ 📂cfg
 ┃ ┗ 📜yolov3.cfg
 ┣ 📂data
 ┃ ┣ 📂Images
 ┃ ┃ ┣ 📜test.jpg
 ┃ ┃ ┗ 📜traffic.jpg
 ┃ ┣ 📂Videos
 ┃ ┃ ┗ 📜readme_data.txt
 ┃ ┗ 📜coco.names
 ┣ 📂img
 ┣ 📂vid
 ┣ 📂weights
 ┃ ┗ 📜read_weight.txt
 ┣ 📂__pycache__
 ┃ ┣ 📜utils.cpython-36.pyc
 ┃ ┗ 📜yolov3.cpython-36.pyc
 ┣ 📜convert_weights.py
 ┣ 📜image.py
 ┣ 📜utils.py
 ┣ 📜video.py
 ┗ 📜yolov3.py
 ```
 
 ### Dataset Used to train the Network:
 
 I have used the Coco datset available from here: <A href= "https://github.com/pjreddie/darknet/blob/master/data/coco.names">Coco Dataset</A>
 
 To download weights for the network go here: <A href= "https://pjreddie.com/media/files/yolov3.weights">YOLO weights</A>
 
 To download the YOLO cfg file (darknet) go here: <A href= "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg">YOLO weights</A>
 
 ### Instructions
 1. First, run convert_weights.py to convert the yolov3 weight in Tensorflow format
 2. The weights will be saved inside the "weigthts" folder and thus can be called inside image.py and video.py
 3. Run the image.py file for object detection on a image. 
 4. Run the video.py for object detection on a video.
 
 Note : To change the image and video simply, change the path
 ```
 line 22     img_path = "data/images/traffic.jpg"   
 ```
 
 ```
 line 34     cap = cv2.VideoCapture("data/Videos/traffic.mp4")
              #To capture from webcam, use cv2.VideoCapture(0)
 ```
 
 
 
 
 | File       | Purpose   | 
 |--- | --- 
 | yolov3.py | We have yolov3.cfg file but to make a model we have to parse it and make a dictionary so that we can extract layers |
 | convert_weights.py | To convert weights into Tensorflow format|
 | utils.py | Contains Helper functions |
 | image.py | Perform Object detection on images|
 | video.py | Perform Object detection on video frames|
