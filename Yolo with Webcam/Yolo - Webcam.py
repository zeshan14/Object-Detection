from ultralytics import YOLO
import cv2
import cvzone
import math

#fro webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 640)


#for video
#cap = cv2.VideoCapture("../Videos/traffic.mp4")
model=YOLO("../Yolo-Weights/yolov8s.pt")


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                "umbrella", "handbag" "tie", "suitcase", "frisbee", "skis",
                "snowboard", "sports ball", "kite", "baseball glove", "skateboard", "surfboard",
                "tennis racket", "bottle", "wine glass", "cup""fork", "knife", "spoon", "bowl", "banana",
                "apple", "sandwich", "orange", "broccoli", " carrot", "hot dog", "pizza", "donut", "cake"
                "chair", "sofa", "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop",
                "mouse", "remote", "keyboard", "cell phone""microwave","oven", "toaster", "sink",
                "refrigerator", "book", "clock" "vase", "scissors", "teddy bear", "hair drier",
                "toothbrush"
                    ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Bounding Box
            #opencv normal rectanle
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # print(x1,y1,x2,y2)

             # cvzone specialized rectabgle
            w, h = x2-x1 , y2-y1
            bbox = int(x1),int(y1),int(w),int(h)
            cvzone.cornerRect(img,(x1,y1,w,h))

            #Confidence Values
            conf =math.ceil((box.conf[0]*100))/100


            #Class Name
            cls=int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)),scale=1, thickness=1)




    cv2. imshow("image",img)
    cv2.waitKey(2)