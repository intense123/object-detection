#libraries
from ultralytics import YOLO
import cv2
import cvzone
import math


#For live
#cap = cv2.VideoCapture(0)
#cap.set(3,1280)
#cap.set(4,720)
#model = YOLO("../Yolo-weight/yolov8m.pt")

#For premade videos
cap = cv2.VideoCapture("../Videos/plant disease.mp4")
model = YOLO("Bestn.pt")

#classnames = ["Person", "bicyle", "car", "motorbike", "aeroplane", "tree", "bus", "train", "truck", "boat", "traffic Light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sport sball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potter plant", "bed", "diningtable", "toilet", "tvmon", "Laptop", "remote", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush"]
classnames = ["Blight"]
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            print(x1, y1, x2, y2)
            #w, h = x2-x1, y2-y1
            #bbox = int(x1), int(y1), int(x2), int(y2)
            cvzone.cornerRect(img, (x1, y1, x2, y2))

            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            cvzone.putTextRect(img, f"{conf}", (max(0, x1), max(35, y1)))

            #classnames
            cls = int(box.cls[0])


            cvzone.putTextRect(img, f"{classnames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=3, thickness=3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)