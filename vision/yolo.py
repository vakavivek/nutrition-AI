import cv2
import numpy as np
import io
from ultralytics import YOLO
from PIL import Image

yolo_model= YOLO("yolov8n.pt")

FOOD_CLASSES={
    "banana","apple","sandwich","orange","carrot","pizza","donut","cake","bowl","plain dosa","masala dosa"
}

def detect_food_objects(image_bytes:bytes,conf_threshold:float=0.3):
    """
    Detect objects (food regions) using YOLO
    Returns:
    [
      {
        "bbox":[x1,y1,x2,y2],
        "confidence":float
       }
    ]
    """
    np_img= np.frombuffer(image_bytes,np.uint8)
    img=cv2.imdecode(np_img,cv2.IMREAD_COLOR)

    if img is None:
        return []
    
    results=yolo_model(img)[0]
    best_box=None
    best_conf=0.0
    detections=[]

    for box in results.boxes:
        confidence= float(box.conf[0])
        

        if confidence<conf_threshold:
            continue

        cls_id=int(box.cls[0])
        cls_name=yolo_model.names[cls_id]

        if cls_name not in FOOD_CLASSES:
            continue

        if confidence>best_conf:
            best_conf=confidence
            x1,y1,x2,y2=map(int,box.xyxy[0])
            detections.append({
                "bbox":[x1,y1,x2,y2],
                "confidence":round(confidence,3),
                "class":cls_name
            })
    return detections

        



def crop_objects(image_bytes:bytes,detections:dict):
    np_img= np.frombuffer(image_bytes,np.uint8)
    img= cv2.imdecode(np_img,cv2.IMREAD_COLOR)

    if img is None:
        return []
    
    crops=[]

    for det in detections:
        x1,y1,x2,y2= det["bbox"]
        crop= img[y1:y2,x1:x2]
        if crop.size==0:
            continue
        pil_img=Image.fromarray(
        cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        )

        buffer= io.BytesIO()
        pil_img.save(buffer,format="JPEG")
        crops.append(buffer.getvalue())

    
    return crops