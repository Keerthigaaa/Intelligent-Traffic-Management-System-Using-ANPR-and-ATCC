import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO

# Load pre-trained YOLOv8 model (Ensure you have a model trained for vehicle classification)
model = YOLO('yolov8n.pt')

# Define vehicle categories (class IDs may vary based on the dataset used for training)
VEHICLE_CLASSES = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

# Traffic light states
green_light = True  # Initially, green light is on
red_light = False

def adjust_traffic_signal(vehicle_count):
    global green_light, red_light
    
    if vehicle_count > 10:
        green_light = True
        red_light = False
    else:
        green_light = False
        red_light = True

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change to video file path if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    vehicle_count = 0
    
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            if class_id in VEHICLE_CLASSES:
                vehicle_count += 1
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = VEHICLE_CLASSES[class_id]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    adjust_traffic_signal(vehicle_count)
    
    signal_text = "Green Light" if green_light else "Red Light"
    color = (0, 255, 0) if green_light else (0, 0, 255)
    cv2.putText(frame, signal_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    cv2.imshow('ATCC System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
