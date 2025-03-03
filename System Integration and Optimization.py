import cv2
import pytesseract
import sqlite3
import torch
import numpy as np
from ultralytics import YOLO

# Load pre-trained YOLOv8 models for license plate detection and vehicle classification
anpr_model = YOLO('yolov8n.pt')  # Replace with trained ANPR model
atcc_model = YOLO('yolov8n.pt')  # Replace with trained vehicle classification model

# Define vehicle categories
VEHICLE_CLASSES = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

green_light = True  # Initial traffic light state
red_light = False

# Initialize database
conn = sqlite3.connect('traffic_management.db')
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS vehicle_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT,
    vehicle_type TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

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
    
    # ANPR - Detect and recognize license plates
    anpr_results = anpr_model(frame)
    
    for r in anpr_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = frame[y1:y2, x1:x2]
            
            # Preprocess image for OCR
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Perform OCR
            plate_text = pytesseract.image_to_string(thresh, config='--psm 7')
            plate_text = ''.join(filter(str.isalnum, plate_text))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # ATCC - Detect and classify vehicles
    atcc_results = atcc_model(frame)
    vehicle_count = 0
    vehicle_type = ""
    
    for r in atcc_results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            if class_id in VEHICLE_CLASSES:
                vehicle_count += 1
                vehicle_type = VEHICLE_CLASSES[class_id]
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = VEHICLE_CLASSES[class_id]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Store recognized data in database
    if plate_text and vehicle_type:
        cursor.execute("INSERT INTO vehicle_data (plate_number, vehicle_type) VALUES (?, ?)", (plate_text, vehicle_type))
        conn.commit()
    
    # Adjust traffic signal based on vehicle count
    adjust_traffic_signal(vehicle_count)
    signal_text = "Green Light" if green_light else "Red Light"
    color = (0, 255, 0) if green_light else (0, 0, 255)
    cv2.putText(frame, signal_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    cv2.imshow('Unified Traffic Management System', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
