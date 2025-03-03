import cv2
import pytesseract
import sqlite3
import torch
from ultralytics import YOLO

# Load YOLOv8 model (Ensure you have a pre-trained model for license plate detection)
model = YOLO('yolov8n.pt')  # Replace with a trained model for license plates

# Initialize database
conn = sqlite3.connect('anpr.db')
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS vehicle_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate_number TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change to video file path if needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect license plates
    results = model(frame)
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_img = frame[y1:y2, x1:x2]
            
            # Preprocess image for OCR
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Perform OCR
            plate_text = pytesseract.image_to_string(thresh, config='--psm 7')
            plate_text = ''.join(filter(str.isalnum, plate_text))
            
            if plate_text:
                print("Detected Plate:", plate_text)
                cursor.execute("INSERT INTO vehicle_data (plate_number) VALUES (?)", (plate_text,))
                conn.commit()
            
            # Draw bounding box and plate text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imshow('ANPR System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
