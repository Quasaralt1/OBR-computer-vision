from ultralytics import YOLO
import time

def detect_heads_yolo(image, model_path='yolov8n.pt'):
    """Detect heads using YOLOv8"""
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    start_time = time.time()
    results = model(image, classes=[0], verbose=False)  # Class 0 = person
    inference_time = time.time() - start_time
    
    # Process results
    output_image = image.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        head_height = int((y2 - y1) * 0.3)  # Assume head is top 30%
        
        # Draw bounding box
        cv2.rectangle(output_image, 
                     (x1, y1), 
                     (x2, y1 + head_height), 
                     (0, 255, 0), 2)  # Green box
    
    return output_image, inference_time

image = load_image("data/head2.jpeg")
yolo_result, yolo_time = detect_heads_yolo(image)    
show_image(yolo_result, "YOLO RESULT")
print("yolo_time", yolo_time)