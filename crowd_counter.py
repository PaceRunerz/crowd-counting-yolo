import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import sqlite3
import time
import argparse

class CrowdCounter:
    def __init__(self, video_source=0, weights="yolov3.weights", config="yolov3.cfg", names="coco.names"):
        self.video_source = video_source
        self.weights = weights
        self.config = config
        self.names = names
        
        # Initialize database
        self.conn = sqlite3.connect("crowd_data.db")
        self.cursor = self.conn.cursor()
        self._init_db()
        
        # Load YOLO model
        self.net = cv2.dnn.readNet(self.weights, self.config)
        self.classes = self._load_classes()
        self.output_layers = self._get_output_layers()
        
        # Visualization setup
        self.crowd_counts = deque(maxlen=50)
        self.frame_numbers = deque(maxlen=50)
        self._init_visualization()
        
    def _init_db(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS CrowdCount (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_index INTEGER,
            timestamp TEXT,
            count INTEGER
        )
        """)
        self.conn.commit()
        
    def _load_classes(self):
        with open(self.names, "r") as f:
            return [line.strip() for line in f.readlines()]
            
    def _get_output_layers(self):
        layer_names = self.net.getLayerNames()
        return [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
    def _init_visualization(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-', label="Crowd Count")
        self.ax.set_xlim(0, 50)
        self.ax.set_ylim(0, 20)
        self.ax.set_xlabel("Frame Index")
        self.ax.set_ylabel("Crowd Count")
        self.ax.legend()
        
    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            height, width = frame.shape[:2]
            
            # Detect people
            boxes, confidences, class_ids = self._detect_people(frame)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            crowd_count = len(indexes)
            
            # Update data
            self._update_data(frame_idx, crowd_count)
            self._visualize(frame, boxes, indexes, class_ids, confidences, crowd_count)
            
            frame_idx += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        self.conn.close()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()
        
    def _detect_people(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        height, width = frame.shape[:2]
        boxes = []
        confidences = []
        class_ids = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and self.classes[class_id] == "person":
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    boxes.append([center_x - w//2, center_y - h//2, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    
        return boxes, confidences, class_ids
        
    def _update_data(self, frame_idx, count):
        self.crowd_counts.append(count)
        self.frame_numbers.append(frame_idx)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            "INSERT INTO CrowdCount (frame_index, timestamp, count) VALUES (?, ?, ?)",
            (frame_idx, timestamp, count)
        )
        self.conn.commit()
        
    def _visualize(self, frame, boxes, indexes, class_ids, confidences, count):
        # Update graph
        self.line.set_xdata(self.frame_numbers)
        self.line.set_ydata(self.crowd_counts)
        self.ax.set_xlim(max(0, len(self.frame_numbers)-50), len(self.frame_numbers))
        self.ax.set_ylim(0, max(10, max(self.crowd_counts) + 2))
        plt.draw()
        plt.pause(0.01)
        
        # Draw bounding boxes
        for i in indexes:
            x, y, w, h = boxes[i]
            label = f"{self.classes[class_ids[i]]} {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Display count
        cv2.putText(frame, f"Crowd Count: {count}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Crowd Detection", frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time crowd counting using YOLOv3")
    parser.add_argument("--video", type=str, default="0",
                       help="Video source (0 for webcam or path to video file)")
    parser.add_argument("--weights", type=str, default="yolov3.weights",
                       help="Path to YOLOv3 weights file")
    parser.add_argument("--config", type=str, default="yolov3.cfg",
                       help="Path to YOLOv3 config file")
    parser.add_argument("--names", type=str, default="coco.names",
                       help="Path to COCO names file")
    
    args = parser.parse_args()
    
    try:
        video_source = int(args.video) if args.video.isdigit() else args.video
        counter = CrowdCounter(video_source, args.weights, args.config, args.names)
        counter.run()
    except Exception as e:
        print(f"Error: {e}")
