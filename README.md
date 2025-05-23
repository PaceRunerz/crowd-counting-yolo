# Real-Time Crowd Counting with YOLOv3
A robust crowd counting system using YOLOv3 object detection with real-time visualization and data logging.

## Table of Contents
- [How YOLO Works](#how-yolo-works)
- [Why Use YOLO for Crowd Counting](#why-use-yolo-for-crowd-counting)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results Interpretation](#results-interpretation)
- [Performance Considerations](#performance-considerations)
- [Future Improvements](#future-improvements)

## How YOLO Works

YOLO (You Only Look Once) is a state-of-the-art object detection system that:
- Processes images in a single neural network evaluation (real-time speed)
- Divides the image into a grid and predicts bounding boxes/class probabilities
- Uses anchor boxes to predict multiple objects in the same grid cell
- Achieves high accuracy with efficient computation

For crowd counting, we specifically:
1. Process each video frame through the YOLOv3 network
2. Filter detections to only include "person" class
3. Apply Non-Maximum Suppression (NMS) to eliminate duplicate detections
4. Count remaining boxes as our crowd estimate

## Why Use YOLO for Crowd Counting

1. **Real-time Performance**: YOLO is significantly faster than R-CNN variants
2. **Good Accuracy**: Balances speed and detection quality effectively
3. **Single-Stage Detection**: Processes the entire image at once
4. **Generalization**: Works well on various crowd densities
5. **Open Implementation**: Easily customizable for specific needs

Traditional methods like background subtraction or Haar cascades struggle with:
- Occlusions in dense crowds
- Varying lighting conditions
- Different person orientations
- Scale variations

## Features

- � **Accurate Detection**: YOLOv3 with configurable confidence thresholds
- 📊 **Real-time Visualization**: Live video feed with bounding boxes and count display
- 📈 **Trend Analysis**: Dynamic plotting of crowd count over time
- 💾 **Data Logging**: Persistent storage in SQLite database
- ⚙️ **Configurable**: Adjustable parameters via command line
- 🖥️ **Multiple Sources**: Works with webcam or video files

 ## Before You Begin

This project requires YOLOv3 model files which are too large for GitHub. You'll need to download them separately:

### Required Files:
1. **yolov3.weights** (237 MB) - [Download](https://www.kaggle.com/datasets/shivam316/yolov3-weights)
2. **yolov3.cfg** - [Download](https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.cfg)
3. **coco.names** - [Download](https://github.com/pjreddie/darknet/blob/master/data/coco.names)


## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crowd-counting-yolo.git
cd crowd-counting-yolo
