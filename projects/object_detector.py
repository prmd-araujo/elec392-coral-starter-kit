# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Real-time object detection visualization using Coral TPU.

This script detects objects in video frames from the camera and displays them
with bounding boxes and confidence scores overlaid on the camera feed. No data
is sent over the network - this is purely for visual inspection and debugging.

Useful for testing object detection models, tuning confidence thresholds, and
verifying model performance before deploying in autonomous systems.

CUSTOM MODELS:
By default, this uses the pre-built SSD MobileNet v2 COCO model. However, you
can train and deploy your own object detection models for your specific use case
(e.g., detecting road signs, ducks, obstacles). See the tutorials in the
tutorials/ folder for training custom models with EfficientDet-Lite.

To use your custom model:
    1. Train your model using the provided tutorials
    2. Convert/compile it for the Coral TPU edge device
    3. Pass the model file path using the --model argument
    4. Provide your labels file with the --labels argument

Usage:
    python projects/object_detector_visual.py [--confidence THRESHOLD] [--model MODEL_PATH] [--labels LABELS_FILE]

Args:
    --confidence: Minimum confidence score for detections (0.0-1.0, default: 0.5)
    --model: Path to custom model file (default: built-in SSD MobileNet v2)
    --labels: Path to labels file for custom model (txt file, one label per line)

The default TensorFlow model is downloaded if you flashed the AIY Maker Kit
system image for Raspberry Pi. Otherwise, run download_models.sh in this directory.
"""

import os.path
import argparse
import cv2
from aiymakerkit import vision
from aiymakerkit import utils


def path(name):
    """Creates an absolute path to a file in the same directory as this script."""
    root = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(root, name)


def draw_detections(frame, objects, labels):
    """
    Draw bounding boxes and labels on the frame using aiymakerkit utilities.
    
    Args:
        frame: Image frame to draw on
        objects: List of detected objects from the model
        labels: Dictionary mapping class IDs to label names
    """
    for obj in objects:
        # Draw bounding box in green
        vision.draw_rect(frame, obj.bbox, color=(0, 255, 0), thickness=2)
        
        # Draw label and confidence score
        label = labels.get(obj.id, "unknown")
        confidence = obj.score
        label_text = f"{label}: {confidence:.2f}"
        
        # Get text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
        
        # Draw background rectangle for text
        text_x = int(obj.bbox.xmin)
        text_y = int(obj.bbox.ymin) - 5
        cv2.rectangle(
            frame,
            (text_x, text_y - text_size[1] - 4),
            (text_x + text_size[0], text_y),
            (0, 255, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            label_text,
            (text_x, text_y - 2),
            font,
            font_scale,
            (0, 0, 0),  # Black text
            font_thickness
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize object detections with bounding boxes"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence score for detections (default: 0.5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to custom object detection model (.tflite). If not provided, uses default SSD MobileNet v2."
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Path to labels file for custom model (txt file, one label per line)."
    )
    args = parser.parse_args()
    
    # Determine which model to use
    if args.model:
        # Use custom model provided by student
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            return
        model_path = args.model
        print(f"Using custom model: {args.model}")
        
        # Load custom labels if provided
        if args.labels:
            if not os.path.exists(args.labels):
                print(f"Error: Labels file not found: {args.labels}")
                return
            print(f"Using custom labels: {args.labels}")
            labels = utils.read_label_file(args.labels)
        else:
            print("Warning: No labels file provided for custom model. Using generic labels.")
            labels = {}
    else:
        # Use default model from AIY Maker Kit
        model_path = path('ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
        print("Using default SSD MobileNet v2 COCO model")
        labels = utils.read_labels_from_metadata(model_path)
    
    # Load the TensorFlow Lite model (compiled for the Edge TPU)
    detector = vision.Detector(model_path)
    
    print(f"Starting object detection with Coral TPU")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Press Ctrl+C to stop\n")
    
    frame_id = 0
    
    try:
        # Run a loop to get images and process them in real-time
        for frame in vision.get_frames():
            # Detect objects with specified confidence threshold
            objects = detector.get_objects(frame, threshold=args.confidence)
            
            # Draw detections on the frame
            draw_detections(frame, objects, labels)
            
            # Print summary every 30 frames
            if frame_id % 30 == 0:
                print(f"Frame {frame_id}: {len(objects)} objects detected")
                for obj in objects:
                    label = labels.get(obj.id, "unknown")
                    print(f"  - {label}: {obj.score:.2f}")
            
            frame_id += 1
            
    except KeyboardInterrupt:
        print("\n\nVisualization stopped.")


if __name__ == "__main__":
    main()
