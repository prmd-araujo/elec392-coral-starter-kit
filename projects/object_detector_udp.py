
"""
Real-time object detection using Coral TPU, sending results over UDP.

This script detects objects in video frames from the camera and broadcasts
the detection results (bounding boxes and confidence scores) over UDP. This
allows multiple processes to access the same detection results via local
inter-process communication (IPC).

Useful for obstacle detection and sign detection in autonomous systems.

CUSTOM MODELS:
By default, this uses the pre-built SSD MobileNet v2 COCO model. However, you
can train and deploy your own object detection models for your specific use case
(e.g., detecting road signs, ducks, obstacles). See the tutorials in the
tutorials/ folder for training custom models with EfficientDet-Lite.

To use your custom model:
    1. Train your model using the provided tutorials
    2. Convert/compile it for the Coral TPU edge device
    3. Replace the MODEL_PATH below with your model file path
    4. Update the labels dictionary accordingly

Usage:
    python projects/object_detector_udp.py [--confidence THRESHOLD] [--model MODEL_PATH] [--labels LABELS_FILE]

Args:
    --confidence: Minimum confidence score for detections (0.0-1.0, default: 0.5)
    --model: Path to custom model file (default: built-in SSD MobileNet v2)
    --labels: Path to labels file for custom model (txt file, one label per line)

The default TensorFlow model is downloaded if you flashed the AIY Maker Kit
system image for Raspberry Pi. Otherwise, run download_models.sh in this directory.
"""

import os.path
import sys
import argparse
import time

# Add parent directory to path so we can import ipc module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiymakerkit import vision
from aiymakerkit import utils
from ipc.udp import UdpDetectionSender


def path(name):
    """Creates an absolute path to a file in the same directory as this script."""
    root = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(root, name)


def format_detections(objects, frame_width, frame_height):
    """
    Convert detected objects to normalized detection format for UDP transmission.
    
    Bounding boxes are normalized to [0,1] range relative to frame size.
    
    Args:
        objects: List of detected objects from the model
        frame_width: Width of the video frame in pixels
        frame_height: Height of the video frame in pixels
    
    Returns:
        List of detection dictionaries with normalized bounding boxes
    """
    detections = []
    
    for obj in objects:
        bbox = obj.bbox
        
        # Normalize coordinates to [0, 1]
        xmin = max(0.0, bbox.xmin / frame_width)
        ymin = max(0.0, bbox.ymin / frame_height)
        xmax = min(1.0, bbox.xmax / frame_width)
        ymax = min(1.0, bbox.ymax / frame_height)
        
        detection = {
            "label": labels.get(obj.id, "unknown"),
            "score": round(obj.score, 2),
            "bbox": [xmin, ymin, xmax, ymax],
        }
        detections.append(detection)
    
    return detections


def main():
    parser = argparse.ArgumentParser(
        description="Detect objects and broadcast detections over UDP"
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
            global labels
            labels = utils.read_label_file(args.labels)
        else:
            print("Warning: No labels file provided for custom model. Using generic labels.")
    else:
        # Use default model from AIY Maker Kit
        model_path = path('ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
        print("Using default SSD MobileNet v2 COCO model")
    
    # Load the TensorFlow Lite model (compiled for the Edge TPU)
    detector = vision.Detector(model_path)
    
    # Get video dimensions
    width, height = vision.VIDEO_SIZE
    
    # Initialize UDP sender
    sender = UdpDetectionSender()
    
    print(f"Starting object detection with Coral TPU")
    print(f"Confidence threshold: {args.confidence}")
    print(f"Broadcasting detections to 127.0.0.1:5005")
    print(f"Press Ctrl+C to stop\n")
    
    frame_id = 0
    last_print_time = time.time()
    last_debug_time = time.time()
    frame_count = 0
    udp_send_count = 0
    
    try:
        # Run without display (headless mode for UDP transmission)
        # Disable mirror to avoid any OpenCV operations that might interact with display
        for frame in vision.get_frames(display=False, mirror=False):
            frame_count += 1
            
            # Debug: print frame rate every 2 seconds
            current_time = time.time()
            if current_time - last_debug_time >= 2.0:
                fps = frame_count / (current_time - last_debug_time)
                print(f"[DEBUG] Processing at {fps:.1f} FPS, frame_id={frame_id}, UDP sends={udp_send_count}")
                frame_count = 0
                last_debug_time = current_time
            
            try:
                start_time = time.time()
                
                # Detect objects with specified confidence threshold
                objects = detector.get_objects(frame, threshold=args.confidence)
                detect_time = time.time() - start_time
                
                # Format detections for UDP transmission
                detections = format_detections(objects, width, height)
                
                # Send detections over UDP
                try:
                    sender.send(detections, frame_id)
                    udp_send_count += 1
                except Exception as udp_error:
                    print(f"[ERROR] UDP send failed: {udp_error}")
                
                # Print summary every 1 second
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    print(f"Frame {frame_id}: {len(detections)} objects | Detection: {detect_time*1000:.1f}ms")
                    if detections:
                        for det in detections:
                            bbox = det['bbox']
                            # Convert normalized bbox back to pixel coordinates for display
                            xmin = int(bbox[0] * width)
                            ymin = int(bbox[1] * height)
                            xmax = int(bbox[2] * width)
                            ymax = int(bbox[3] * height)
                            print(f"  - {det['label']}: {det['score']} | bbox: ({xmin}, {ymin}, {xmax}, {ymax})")
                    last_print_time = current_time
                
                frame_id += 1
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"Error processing frame {frame_id}: {e}")
                frame_id += 1
                continue
            
    except KeyboardInterrupt:
        print("\n\nDetection stopped.")


if __name__ == "__main__":
    # Load labels before main execution
    # NOTE: For custom models, you'll need to provide your own labels file
    # and load them here instead of using the default COCO labels
    OBJECT_DETECTION_MODEL = path('ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
    labels = utils.read_labels_from_metadata(OBJECT_DETECTION_MODEL)
    
    main()
