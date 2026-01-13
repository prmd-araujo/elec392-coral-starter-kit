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
    python projects/object_detector.py [--confidence THRESHOLD] [--model MODEL_PATH] [--labels LABELS_FILE] [--headless] [--no-draw]

Args:
    --confidence: Minimum confidence score for detections (0.0-1.0, default: 0.5)
    --model: Path to custom model file (default: built-in SSD MobileNet v2)
    --labels: Path to labels file for custom model (txt file, one label per line)
    --headless: Run without display (useful for SSH/headless Pi). Only prints detections.
    --no-draw: Don't draw bounding boxes on video (useful for Raspberry Pi Connect)

The default TensorFlow model is downloaded if you flashed the AIY Maker Kit
system image for Raspberry Pi. Otherwise, run download_models.sh in this directory.
"""

import os.path
import argparse
import time
from aiymakerkit import vision
from aiymakerkit import utils


def path(name):
    """Creates an absolute path to a file in the same directory as this script."""
    root = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(root, name)


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
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display (useful for SSH/headless Pi). Only prints detections."
    )
    parser.add_argument(
        "--no-draw",
        action="store_true",
        help="Don't draw bounding boxes (useful for Raspberry Pi Connect). Only prints detections."
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
    if args.headless:
        print("Mode: HEADLESS (no display)")
    else:
        print("Mode: VISUAL (display enabled)")
    print(f"Press Ctrl+C to stop\n")
    
    frame_id = 0
    last_print_time = time.time()
    
    # Choose display mode based on flags
    display_mode = not args.headless
    
    print(f"Starting frame loop...")
    
    try:
        # Get frames with or without display
        for frame in vision.get_frames(display=display_mode):
            try:
                start_time = time.time()
                
                # Detect objects with specified confidence threshold
                objects = detector.get_objects(frame, threshold=args.confidence)
                detect_time = time.time() - start_time
                
                # Draw detections only if display is on and --no-draw is not set
                if display_mode and not args.no_draw:
                    vision.draw_objects(frame, objects, labels=labels, color=(0, 255, 0), thickness=2)
                
                # Print summary every 1 second
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    print(f"Frame {frame_id}: {len(objects)} objects | Detection: {detect_time*1000:.1f}ms")
                    if objects:
                        for obj in objects:
                            label = labels.get(obj.id, "unknown")
                            print(f"  - {label}: {obj.score:.2f}")
                    last_print_time = current_time
                
                frame_id += 1
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # Skip frames with errors and continue
                print(f"Error processing frame {frame_id}: {e}")
                frame_id += 1
                continue
        
    except KeyboardInterrupt:
        print("\n\nDetection stopped.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
