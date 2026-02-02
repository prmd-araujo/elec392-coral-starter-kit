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
Performs continuous object detection with the camera.

Simply run the script and it will draw boxes around detected objects along
with the predicted labels:

    python3 detect_objects.py

For more instructions, see g.co/aiy/maker
"""

from aiymakerkit import vision
from aiymakerkit import utils
from pycoral.utils.dataset import read_label_file
import models
import os
import time

# Preventing QT errors when running without a display
os.environ['QT_QPA_PLATFORM'] = 'xcb'

detector = vision.Detector(models.OBJECT_DETECTION_MODEL)
labels = read_label_file(models.OBJECT_DETECTION_LABELS)

print(f"Loaded model: {models.OBJECT_DETECTION_MODEL}")
print(f"Loaded {len(labels)} labels")
print("Starting detection loop...\n")

import cv2

frame_count = 0
total_processing_time = 0.0

for frame in vision.get_frames(display=False):
    start_time = time.time()
    frame_count += 1
    objects = detector.get_objects(frame, threshold=0.2)
    
    # Show the frame ourselves after drawing
    vision.draw_objects(frame, objects, labels)
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Track processing time
    frame_time = time.time() - start_time
    total_processing_time += frame_time

cv2.destroyAllWindows()
print(f"\nProcessed {frame_count} frames total")
if frame_count > 0:
    avg_time = total_processing_time / frame_count
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"Average processing time per frame: {avg_time*1000:.2f} ms")
    print(f"Average FPS: {fps:.2f}")
