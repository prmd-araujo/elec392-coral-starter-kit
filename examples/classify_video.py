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
Performs continuous image classification with the camera video.

To classify things using a default MobileNet model, simply run the script:

    python3 classify_video.py

Or classify using your own model and labels file:

    python3 classify_video.py -m my_model.tflite

For information about the script options, run:

    python3 classify_video.py --help

For more instructions, see g.co/aiy/maker
"""

import os
import argparse
import time
import cv2
from pycoral.utils.dataset import read_label_file
from aiymakerkit import vision
from aiymakerkit.utils import read_labels_from_metadata
import models

# Preventing QT errors when running without a display
os.environ['QT_QPA_PLATFORM'] = 'xcb'


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default=models.CLASSIFICATION_MODEL,
                        help='File path of .tflite file. Default is models.CLASSIFICATION_MODEL')
    parser.add_argument('-l', '--labels', default=models.CLASSIFICATION_LABELS,
                        help='File path of labels file. Default is models.CLASSIFICATION_LABELS.')
    args = parser.parse_args()

    classifier = vision.Classifier(args.model)
    if args.labels is not None:
        labels = read_label_file(args.labels)
    else:
        labels = read_labels_from_metadata(args.model)

    print(f"Loaded model: {args.model}")
    print(f"Loaded {len(labels)} labels")
    print("Starting classification loop...\n")

    frame_count = 0
    total_processing_time = 0.0

    for frame in vision.get_frames(display=False):
        start_time = time.time()
        frame_count += 1
        
        classes = classifier.get_classes(frame, top_k=1, threshold=0.2)
        if classes:
            score = classes[0].score
            label = labels.get(classes[0].id)
            vision.draw_label(frame, f'{label}: {round(score, 4)}')
        
        # Show the frame after drawing
        cv2.imshow('Image Classification', frame)
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


if __name__ == '__main__':
    main()