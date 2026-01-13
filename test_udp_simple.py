#!/usr/bin/env python3
"""Simple UDP test without OpenCV"""
import socket
import json
import time

addr = ("127.0.0.1", 5005)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("Sending UDP packets to 127.0.0.1:5005")
print("Start the receiver in another terminal")
print("Press Ctrl+C to stop\n")

frame_id = 0
try:
    while True:
        msg = {
            "timestamp": time.time(),
            "frame_id": frame_id,
            "objects": [{"label": "test", "score": 0.99, "bbox": [0.1, 0.2, 0.3, 0.4]}],
        }
        sock.sendto(json.dumps(msg).encode("utf-8"), addr)
        
        if frame_id % 30 == 0:
            print(f"Sent frame {frame_id}")
        
        frame_id += 1
        time.sleep(1/30)  # 30 FPS
        
except KeyboardInterrupt:
    print(f"\nSent {frame_id} packets total")
