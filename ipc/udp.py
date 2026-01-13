import json
import socket
import time
import sys

DEFAULT_ADDR = ("127.0.0.1", 5005)

class UdpDetectionSender:
    def __init__(self, addr=DEFAULT_ADDR):
        self.addr = addr
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Force immediate delivery - disable Nagle's algorithm equivalent for UDP
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_TTL, 1)
        # Bind to specific source to ensure routing
        self.sock.bind(('127.0.0.1', 0))
        self.dropped_count = 0

    def send(self, objects, frame_id):
        msg = {
            "timestamp": time.time(),
            "frame_id": frame_id,
            "objects": objects,
        }
        try:
            self.sock.sendto(json.dumps(msg).encode("utf-8"), self.addr)
            sys.stdout.flush()  # Force flush to ensure no buffering blocks socket
        except Exception as e:
            self.dropped_count += 1
            if self.dropped_count % 100 == 1:
                print(f"[WARNING] UDP send error (count={self.dropped_count}): {e}")
                sys.stdout.flush()
