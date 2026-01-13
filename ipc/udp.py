import json
import socket
import time

DEFAULT_ADDR = ("127.0.0.1", 5005)

class UdpDetectionSender:
    def __init__(self, addr=DEFAULT_ADDR):
        self.addr = addr
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Keep socket blocking for reliable localhost delivery
        # Reduce send buffer to prevent excessive kernel buffering
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        self.dropped_count = 0

    def send(self, objects, frame_id):
        msg = {
            "timestamp": time.time(),
            "frame_id": frame_id,
            "objects": objects,
        }
        try:
            self.sock.sendto(json.dumps(msg).encode("utf-8"), self.addr)
        except Exception as e:
            self.dropped_count += 1
            if self.dropped_count % 100 == 1:
                print(f"[WARNING] UDP send error (count={self.dropped_count}): {e}")
