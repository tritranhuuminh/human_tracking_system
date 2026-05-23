# src/utils/stream_manager.py

import cv2
from threading import Thread
import time

class RTSPStreamManager:
    """
    Bộ quản lý luồng camera chạy đa luồng (Multi-threading).
    Tách biệt luồng đọc ảnh và luồng xử lý AI để triệt tiêu độ trễ của camera.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        # Khởi chạy một luồng phụ độc lập để đọc dữ liệu từ camera
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # Vòng lặp chạy ngầm liên tục cướp khung hình mới nhất
        while True:
            if self.stopped:
                self.stream.release()
                return
            self.grabbed, self.frame = self.stream.read()
            if not self.grabbed:
                self.stopped = True

    def read(self):
        # Trả về khung hình mới nhất có sẵn trong bộ nhớ đệm
        return self.frame

    def is_opened(self):
        return self.stream.isOpened() and self.grabbed

    def stop(self):
        self.stopped = True
