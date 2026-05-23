# src/utils/stream_factory.py

import cv2
import sys
from src.utils.stream_manager import RTSPStreamManager  # Chỉ dùng cho Camera/RTSP

def init_video_stream(video_source):
    """
    HÀM NHẬN DIỆN NGUỒN VIDEO: Tự động phân tích nguồn vào để kích hoạt 
    Đa luồng công nghiệp (Camera Live) hoặc Đọc tuần tự đồng bộ (File Video).
    """
    is_live_stream = True
    
    # Logic nhận diện: Nếu nguồn là chuỗi văn bản và KHÔNG bắt đầu bằng các giao thức mạng rtsp/http
    if isinstance(video_source, str) and not (video_source.lower().startswith("rtsp://") or video_source.lower().startswith("http://")):
        is_live_stream = False  # Xác định đây là File Video cục bộ (.mp4, .avi, .mkv...)

    cap = None
    frame_delay = 1

    if is_live_stream:
        print(f"📡 [Nhận diện nguồn]: CAMERA LIVE / RTSP STREAM. Kích hoạt Đa Luồng Công Nghiệp...")
        cap = RTSPStreamManager(src=video_source).start()
    else:
        print(f"🎬 [Nhận diện nguồn]: FILE VIDEO CỤC BỘ. Kích hoạt Đọc Tuần Tự Đồng Bộ 100% khung hình...")
        cap = cv2.VideoCapture(video_source)
        # Tính toán thời gian trễ tiêu chuẩn dựa theo FPS gốc của file video
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0
        frame_delay = max(1, int(1000 / video_fps))

    # Kiểm tra điều kiện mở nguồn an toàn thích ứng theo class của cap
    #  Đã đổi thành .isOpened() chuẩn cú pháp OpenCV
    is_opened = cap.is_opened() if hasattr(cap, 'is_opened') else cap.isOpened()

    if not is_opened:
        print(f"❌ Lỗi: Không thể mở kết nối đến nguồn video ({video_source}).")
        sys.exit(1)

    return cap, is_live_stream, frame_delay
