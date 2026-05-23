# main.py

import cv2
import sys
import yaml
import time
import os
from src.detectors.yolo_pose import YoloPoseDetector
from src.utils.visualizer import PoseVisualizer
from src.utils.stream_factory import init_video_stream
from src.trackers.bot_sort import PKLTracker
from src.trackers.detection import Detection

def load_config(config_path="config/default_config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Lỗi đọc cấu hình: {e}")
        sys.exit(1)

def run_pipeline():
    config = load_config()
    model_path = config.get("model_path", "yolo26n-pose.pt")
    video_source = config.get("video_source", 0)
    draw_links = config.get("draw_skeleton_links")
    draw_dots = config.get("draw_keypoint_dots")
    
    # ĐỌC CỜ BẬT/TẮT LƯU VIDEO TỪ CONFIG
    save_video = config.get("save_debug_video", True)

    detector = YoloPoseDetector(model_path=model_path)
    visualizer = PoseVisualizer()
    pkl_tracker = PKLTracker(config=config)
    
    cap, is_live_stream, frame_delay = init_video_stream(video_source)
        
    print("🚀 Hệ thống Giám sát Thích ứng đang vận hành trơn tru...")
    print("👉 Bấm phím 'q' tại cửa sổ hiển thị để đóng chương trình.")

    # 1. KIỂM TRA TRÍCH XUẤT THÔNG SỐ BAN ĐẦU CHO VIDEO WRITER (NẾU BẬT CỜ)
    video_writer = None
    if save_video:
        if is_live_stream:
            video_fps = getattr(cap, 'fps', 30.0)
            probe_frame = cap.read()
            while probe_frame is None:
                time.sleep(0.01)
                probe_frame = cap.read()
        else:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps <= 0: video_fps = 30.0
            _, probe_frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Đưa video về khung hình đầu tiên

        frame_h, frame_w = probe_frame.shape[:2]
        
        # KHỞI TẠO BỘ GHI XUẤT FILE VIDEO CỤC BỘ
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "debug_reid_output.mp4")
        
        # 🌟 SỬA TẠI ĐÂY: Thay đổi Codec sang 'avc1' (Chuẩn nén H.264 quốc dân)
        # Giúp video mở mượt mà trên macOS, iOS, Windows Player và Web Browser
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        
        # Phương án dự phòng (Fallback): Nếu máy bạn thiếu bộ mã hóa avc1 gốc của OpenCV
        # Ta chuyển sang 'X264' (Yêu cầu hệ thống đã cài thư viện openh264)
        # fourcc = cv2.VideoWriter_fourcc(*'X264')
        
        video_writer = cv2.VideoWriter(output_path, fourcc, video_fps, (frame_w, frame_h))
        print(f"💾 [Video Writer]: Kích hoạt lưu luồng H.264 đa nền tảng tại: {output_path} ({frame_w}x{frame_h} @ {video_fps:.1f} FPS)")

    # KHỞI TẠO CỬA SỔ HIỂN THỊ THÔNG MINH CHỐNG MÉO VIDEO
    win_name = "Industrial Human Pose Stream"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(win_name, 960, 540)

    while True:
        start_time = time.time()
        frame = None
        
        # ĐỌC KHUNG HÌNH THÍCH ỨNG
        if is_live_stream:
            if cap.stopped: break
            frame = cap.read()
            if frame is None: continue
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("🎬 Đã xử lý hết toàn bộ khung hình của file video.")
                break
            
        raw_data = detector.get_detections(frame)
        
        detections = []
        if raw_data is not None:
            for obj in raw_data:
                bbox = obj.get('bbox', obj.get('box'))
                conf = obj.get('confidence', obj.get('score', 0.0))
                kpts = obj.get('keypoints', obj.get('kpts'))
                
                if bbox is not None and kpts is not None:
                    detections.append(Detection(bbox=bbox, confidence=conf, keypoints=kpts))
            
        online_tracks = pkl_tracker.update(detections)
        rendered_frame = visualizer.draw_pose(
            frame,
            online_tracks,
            draw_links=draw_links,
            draw_dots=draw_dots
        )
        
        # 2. GHI KHUNG HÌNH ĐÃ VẼ VÀO VIDEO WRITER
        if video_writer is not None:
            video_writer.write(rendered_frame)
        
        # HIỂN THỊ VÀO CỬA SỔ THÔNG MINH ĐÃ KHỞI TẠO
        cv2.imshow(win_name, rendered_frame)
        
        # ĐIỀU KHIỂN ĐỘ TRỄ HIỂN THỊ
        if is_live_stream:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.stop()
                break
        else:
            elapsed_time = (time.time() - start_time) * 1000
            dynamic_delay = max(1, int(frame_delay - elapsed_time))
            if cv2.waitKey(dynamic_delay) & 0xFF == ord('q'):
                break
                
    # 3. GIẢI PHÓNG TÀI NGUYÊN BỘ GHI VIDEO AN TOÀN
    if video_writer is not None:
        video_writer.release()
        print(f"💾 Đã đóng gói và xuất file video kết quả thành công tại thư mục 'output/'.")

    if is_live_stream: cap.stop()
    else: cap.release()
        
    cv2.destroyAllWindows()
    print("👋 Hệ thống đóng an toàn.")

if __name__ == "__main__":
    run_pipeline()
