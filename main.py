# main.py

import cv2
import sys
import yaml
from src.detectors.yolo_pose import YoloPoseDetector
from src.utils.visualizer import PoseVisualizer
from src.utils.stream_manager import RTSPStreamManager  # NẠP ĐA LUỒNG

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
    conf_thresh = config.get("min_pose_conf", 0.5)
    video_source = config.get("video_source", 0)

    detector = YoloPoseDetector(model_path=model_path, conf_thresh=conf_thresh)
    visualizer = PoseVisualizer()
    
    # KÍCH HOẠT ĐỌC CAMERA ĐA LUỒNG CÔNG NGHIỆP
    cap = RTSPStreamManager(src=video_source).start()
    if not cap.is_opened():
        print(f"❌ Lỗi: Không thể mở nguồn video ({video_source}).")
        sys.exit(1)
        
    print("🚀 Hệ thống Giám sát Đa Luồng (Multi-Threaded) đang chạy...")
    print("👉 Bấm phím 'q' tại cửa sổ hiển thị để đóng chương trình.")

    # Vòng lặp AI không bị nghẽn bởi camera
    while not cap.stopped:
        frame = cap.read()
        if frame is None:
            continue
            
        # Bước A: Tốc độ xử lý của GPU CUDA lúc này sẽ đạt tối đa
        raw_data = detector.get_detections(frame)
        
        # Bước B: Vẽ đồ họa trực tiếp, không tốn tài nguyên copy vùng nhớ
        rendered_frame = visualizer.draw_pose(frame, raw_data)
        
        cv2.imshow("Industrial Human Pose Stream", rendered_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.stop()  # Dừng luồng phụ an toàn
            break
            
    cv2.destroyAllWindows()
    print("👋 Hệ thống đóng an toàn.")

if __name__ == "__main__":
    run_pipeline()
