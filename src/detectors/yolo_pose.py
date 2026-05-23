# src/detectors/yolo_pose.py

import os
import yaml
import torch
import numpy as np
from ultralytics import YOLO

class YoloPoseDetector:
    """
    Class nạp mô hình YOLO-Pose tích hợp Bộ lọc ngưỡng tin cậy thích ứng động (Adaptive Confidence Gating).
    Tự động rẽ nhánh chạy BoT-SORT hộp bao phẳng hoặc Pose nâng cao dựa trên file cấu hình hệ thống.
    """
    def __init__(self, model_path="yolo26n-pose.pt", imgsz=1280, config_path="config/default_config.yaml"):
        """
        imgsz: Kích thước xử lý ảnh đầu vào diện rộng.
        """
        self.imgsz = imgsz 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️ [Khởi tạo] Đang cấu hình phần cứng chạy trên: {self.device.upper()}")
        print(f"🔭 [Detector Adaptive]: Kích hoạt bộ lọc ngưỡng tin cậy phi tuyến Sigmoid...")
        
        # 🌟 ĐỌC ẨN CỜ CẤU HÌNH TỪ FILE YAML ĐỂ TỰ ĐỘNG RẼ NHÁNH MÀ KHÔNG THAY ĐỔI HÀM MAIN
        self.enable_pose = True
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                    if cfg is not None:
                        self.enable_pose = cfg.get("enable_pose_estimation", True)
        except Exception as e:
            print(f"⚠️ Không thể đọc cờ enable_pose từ cấu hình, mặc định đặt True: {e}")
            
        print(f"⚙️ [Detector Status]: Chế độ trích xuất đang bật: {'YOLO-Pose Nâng cao' if self.enable_pose else 'YOLO-Object (BoT-SORT thường)'}")

        # 2. Cơ chế kiểm tra và nạp mô hình
        dir_name = os.path.dirname(model_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            
        if not os.path.exists(model_path):
            print(f"📥 [Thông báo] Không tìm thấy file {model_path} cục bộ.")
            self.model = YOLO(os.path.basename(model_path))
            self.model.save(model_path)
        else:
            self.model = YOLO(model_path)
        
        self.model.to(self.device)

    def get_detections(self, frame):
        """
        Hàm xử lý hình ảnh và áp dụng bộ lọc ngưỡng thích ứng thực tế theo từng đối tượng.
        """
        results = self.model(
            frame, 
            imgsz=self.imgsz, 
            device=self.device, 
            verbose=False
        )
        
        formatted_detections = []
        
        if results is None or len(results) == 0 or results[0].boxes is None:
            return formatted_detections

        frame_results = results[0]
        
        boxes = frame_results.boxes.xyxy.cpu().numpy()       
        box_confs = frame_results.boxes.conf.cpu().numpy()   
        class_ids = frame_results.boxes.cls.cpu().numpy()    
        
        # Chỉ lấy dữ liệu keypoints từ mô hình nếu cờ cấu hình cho phép chạy luồng Pose
        if self.enable_pose and frame_results.keypoints is not None:
            keypoints_data = frame_results.keypoints.data.cpu().numpy()
        else:
            keypoints_data = []

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            # CÔNG THỨC SIGMOID THÍCH ỨNG ĐỘNG
            k_slope = 0.0002 
            adaptive_conf_gate = 0.15 + (0.45 - 0.15) / (1.0 + np.exp(-k_slope * (area - 10000.0)))
            
            if box_confs[i] < adaptive_conf_gate:
                continue
                
            # 🌟 RẼ NHÁNH ĐÓNG GÓI DỮ LIỆU ĐỒNG BỘ:
            # Nếu tắt luồng Pose, tự sinh mảng 2 chiều rỗng chuẩn kích thước (17, 3) có độ tin cậy bằng 0.
            # Màng lọc Hungarian phẳng của BoT-SORT gốc nhận dạng mảng này và tự động khóa tính năng tính tâm xương
            if not self.enable_pose:
                final_keypoints = np.zeros((17, 3), dtype=np.float32)
            else:
                final_keypoints = keypoints_data[i] if len(keypoints_data) > i else np.zeros((17, 3), dtype=np.float32)
                
            det_item = {
                'bbox': boxes[i],
                'confidence': float(box_confs[i]),
                'class_id': int(class_ids[i]),
                'keypoints': final_keypoints
            }
            formatted_detections.append(det_item)
            
        return formatted_detections
