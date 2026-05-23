# src/detectors/yolo_pose.py

import os
import torch
from ultralytics import YOLO

class YoloPoseDetector:
    """
    Class nạp mô hình YOLO-Pose, tự động cấu hình chạy trên phần cứng GPU CUDA,
    tự động tải trọng số từ internet nếu chưa có và trích xuất đầy đủ thông số 17 điểm keypoints.
    """
    def __init__(self, model_path="yolo26n-pose.pt", conf_thresh=0.5):
        """
        model_path: Tên hoặc đường dẫn tới file trọng số (.pt) của mô hình.
        conf_thresh: Ngưỡng tin cậy tối thiểu để chấp nhận một phát hiện.
        """
        self.conf_thresh = conf_thresh
        
        # 1. Tự động kiểm tra và cấu hình phần cứng (Ưu tiên CUDA của card RTX A1000)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚙️ [Khởi tạo] Đang cấu hình phần cứng chạy trên: {self.device.upper()}")
        
        # 2. Cơ chế kiểm tra và tự động tải mô hình nếu ổ đĩa cục bộ chưa có [1]
        dir_name = os.path.dirname(model_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            
        if not os.path.exists(model_path):
            print(f"📥 [Thông báo] Không tìm thấy file {model_path} cục bộ.")
            print("👉 Hệ thống đang tự động tải mô hình YOLO-Pose chính thức từ kho lưu trữ...")
            # Lấy tên file gốc (ví dụ: yolov11n-pose.pt) để kích hoạt tự động tải của Ultralytics [1]
            model_name = os.path.basename(model_path)
            self.model = YOLO(model_name)
            # Lưu lại vào đúng đường dẫn cấu hình để sử dụng ngoại tuyến (offline) cho các lần sau [1]
            self.model.save(model_path)
            print(f"💾 Đã tải và lưu trữ mô hình thành công tại: {model_path}")
        else:
            print(f"📦 Đã tìm thấy mô hình cục bộ tại: {model_path}")
            self.model = YOLO(model_path)
        
        # Đẩy mô hình lên thiết bị phần cứng đích (GPU CUDA)
        self.model.to(self.device)

    def get_detections(self, frame):
        """
        Hàm xử lý hình ảnh và trích xuất dữ liệu thô.
        Input: frame (Mảng ảnh BGR đọc từ OpenCV)
        Output: Danh sách chứa thông tin của từng người phát hiện được với đầy đủ thông số:
                [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'class_id': int, 'keypoints': [[x, y, conf], ...]}]
        """
        # Chạy mô hình dự đoán (verbose=False để tránh tràn log ra màn hình terminal)
        results = self.model(frame, device=self.device, verbose=False)
        
        formatted_detections = []
        
        # Kiểm tra nếu khung hình hoàn toàn trống hoặc không phát hiện được ai
        if results is None or len(results) == 0 or results[0].boxes is None:
            return formatted_detections

        # Lấy kết quả của khung hình hiện tại (YOLO xử lý theo batch, lấy phần tử đầu tiên)
        frame_results = results[0]
        
        # Chuyển dữ liệu Hộp bao, Độ tin cậy và ID lớp về bộ nhớ RAM (CPU) dưới dạng mảng numpy
        boxes = frame_results.boxes.xyxy.cpu().numpy()       # Tọa độ hộp [x1, y1, x2, y2]
        box_confs = frame_results.boxes.conf.cpu().numpy()   # Độ tin cậy của hộp bao (0.0 -> 1.0)
        class_ids = frame_results.boxes.cls.cpu().numpy()    # ID của lớp đối tượng (0 là người) [2]
        
        # Trích xuất mảng 17 điểm keypoints (Hình học: Số người x 17 điểm x 3 thông số [x, y, conf])
        if frame_results.keypoints is not None:
            keypoints_data = frame_results.keypoints.data.cpu().numpy()
        else:
            keypoints_data = []

        # Vòng lặp duyệt qua từng người phát hiện được trong ảnh
        for i in range(len(boxes)):
            # Lọc bỏ các phát hiện bị mờ, nhiễu hoặc độ chính xác dưới ngưỡng cấu hình
            if box_confs[i] < self.conf_thresh:
                continue
                
            # Đóng gói dữ liệu đầy đủ thông số chuẩn công nghiệp phục vụ cho Kalman Filter [2]
            det_item = {
                'bbox': boxes[i],
                'confidence': float(box_confs[i]),
                'class_id': int(class_ids[i]),
                'keypoints': keypoints_data[i] if len(keypoints_data) > i else []
            }
            formatted_detections.append(det_item)
            
        return formatted_detections
