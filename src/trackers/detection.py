# src/trackers/detection.py
import numpy as np

class Detection:
    """Đối tượng bọc cấu trúc dữ liệu thô đầu vào từ mô hình YOLO-Pose."""
    def __init__(self, bbox, confidence, keypoints):
        self.tlbr = np.asarray(bbox, dtype=np.float32)
        self.confidence = float(confidence)
        self.keypoints = np.asarray(keypoints, dtype=np.float32)
        
        # TỰ ĐỘNG TÍNH TOÁN NGƯỠNG ADAPTIVE DYNAMIC THEO DIỆN TÍCH & CONFIDENCE
        w = self.tlbr[2] - self.tlbr[0]
        h = self.tlbr[3] - self.tlbr[1]
        area = w * h
        scale_factor = min(1.0, area / 40000.0) 
        
        # 1. Thích ứng ngưỡng Giai đoạn 1 (Kpt-IoU)
        t1 = 0.15 + (1.0 - self.confidence) * 0.2
        t2 = 0.40 + (1.0 - scale_factor) * 0.2
        t3 = 0.80 + (1.0 - self.confidence) * 0.15
        self.dynamic_kpt_iou_thresholds = [min(0.35, t1), min(0.65, t2), min(0.98, t3)]
        
        # 2. Thích ứng ngưỡng Giai đoạn 2 (Norm-L2)
        l2_t1 = 0.50 + (1.0 - self.confidence) * 0.25
        l2_t2 = 0.85 + (1.0 - scale_factor) * 0.35
        self.dynamic_norm_l2_thresholds = [min(0.75, l2_t1), min(1.25, l2_t2)]
        
        # 3. Thích ứng ngưỡng min_pose_conf tối thiểu
        self.dynamic_min_pose_conf = max(0.25, 0.5 - 0.2 * (1.0 - scale_factor))
