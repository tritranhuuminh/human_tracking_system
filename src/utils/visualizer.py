# src/utils/visualizer.py

import cv2
import numpy as np

class PoseVisualizer:
    """
    Class chuyên dụng chịu trách nhiệm đo tốc độ xử lý (FPS), vẽ đồ họa, 
    hộp bao và kết nối 17 khớp xương (Skeleton) của con người lên camera.
    """
    def __init__(self):
        # 1. Định nghĩa sơ đồ kết nối 17 điểm keypoints chuẩn COCO (Xương người)
        # Mỗi cặp số đại diện cho 2 điểm khớp nối với nhau (ví dụ: 5-7 là vai trái nối khuỷu tay trái)
        self.skeleton_connections = [
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (1, 2), (0, 1), (0, 2),
            (1, 3), (2, 4), (3, 5), (4, 6)
        ]
        
        # 2. Định nghĩa bảng màu (Bgr) cho các đường xương để tạo hiệu ứng trực quan sinh động
        self.link_colors = [
            (0, 255, 255), (0, 255, 255), (255, 0, 255), (255, 0, 255), (0, 255, 0),
            (255, 128, 0), (255, 128, 0), (0, 255, 0), (0, 128, 255), (0, 128, 255),
            (0, 0, 255), (0, 0, 255), (255, 255, 0), (255, 255, 0), (255, 255, 0),
            (255, 255, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0)
        ]

        # 3. Định nghĩa màu sắc cho 17 chấm điểm khớp (Keypoint Dots)
        self.kpt_colors = [
            (0, 0, 255), (0, 128, 255), (0, 128, 255), (0, 255, 255), (0, 255, 255),
            (0, 255, 0), (0, 255, 0), (255, 128, 0), (255, 128, 0), (255, 0, 255),
            (255, 0, 255), (128, 0, 255), (128, 0, 255), (255, 0, 128), (255, 0, 128),
            (0, 128, 128), (0, 128, 128)
        ]
        
        # 4. Khởi tạo mốc thời gian hệ thống để tính toán giá trị FPS chuẩn xác nhất
        self.prev_time = cv2.getTickCount()

    def draw_fps(self, frame):
        """
        Hàm tính toán tần suất lấy mẫu khung hình và vẽ chỉ số FPS lên góc màn hình.
        """
        # Lấy mốc xung nhịp thời gian hiện tại của hệ thống
        current_time = cv2.getTickCount()
        
        # Tính toán khoảng thời gian chênh lệch thực tế (tính bằng giây)
        time_diff = (current_time - self.prev_time) / cv2.getTickFrequency()
        
        # Cập nhật lại mốc thời gian cũ làm bàn đạp cho khung hình tiếp theo
        self.prev_time = current_time
        
        # Tính toán FPS (Đặt điều kiện bảo vệ tránh lỗi chia cho 0 nếu luồng video bị nghẽn)
        fps = 1.0 / time_diff if time_diff > 0 else 0.0
        
        # Vẽ một hộp nền đen mờ bo góc nhẹ ở góc trên bên trái để làm nổi bật chữ FPS
        cv2.rectangle(frame, (10, 10), (145, 45), (0, 0, 0), -1)
        
        # Định dạng chuỗi văn bản hiển thị FPS rút gọn lấy 1 chữ số thập phân
        fps_text = f"FPS: {fps:.1f}"
        
        # Vẽ chữ chỉ số FPS màu xanh neon sắc nét lên khung nền đen mờ
        cv2.putText(frame, fps_text, (20, 34), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        
        return frame

    def draw_pose(self, frame, detections):
        """
        Nhận vào khung hình gốc và danh sách kết quả thô từ YOLO Detector,
        tiến hành vẽ đè đồ họa lên ảnh và trả về khung hình đã được xử lý (Rendered Frame).
        """
        # Tạo một bản sao để tránh ghi đè làm hỏng mảng ảnh gốc nếu cần luồng xử lý song song
        out_frame = frame.copy()

        for det in detections:
            bbox = det['bbox'].astype(int)
            kpts = det['keypoints']
            conf = det['confidence']
            
            # --- BƯỚC 1: VẼ HỘP BAO (BOUNDING BOX) VÀ THÔNG TIN ---
            x1, y1, x2, y2 = bbox
            # Vẽ hộp chữ nhật màu xanh lá bao quanh công nhân
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Hiển thị nhãn text ghi thông số độ tin cậy thô của AI
            label = f"Worker: {conf:.2f}"
            cv2.putText(out_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, lineType=cv2.LINE_AA)

            # Nếu không trích xuất được khớp xương (mảng trống), bỏ qua bước vẽ xương
            if len(kpts) == 0:
                continue

            # --- BƯỚC 2: VẼ CÁC ĐƯỜNG NỐI XƯƠNG (SKELETON LINKS) ---
            for link, color in zip(self.skeleton_connections, self.link_colors):
                pt1_idx, pt2_idx = link
                kpt1 = kpts[pt1_idx]
                kpt2 = kpts[pt2_idx]

                # Chỉ vẽ đường nối nếu cả 2 đầu khớp đều được AI nhìn thấy rõ (Ngưỡng tin cậy > 0.4)
                if kpt1[2] > 0.4 and kpt2[2] > 0.4:
                    pos1 = (int(kpt1[0]), int(kpt1[1]))
                    pos2 = (int(kpt2[0]), int(kpt2[1]))
                    cv2.line(out_frame, pos1, pos2, color, thickness=2, lineType=cv2.LINE_AA)

            # --- BƯỚC 3: VẼ CÁC CHẤM TRÒN KHỚP NỐI (KEYPOINT DOTS) ---
            for idx, kpt in enumerate(kpts):
                kx, ky, kconf = kpt
                # Chỉ chấm điểm tròn nếu khớp xương có độ tin cậy cao
                if kconf > 0.4:
                    color = self.kpt_colors[idx]
                    cv2.circle(out_frame, (int(kx), int(ky)), 4, color, -1, lineType=cv2.LINE_AA)

        # --- BƯỚC 4: GỌI TỰ ĐỘNG CHÈN CHỈ SỐ FPS LÊN KHUNG HÌNH CUỐI ---
        out_frame = self.draw_fps(out_frame)

        return out_frame
