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
        self.skeleton_connections = [
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (1, 2), (0, 1), (0, 2),
            (1, 3), (2, 4), (3, 5), (4, 6)
        ]
        
        # 2. Định nghĩa bảng màu (Bgr) cho các đường xương
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
        """Hàm tính toán tần suất lấy mẫu khung hình và vẽ chỉ số FPS lên góc màn hình."""
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.prev_time) / cv2.getTickFrequency()
        self.prev_time = current_time
        
        fps = 1.0 / time_diff if time_diff > 0 else 0.0
        
        cv2.rectangle(frame, (10, 10), (145, 45), (0, 0, 0), -1)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (20, 34), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        
        return frame

    def draw_pose(self, frame, online_tracks, draw_links=True, draw_dots=True):
        """
        Nhận vào khung hình gốc, danh sách đối tượng lưu vết từ bộ điều phối PKLTracker,
        và các cờ bật/tắt vẽ khung xương, trả về khung hình đã được render hoàn chỉnh.
        """
        out_frame = frame.copy()

        for track in online_tracks:
            x1, y1, x2, y2 = map(int, track.tlbr)
            keypoints = track.keypoints
            track_id = track.track_id
            
            # --- BƯỚC 1: VẼ HỘP BAO VÀ HIGHLIGHT TIÊU ĐỀ ID RÕ NÉT ---
            # Vẽ hộp chữ nhật màu xanh Neon dày dặn (độ dày 2) bao quanh đối tượng
            main_color = (0, 255, 0) # Màu xanh lá Neon chủ đạo
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), main_color, 2)
            
            # Định dạng chuỗi văn bản nhãn ID hiển thị lớn
            label = f"ID: {track_id}"
            
            # Tự động tính toán kích thước chuỗi chữ (Chiều rộng, Chiều cao) dựa trên Font chữ
            font_scale = 0.65  # Tăng kích thước chữ từ 0.5 lên 0.65 cho lớn và rõ ràng
            font_thickness = 2 # Tăng độ dày nét chữ lên 2
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Tính toán tọa độ vẽ Thanh Banner nền đặc nằm ngay phía trên hộp bao
            banner_y1 = max(0, y1 - label_h - 12)
            banner_y2 = y1
            banner_x1 = x1
            banner_x2 = x1 + label_w + 14
            
            # Vẽ Thanh Banner nền đen đặc (-1) tạo độ tương phản tuyệt đối cho chữ nổi lên
            cv2.rectangle(out_frame, (banner_x1, banner_y1), (banner_x2, banner_y2), (0, 0, 0), -1)
            # Vẽ thêm đường viền cho Banner tiệp màu với hộp bao
            cv2.rectangle(out_frame, (banner_x1, banner_y1), (banner_x2, banner_y2), main_color, 1)
            
            # Chèn chữ ID màu xanh Neon nổi bật, sắc nét (LINE_AA) vào tâm thanh Banner đen
            text_x = x1 + 7
            text_y = y1 - 6
            cv2.putText(out_frame, label, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, main_color, font_thickness, lineType=cv2.LINE_AA)

            if len(keypoints) == 0:
                continue

            # --- BƯỚC 2: VẼ CÁC ĐƯỜNG NỐI XƯƠNG (SKELETON LINKS) CÓ ĐIỀU KIỆN ---
            if draw_links:
                for link, color in zip(self.skeleton_connections, self.link_colors):
                    pt1_idx, pt2_idx = link
                    kpt1 = keypoints[pt1_idx]
                    kpt2 = keypoints[pt2_idx]

                    if kpt1[2] > 0.4 and kpt2[2] > 0.4:
                        pos1 = (int(kpt1[0]), int(kpt1[1]))
                        pos2 = (int(kpt2[0]), int(kpt2[1]))
                        cv2.line(out_frame, pos1, pos2, color, thickness=2, lineType=cv2.LINE_AA)

            # --- BƯỚC 3: VẼ CÁC CHẤM TRÒN KHỚP NỐI (KEYPOINT DOTS) CÓ ĐIỀU KIỆN ---
            if draw_dots:
                for idx, kpt in enumerate(keypoints):
                    kx, ky, kconf = kpt
                    if kconf > 0.4:
                        color = self.kpt_colors[idx]
                        cv2.circle(out_frame, (int(kx), int(ky)), 4, color, -1, lineType=cv2.LINE_AA)

        # --- BƯỚC 4: GỌI TỰ ĐỘNG CHÈN CHỈ SỐ FPS LÊN KHUNG HÌNH CUỐI ---
        out_frame = self.draw_fps(out_frame)

        return out_frame
