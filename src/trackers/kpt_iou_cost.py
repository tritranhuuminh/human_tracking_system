# src/trackers/kpt_iou_cost.py

import numpy as np
import cv2

def calculate_kpt_convex_hull_iou(kpts1, kpts2):
    """
    Tính toán diện tích đè lên nhau (Intersection over Union - IoU) 
    giữa hai vùng bao đa giác (Convex Hull) trích xuất từ 17 keypoints xương người.
    """
    # Bước 1: Lọc nhiễu - Chỉ lấy các khớp có độ tin cậy tốt (conf > 0.3)
    pts1 = kpts1[kpts1[:, 2] > 0.3][:, :2].astype(np.int32)
    pts2 = kpts2[kpts2[:, 2] > 0.3][:, :2].astype(np.int32)
    
    # Điều kiện biên toán học: Một hình đa giác lồi bắt buộc phải có ít nhất 3 điểm
    if len(pts1) < 3 or len(pts2) < 3:
        return 0.0
        
    # Bước 2: Tạo đa giác lồi bao quanh tập hợp các khớp xương người bằng OpenCV
    hull1 = cv2.convexHull(pts1)
    hull2 = cv2.convexHull(pts2)
    
    # Tìm hộp bao giới hạn (Bounding Rect) chứa cả 2 đa giác để thu hẹp không gian tính toán ma trận
    all_pts = np.vstack((hull1, hull2))
    x, y, w, h = cv2.boundingRect(all_pts)
    
    # Khởi tạo 2 mặt nạ ảnh đen trắng (Mask) kích thước tối giản để tiết kiệm RAM
    mask1 = np.zeros((h + 20, w + 20), dtype=np.uint8)
    mask2 = np.zeros((h + 20, w + 20), dtype=np.uint8)
    
    # Dịch chuyển tọa độ các đa giác về gốc không gian mặt nạ nhỏ (0,0) và tô màu trắng (255)
    cv2.drawContours(mask1, [hull1 - [x - 10, y - 10]], -1, 255, -1)
    cv2.drawContours(mask2, [hull2 - [x - 10, y - 10]], -1, 255, -1)
    
    # Bước 3: Áp dụng phép toán logic bitwise để tính diện tích Giao (Intersection) và Hợp (Union)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # Trả về kết quả IoU (Kiểu số thực float từ 0.0 đến 1.0)
    if union == 0:
        return 0.0
    return float(intersection) / union


def kpt_iou_distance(tracks, detections):
    num_tracks = len(tracks)
    num_dets = len(detections)
    cost_matrix = np.ones((num_tracks, num_dets), dtype=np.float32)
    
    for i in range(num_tracks):
        # Tính tâm của track cũ dựa trên các keypoint uy tín
        t_valid = tracks[i].keypoints[:, 2] > 0.4
        t_center = np.mean(tracks[i].keypoints[t_valid, :2], axis=0) if np.sum(t_valid) > 0 else np.array([0, 0])
        
        for j in range(num_dets):
            iou = calculate_kpt_convex_hull_iou(tracks[i].keypoints, detections[j].keypoints)
            
            # Tính tâm của detection mới
            d_valid = detections[j].keypoints[:, 2] > 0.4
            d_center = np.mean(detections[j].keypoints[d_valid, :2], axis=0) if np.sum(d_valid) > 0 else np.array([0, 0])
            
            # Tính khoảng cách khoảng cách tâm xương chuẩn hóa theo kích thước hình học
            if np.sum(t_valid) > 0 and np.sum(d_valid) > 0:
                center_dist = np.linalg.norm(t_center - d_center)
                # Chuẩn hóa khoảng cách tâm theo đường chéo hộp bao
                w = tracks[i].tlbr[2] - tracks[i].tlbr[0]
                h = tracks[i].tlbr[3] - tracks[i].tlbr[1]
                diag = np.sqrt(w**2 + h**2) + 1e-6
                center_penalty = min(1.0, center_dist / diag)
            else:
                center_penalty = 1.0

            # 🌟 ĐÓNG GÓP MỚI: Nếu IoU lớn (chồng nhau) nhưng tâm xương lệch xa nhau -> Phạt nặng chi phí để không gán nhầm ID
            cost_matrix[i, j] = 1.0 - iou + 0.4 * center_penalty
            
    return cost_matrix
