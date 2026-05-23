# src/trackers/matching.py

import numpy as np
from scipy.optimize import linear_sum_assignment
from src.trackers.kpt_iou_cost import kpt_iou_distance  # Gọi khối chi phí 1
from src.trackers.norm_l2_cost import norm_l2_distance  # Gọi khối chi phí 2

def kpt_iou_matching(tracks, detections, thresh_list=[0.25, 0.5, 0.9]):
    """
    GIAI ĐOẠN 1: Sử dụng ma trận Kpt-IoU để giải bài toán gán nhãn Hungarian toàn cục.
    """
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    # Tính toán ma trận chi phí hình học đa giác xương người
    cost_matrix = kpt_iou_distance(tracks, detections)

    # Thực thi giải thuật Hungarian tối ưu toàn cục
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches, u_track, u_det = [], list(range(len(tracks))), list(range(len(detections)))
    
    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        # Bộ lọc phân tách đa ngưỡng cấu hình của bài báo khoa học áp dụng cho người
        if cost < thresh_list[0] or cost < thresh_list[1] or cost < thresh_list[2]:
            matches.append((r, c))
            u_track.remove(r)
            u_det.remove(c)
            
    return matches, u_track, u_det


def norm_l2_matching(tracks, detections, unmatched_tracks, unmatched_detections, 
                     thresh_list=[0.6, 1.0], conf_thresh=0.8):
    """
    GIAI ĐOẠN 2: Giải cứu các đối tượng sót lại bằng giải thuật Hungarian trên ma trận Norm-L2.
    """
    if len(unmatched_tracks) == 0 or len(unmatched_detections) == 0:
        return [], unmatched_tracks, unmatched_detections

    # Tính toán ma trận khoảng cách dịch chuyển khớp xương chuẩn hóa
    cost_matrix = norm_l2_distance(tracks, detections, unmatched_tracks, unmatched_detections)

    # Chạy giải toán tối ưu Hungarian cho Giai đoạn 2
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches_g2 = []
    final_u_track = list(unmatched_tracks)
    final_u_det = list(unmatched_detections)
    
    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        
        # Ánh xạ chỉ số ma trận về đúng chỉ số ID thực tế của bể chứa (Pool)
        t_idx = unmatched_tracks[r]
        d_idx = unmatched_detections[c]
        
        # Tính độ tin cậy trung bình của các khớp mới phát hiện (YOLO-Pose)
        det_conf = np.mean(detections[d_idx].keypoints[:, 2])
        
        # Logic điều kiện gán nhãn bên phải sơ đồ:
        # Thỏa mãn ngưỡng khoảng cách nghiêm ngặt HOẶC (Ngưỡng nới lỏng VÀ độ tin cậy mô hình AI cao)
        if cost < thresh_list[0] or (cost < thresh_list[1] and det_conf > conf_thresh):
            matches_g2.append((t_idx, d_idx))
            final_u_track.remove(t_idx)
            final_u_det.remove(d_idx)
            
    return matches_g2, final_u_track, final_u_det
