# src/trackers/norm_l2_cost.py

import numpy as np

def norm_l2_distance(tracks, detections, unmatched_tracks, unmatched_detections):
    num_tracks = len(unmatched_tracks)
    num_dets = len(unmatched_detections)
    cost_matrix = np.ones((num_tracks, num_dets), dtype=np.float32)
    
    for i, t_idx in enumerate(unmatched_tracks):
        t = tracks[t_idx]
        bbox_w = max(1.0, t.tlbr[2] - t.tlbr[0])
        bbox_h = max(1.0, t.tlbr[3] - t.tlbr[1])
        
        for j, d_idx in enumerate(unmatched_detections):
            d = detections[d_idx]
            
            # 🌟 CƠ CHẾ CHÚ Ý (ATTENTION MASK): Khớp xương phải rõ ràng ở cả Track và Det
            # Thay vì lọc cứng (>0.3), ta tính toán trọng số tin cậy kết hợp (Joint Confidence)
            joint_conf = t.keypoints[:, 2] * d.keypoints[:, 2]
            valid_mask = joint_conf > 0.16  # Tương đương cả 2 điểm đều đạt conf > 0.4
            
            if np.sum(valid_mask) == 0:
                cost_matrix[i, j] = 1.0
                continue
                
            # Tính độ lệch khoảng cách gốc
            diff_raw = t.keypoints[valid_mask, :2] - d.keypoints[valid_mask, :2]
            diff = diff_raw.astype(np.float32, copy=True)
            
            # Chuẩn hóa hình học không gian
            diff[:, 0] /= bbox_w
            diff[:, 1] /= bbox_h
            
            # Tính khoảng cách Euclid cho từng khớp xương
            l2_distances = np.linalg.norm(diff, axis=1)
            
            # 🌟 ÁP DỤNG TRỌNG SỐ CHÚ Ý (Soft Attention Weighting)
            # Khớp nào rõ nét sẽ nhân với trọng số lớn, khớp mờ nhân với trọng số nhỏ
            weights = joint_conf[valid_mask]
            weights /= (np.sum(weights) + 1e-6) # Chuẩn hóa tổng trọng số về 1
            
            # Chi phí cuối cùng là trung bình có trọng số (Weighted Mean)
            weighted_l2_dist = np.sum(l2_distances * weights)
            
            cost_matrix[i, j] = min(1.0, float(weighted_l2_dist))
            
    return cost_matrix
