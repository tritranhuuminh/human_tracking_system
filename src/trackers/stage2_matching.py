# src/trackers/stage2_matching.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.trackers.norm_l2_cost import norm_l2_distance

def norm_l2_matching(tracks, detections, unmatched_tracks, unmatched_detections, thresh_list, conf_thresh):
    """GIAI ĐOẠN 2: Giải cứu đối tượng sót do che khuất bằng khoảng cách Norm-L2 Attention Mask."""
    if len(unmatched_tracks) == 0 or len(unmatched_detections) == 0:
        return [], unmatched_tracks, unmatched_detections

    sub_tracks = [tracks[i] for i in unmatched_tracks]
    sub_dets = [detections[j] for j in unmatched_detections]

    cost_matrix = norm_l2_distance(sub_tracks, sub_dets, list(range(len(sub_tracks))), list(range(len(sub_dets))))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches_g2 = []
    final_u_track = list(unmatched_tracks)
    final_u_det = list(unmatched_detections)
    
    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        t_idx = unmatched_tracks[r]
        d_idx = unmatched_detections[c]
        
        det_conf = np.mean(detections[d_idx].keypoints[:, 2])
        
        if cost < thresh_list[0] or (cost < thresh_list[1] and det_conf > conf_thresh):
            matches_g2.append((t_idx, d_idx))
            final_u_track.remove(t_idx)
            final_u_det.remove(d_idx)
            
    return matches_g2, final_u_track, final_u_det
