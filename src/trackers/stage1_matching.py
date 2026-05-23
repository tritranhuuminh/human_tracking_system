# src/trackers/stage1_matching.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.trackers.kpt_iou_cost import kpt_iou_distance

def kpt_iou_matching(tracks, detections, thresh_list):
    """GIAI ĐOẠN 1: Sử dụng ma trận Kpt-IoU đa ngưỡng để giải gán nhãn toàn cục."""
    if len(tracks) == 0 or len(detections) == 0:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost_matrix = kpt_iou_distance(tracks, detections)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches, u_track, u_det = [], list(range(len(tracks))), list(range(len(detections)))
    
    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        # So sánh với ngưỡng adaptive động được truyền vào
        if cost < thresh_list[0] or cost < thresh_list[1] or cost < thresh_list[2]:
            matches.append((r, c))
            u_track.remove(r)
            u_det.remove(c)
            
    return matches, u_track, u_det
