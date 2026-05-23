# src/trackers/momentum_cost.py
import numpy as np
from src.trackers.kpt_utils import get_skeleton_biometric_center

def calculate_momentum_score(track, det):
    """Tính toán điểm tương đồng động lượng đa kiểm chứng."""
    v_cx = float(track.kf.x[4][0])
    v_cy = float(track.kf.x[5][0])
    t_velocity = np.array([v_cx, v_cy])
    
    # 🌟 ĐÃ SỬA: Gọi từ file chức năng xương biệt lập
    t_center = get_skeleton_biometric_center(track.keypoints, track.tlbr)
    pred_center_momentum = t_center + t_velocity * track.time_since_update
    d_center = get_skeleton_biometric_center(det.keypoints, det.tlbr)
    
    spatial_distance = np.linalg.norm(pred_center_momentum - d_center)
    track_w = track.tlbr[2] - track.tlbr[0]
    track_h = track.tlbr[3] - track.tlbr[1]
    normalization_factor = np.sqrt(track_w**2 + track_h**2) + 1e-6
    momentum_cost = spatial_distance / normalization_factor
    
    det_w = det.tlbr[2] - det.tlbr[0]
    det_h = det.tlbr[3] - det.tlbr[1]
    scale_ratio = min(track_w * track_h, det_w * det_h) / max(track_w * track_h, det_w * det_h + 1e-6)
    
    ar_track = track_w / (track_h + 1e-6)
    ar_det = det_w / (det_h + 1e-6)
    ar_similarity = min(ar_track, ar_det) / max(ar_track, ar_det + 1e-6)
    
    return (1.0 - momentum_cost) * 0.6 + scale_ratio * 0.2 + ar_similarity * 0.2
