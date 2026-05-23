# src/trackers/ghost_tracker.py
import numpy as np
from src.trackers.adaptive_gates import compute_adaptive_containment_gate

def execute_ghost_tracking(final_u_track, tracked_tracks, final_activated_tracks, collision_pairs, 
                           en_ivbb_ghost, tracker_instance):
    """Thực thi Bước 7: Bộ lọc chiều sâu sinh học IVBB sinh bóng ma Ghost Track chạy ngầm."""
    if en_ivbb_ghost:
        for t_idx in list(final_u_track):
            lost_track = tracked_tracks[t_idx]
            for active_track in final_activated_tracks:
                containment_ratio = tracker_instance._calculate_containment_ratio(lost_track.tlbr, active_track.tlbr)
                lost_h = lost_track.tlbr[3] - lost_track.tlbr[1]
                active_h = active_track.tlbr[3] - active_track.tlbr[1]
                
                adaptive_containment_gate = compute_adaptive_containment_gate(lost_h, active_h)
                
                if containment_ratio > adaptive_containment_gate and active_h > lost_h:
                    v_cx = float(lost_track.kf.x[4][0])
                    v_cy = float(lost_track.kf.x[5][0])
                    lost_track.keypoints[:, :2] += np.array([v_cx, v_cy])
                    
                    lost_cx = lost_track.tlbr[0] + (lost_track.tlbr[2] - lost_track.tlbr[0]) / 2.0
                    active_cx = active_track.tlbr[0] + (active_track.tlbr[2] - active_track.tlbr[0]) / 2.0
                    original_x_rel = np.sign(lost_cx - active_cx)
                    
                    collision_pairs[lost_track.track_id] = [active_track.track_id, 0, original_x_rel]
                    collision_pairs[active_track.track_id] = [lost_track.track_id, 0, -original_x_rel]

                    lost_track.time_since_update = 0 
                    final_activated_tracks.append(lost_track)
                    final_u_track.remove(t_idx)
                    break


def compute_adaptive_overlap_thresh(det, active_people_count):
    """
    🌟 HÀM TOÁN HỌC ADAPTIVE MỚI: Tự động điều tiết ngưỡng chống đè lấp không gian.
    - Khi xưởng đông người (active_people_count tăng), không gian tranh chấp hẹp -> Nới lỏng ngưỡng trần lên đến 0.65.
    - Khi det có confidence rất cao -> Ưu tiên mô hình AI, nới lỏng ngưỡng để cho phép khởi tạo nhanh, chống bỏ sót.
    """
    # Mô hình hóa hàm thích ứng phi tuyến bằng hàm kẹp biên min-max
    base_overlap = 0.35 + 0.15 * min(1.0, active_people_count / 8.0)
    adaptive_thresh = base_overlap + 0.15 * det.confidence
    return min(0.65, adaptive_thresh)