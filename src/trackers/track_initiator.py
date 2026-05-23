# src/trackers/track_initiator.py
import numpy as np
from src.trackers.b_track import STrack
from src.trackers.adaptive_gates import compute_adaptive_overlap_thresh

def execute_track_initiation(still_unmatched_dets, detections, final_activated_tracks, 
                             adaptive_init_thresh, tracker_instance):
    """Thực thi Bước 6: Khởi tạo ID người mới sử dụng Ngưỡng Đè Lấp Thích Ứng Động."""
    active_people_count = len(final_activated_tracks)
    
    for d_idx in still_unmatched_dets:
        det = detections[d_idx]
        if det.confidence < adaptive_init_thresh: continue
        
        is_in_collision_zone = False
        for active_track in final_activated_tracks:
            # 🌟 ĐỒNG BỘ: Gọi tính toán ngưỡng động dựa trên diện thể và mật độ người thực tế
            adaptive_overlap_thresh = compute_adaptive_overlap_thresh(det, active_people_count)
            
            if tracker_instance._calculate_iou(det.tlbr, active_track.tlbr) > adaptive_overlap_thresh:
                is_in_collision_zone = True
                break
        if is_in_collision_zone: continue

        tracker_instance.id_count += 1
        new_track = STrack(det.tlbr, det.confidence, det.keypoints)
        new_track.track_id = tracker_instance.id_count
        new_track.is_activated = True
        final_activated_tracks.append(new_track)
