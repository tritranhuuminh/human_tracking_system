# src/trackers/cascade_recovery.py
import numpy as np
from src.trackers.momentum_cost import calculate_momentum_score
from src.trackers.adaptive_gates import compute_adaptive_momentum_gate

def execute_cascade_recovery(final_u_track, final_u_det, tracked_tracks, detections, 
                             final_activated_tracks, matched_detect_indices, collision_pairs, 
                             en_cascade_recovery, en_depth_lock, en_kpt_ema):
    """Thực thi Bước 5.5: Dòng thác khóa quán tính chiều sâu đảo chiều vật lý."""
    still_unmatched_dets = []
    
    for d_idx in final_u_det:
        if d_idx in matched_detect_indices: continue
        det = detections[d_idx]
        
        # Nếu TẮT cơ chế quán tính nâng cao, lùi về dùng IoU hộp bao phẳng truyền thống
        if not en_cascade_recovery:
            best_iou, best_track_idx = 0.0, -1
            for t_idx in final_u_track:
                ix1 = np.maximum(det.tlbr[0], tracked_tracks[t_idx].tlbr[0])
                iy1 = np.maximum(det.tlbr[1], tracked_tracks[t_idx].tlbr[1])
                ix2 = np.minimum(det.tlbr[2], tracked_tracks[t_idx].tlbr[2])
                iy2 = np.minimum(det.tlbr[3], tracked_tracks[t_idx].tlbr[3])
                iw = np.maximum(0., ix2 - ix1)
                ih = np.maximum(0., iy2 - iy1)
                if iw * ih > 0:
                    a1 = (det.tlbr[2] - det.tlbr[0]) * (det.tlbr[3] - det.tlbr[1])
                    a2 = (tracked_tracks[t_idx].tlbr[2] - tracked_tracks[t_idx].tlbr[0]) * (tracked_tracks[t_idx].tlbr[3] - tracked_tracks[t_idx].tlbr[1])
                    iou = (iw * ih) / (a1 + a2 - iw * ih + 1e-6)
                    if iou > best_iou: best_iou, best_track_idx = iou, t_idx
            
            if best_iou > 0.25 and best_track_idx != -1:
                target_track = tracked_tracks[best_track_idx]
                target_track.update(det, en_kpt_ema)
                target_track.time_since_update = 0
                final_activated_tracks.append(target_track)
                matched_detect_indices.add(d_idx)
                final_u_track.remove(best_track_idx)
            else:
                still_unmatched_dets.append(d_idx)
            continue
            
        # LUỒNG BẬT QUÁN TÍNH CHUYÊN SÂU
        best_momentum_score, best_track_idx = -1e6, -1
        for t_idx in final_u_track:
            momentum_score = calculate_momentum_score(tracked_tracks[t_idx], det)
            if momentum_score > best_momentum_score: best_momentum_score, best_track_idx = momentum_score, t_idx
                
        if best_momentum_score > 0.40 and best_track_idx != -1:
            target_track = tracked_tracks[best_track_idx]
            adaptive_momentum_gate = compute_adaptive_momentum_gate(target_track)
            
            if best_momentum_score > adaptive_momentum_gate:
                is_id_hijacked = False
                if en_depth_lock and target_track.track_id in collision_pairs:
                    partner_id, _, original_x_rel = collision_pairs[target_track.track_id]
                    partner_track = next((t for t in final_activated_tracks if t.track_id == partner_id), None)
                    if partner_track is not None:
                        det_cx = det.tlbr[0] + (det.tlbr[2] - det.tlbr[0]) / 2.0
                        partner_cx = partner_track.tlbr[0] + (partner_track.tlbr[2] - partner_track.tlbr[0]) / 2.0
                        if np.sign(det_cx - partner_cx) == original_x_rel: is_id_hijacked = True
                
                if not is_id_hijacked:
                    target_track.update(det, en_kpt_ema)
                    target_track.time_since_update = 0 
                    final_activated_tracks.append(target_track)
                    matched_detect_indices.add(d_idx)
                    final_u_track.remove(best_track_idx)
                else: still_unmatched_dets.append(d_idx)
            else: still_unmatched_dets.append(d_idx)
        else: still_unmatched_dets.append(d_idx)
        
    return still_unmatched_dets
