# src/trackers/bot_sort.py

import numpy as np
from src.trackers.b_track import STrack 
from src.trackers.stage1_matching import kpt_iou_matching
from src.trackers.stage2_matching import norm_l2_matching
from src.trackers.adaptive_gates import compute_adaptive_init_thresh, compute_adaptive_max_age

# NẠP CÁC KHỐI CHỨC NĂNG ĐÃ ĐƯỢC PHÂN RÃ BIỆT LẬP
from src.trackers.cascade_recovery import execute_cascade_recovery
from src.trackers.track_initiator import execute_track_initiation
from src.trackers.ghost_tracker import execute_ghost_tracking

class PKLTracker:
    """Bộ điều phối trung tâm tối cao đạt chuẩn Clean Architecture và Adaptive toàn diện 100%."""
    def __init__(self, config=None):
        self.id_count = 0
        self.tracked_tracks = [] 
        self.collision_pairs = {} 
        self.consecutive_empty_frames = 0
        
        if config is None: config = {}
        # Ép lọc ngưỡng khởi tạo ban đầu tĩnh từ cấu hình nếu hệ thống chưa có dữ liệu mồi
        self.init_thresh = config.get("init_thresh", 0.6)
        
        self.en_kpt_ema = config.get("en_kpt_ema", True)
        self.en_cascade_recovery = config.get("en_cascade_recovery", True)
        self.en_depth_lock = config.get("en_depth_lock", True)
        self.en_ivbb_ghost = config.get("en_ivbb_ghost", True)
        self.system_fps = config.get("system_fps", 30.0)

    def _calculate_iou(self, box1, box2):
        """Tính toán nhanh IoU diện tích giữa 2 hộp bao [x1, y1, x2, y2]."""
        ix1 = np.maximum(box1[0], box2[0])
        iy1 = np.maximum(box1[1], box2[1])
        ix2 = np.minimum(box1[2], box2[2])
        iy2 = np.minimum(box1[3], box2[3])
        iw = np.maximum(0., ix2 - ix1)
        ih = np.maximum(0., iy2 - iy1)
        if iw * ih <= 0: return 0.0
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return (iw * ih) / (area1 + area2 - iw * ih + 1e-6)

    def _calculate_containment_ratio(self, lost_box, active_box):
        """Tính tỷ lệ hộp bao lọt thỏm hoàn toàn (nuốt trọn hình thể)."""
        ix1 = np.maximum(lost_box[0], active_box[0])
        iy1 = np.maximum(lost_box[1], active_box[1])
        ix2 = np.minimum(lost_box[2], active_box[2])
        iy2 = np.minimum(lost_box[3], active_box[3])
        iw = np.maximum(0., ix2 - ix1)
        ih = np.maximum(0., iy2 - iy1)
        if iw * ih <= 0: return 0.0
        area_lost = (lost_box[2] - lost_box[0]) * (lost_box[3] - lost_box[1])
        return (iw * ih) / (area_lost + 1e-6)

    def update(self, detections):
        """Pipeline thực thi xử lý bám vết thích ứng ngữ cảnh."""
        num_dets, num_tracks = len(detections), len(self.tracked_tracks)
        
        # GIAI ĐOẠN 0: BIÊN TRỐNG THÍCH ỨNG (Tự động giải phóng bộ đệm và reset ID)
        adaptive_max_idle_frames = int(self.system_fps * 2.0)
        if num_dets == 0 and num_tracks == 0:
            if self.consecutive_empty_frames + 1 >= adaptive_max_idle_frames: self.id_count = 0 
            self.consecutive_empty_frames += 1
            return []
        self.consecutive_empty_frames = 0

        # Ước lượng ngưỡng khởi tạo động của khung hình
        adaptive_init_thresh = compute_adaptive_init_thresh(detections)

        # Trường hợp khung hình đầu tiên khởi chạy hệ thống
        if num_tracks == 0:
            for det in detections:
                if det.confidence >= adaptive_init_thresh:
                    self.id_count += 1
                    new_track = STrack(det.tlbr, det.confidence, det.keypoints)
                    new_track.track_id = self.id_count
                    new_track.is_activated = True
                    self.tracked_tracks.append(new_track)
            return self.tracked_tracks

        # BƯỚC 1: KALMAN PREDICT (Dự đoán xu hướng động lượng)
        for track in self.tracked_tracks: track.predict()

        # 🌟 VÁ LỖI RUNTIME: Chỉ cho phép tính trung bình ma trận ngưỡng nếu mảng detections có phần tử
        if detections is not None and len(detections) > 0:
            avg_kpt_iou_thresh = np.mean([d.dynamic_kpt_iou_thresholds for d in detections], axis=0).tolist()
            avg_norm_l2_thresh = np.mean([d.dynamic_norm_l2_thresholds for d in detections], axis=0).tolist()
            avg_min_pose_conf = float(np.mean([d.dynamic_min_pose_conf for d in detections]))
        else:
            # Tham số biên an toàn mặc định nếu khung hình hiện tại hoàn toàn trống người
            avg_kpt_iou_thresh, avg_norm_l2_thresh, avg_min_pose_conf = [0.25, 0.5, 0.9], [0.6, 1.0], 0.5

        # BƯỚC 2 & 3: GIAI ĐOẠN 1 MATCHING (Kpt-IoU)
        matches_g1, u_track_g1, u_det_g1 = kpt_iou_matching(self.tracked_tracks, detections, avg_kpt_iou_thresh)

        final_activated_tracks, matched_detect_indices = [], set()
        for t_idx, d_idx in matches_g1:
            track = self.tracked_tracks[t_idx]
            track.update(detections[d_idx], self.en_kpt_ema) 
            final_activated_tracks.append(track)
            matched_detect_indices.add(d_idx)

        # BƯỚC 4 & 5: GIAI ĐOẠN 2 MATCHING (Norm-L2 Attention Mask)
        if len(u_track_g1) > 0 and len(u_det_g1) > 0:
            matches_g2, final_u_track, final_u_det = norm_l2_matching(
                self.tracked_tracks, detections, u_track_g1, u_det_g1, avg_norm_l2_thresh, avg_min_pose_conf
            )
            for t_idx, d_idx in matches_g2:
                track = self.tracked_tracks[t_idx]
                track.update(detections[d_idx], self.en_kpt_ema)
                final_activated_tracks.append(track)
                matched_detect_indices.add(d_idx)
        else:
            final_u_track, final_u_det = u_track_g1, u_det_g1

        # BƯỚC 5.5: GỌI KHỐI ĐỘC LẬP - CASCADE MOMENTUM RE-ID
        still_unmatched_dets = execute_cascade_recovery(
            final_u_track, final_u_det, self.tracked_tracks, detections, final_activated_tracks,
            matched_detect_indices, self.collision_pairs, self.en_cascade_recovery, self.en_depth_lock, self.en_kpt_ema
        )

        # BƯỚC 6: GỌI KHỐI ĐỘC LẬP - ADAPTIVE TRACK INITIATION
        execute_track_initiation(
            still_unmatched_dets, detections, final_activated_tracks, adaptive_init_thresh, self
        )

        # BƯỚC 7: GỌI KHỐI ĐỘC LẬP - BIOMETRIC DEPTH SCALE IVBB
        execute_ghost_tracking(
            final_u_track, self.tracked_tracks, final_activated_tracks, self.collision_pairs, self.en_ivbb_ghost, self
        )

        # 🌟 CẬP NHẬT CHUẨN XÁC: DỌN DẸP BỘ NHỚ ĐỆM COLLISION AN TOÀN CHỐNG RÒ RỈ RAM
        # Căn lề độc lập, loại bỏ hoàn toàn hiện tượng ngắt hàm sớm, hỗ trợ index [0] của mảng cấu trúc cặp
        for active_t in final_activated_tracks:
            if active_t.track_id in self.collision_pairs:
                partner_id = self.collision_pairs[active_t.track_id][0]
                partner = next((t for t in final_activated_tracks if t.track_id == partner_id), None)
                if partner is not None and self._calculate_iou(active_t.tlbr, partner.tlbr) < 0.02:
                    self.collision_pairs.pop(active_t.track_id, None)

        # 🌟 CẬP NHẬT CHUẨN XÁC: TUỔI THỌ LƯU VẾT ĐỘNG THÍCH ỨNG (Adaptive Max Age)
        # Đẩy hẳn ra ngoài rìa biên vòng lặp, chạy tuần tự để kết xuất đầy đủ ID về màn hình laptop
        active_people_count = len(final_activated_tracks)
        for lost_idx in final_u_track:
            lost_t = self.tracked_tracks[lost_idx]
            adaptive_max_age = compute_adaptive_max_age(lost_t, active_people_count, self.system_fps)
            if lost_t.time_since_update <= adaptive_max_age: 
                final_activated_tracks.append(lost_t)

        self.tracked_tracks = final_activated_tracks
        return [t for t in self.tracked_tracks if t.time_since_update == 0]
