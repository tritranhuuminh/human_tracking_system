# src/trackers/adaptive_gates.py

import numpy as np

def compute_adaptive_init_thresh(detections):
    """
    Mô hình hóa toán học: Ngưỡng khởi tạo ID mới tự thích nghi theo môi trường.
    Môi trường càng mờ/nhiễu (env_conf_mean thấp) -> siết chặt ngưỡng để chống ID ma.
    """
    if len(detections) > 0:
        env_conf_mean = np.mean([d.confidence for d in detections])
        return max(0.45, min(0.75, 0.95 - 0.5 * env_conf_mean))
    return 0.60

def compute_adaptive_momentum_gate(target_track):
    """
    Mô hình hóa toán học: Ngưỡng động lượng quán tính thích ứng theo vận tốc Kalman.
    Vận tốc thực tế càng cao, sai số tích lũy càng lớn -> tự động nới lỏng biên để Re-ID.
    """
    v_cx = float(target_track.kf.x[4][0])
    v_cy = float(target_track.kf.x[5][0])
    speed = np.sqrt(v_cx**2 + v_cy**2)
    return max(0.25, 0.55 - 0.02 * speed)

def compute_adaptive_overlap_thresh(det, active_people_count):
    """
    🌟 HÀM TOÁN HỌC ADAPTIVE ĐÃ ĐỒNG BỘ: Tự động điều tiết ngưỡng chống đè lấp không gian.
    - Khi xưởng đông người (active_people_count tăng), không gian tranh chấp hẹp -> Nới lỏng ngưỡng trần lên đến 0.65.
    - Khi det có confidence rất cao -> Ưu tiên mô hình AI, nới lỏng ngưỡng để cho phép khởi tạo nhanh, chống bỏ sót.
    """
    # Mô hình hóa hàm thích ứng phi tuyến bằng hàm kẹp biên min-max
    base_overlap = 0.35 + 0.15 * min(1.0, active_people_count / 8.0)
    adaptive_thresh = base_overlap + 0.15 * det.confidence
    return min(0.65, adaptive_thresh)


def compute_adaptive_containment_gate(lost_h, active_h):
    """
    Mô hình hóa toán học: Ngưỡng kích hoạt Ghost Track thích ứng theo tỉ lệ chiều cao (Delta Height).
    Người đứng trước càng to (vùng khuất càng lớn) -> hạ ngưỡng kích hoạt để bắt trọn thực thể ảo.
    """
    delta_h_ratio = min(1.0, active_h / (lost_h + 1e-6))
    return max(0.60, 0.85 - 0.25 * delta_h_ratio)

def compute_adaptive_max_age(lost_track, active_people_count, system_fps):
    """
    Mô hình hóa toán học: Tuổi thọ lưu vết động (Instance-Level Adaptive Max Age).
    Người có độ tin cậy lịch sử cao -> nhớ lâu. Mật độ xưởng quá đông -> thu ngắn bộ nhớ tránh rác ma trận.
    """
    adaptive_max_age = int(system_fps * (5.0 + 5.0 * lost_track.confidence - min(4.0, 0.5 * active_people_count)))
    return max(int(system_fps * 2.0), adaptive_max_age)
