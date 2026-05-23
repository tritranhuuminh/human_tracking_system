# src/trackers/kpt_utils.py
import numpy as np

def update_kpt_ema(current_keypoints, new_keypoints, speed, en_kpt_ema=True):
    """Cập nhật làm mượt khung xương bằng EMA Động Lũy Thừa."""
    if not en_kpt_ema:
        # Nếu tắt EMA, lấy 100% tọa độ thô tươi từ YOLO-Pose
        current_keypoints[:, :2] = new_keypoints[:, :2]
        current_keypoints[:, 2] = new_keypoints[:, 2]
        return current_keypoints

    # Công thức lũy thừa phi tuyến động dựa trên vận tốc thực của Kalman
    decay_rate = 0.5
    g_dynamic = 0.05 + (0.7 - 0.05) * np.exp(-decay_rate * speed)
    
    current_keypoints[:, :2] = g_dynamic * current_keypoints[:, :2] + (1 - g_dynamic) * new_det_kpts[:, :2] if 'new_det_kpts' in locals() else g_dynamic * current_keypoints[:, :2] + (1 - g_dynamic) * new_keypoints[:, :2]
    current_keypoints[:, 2] = new_keypoints[:, 2]
    return current_keypoints

def get_skeleton_biometric_center(keypoints, tlbr):
    """Tính toán trọng tâm thực sinh học từ các khớp xương rõ nét."""
    valid_mask = keypoints[:, 2] > 0.4
    if np.sum(valid_mask) >= 3:
        return np.mean(keypoints[valid_mask, :2], axis=0)
    else:
        # Phương án dự phòng nếu bị che khuất sâu: Dùng tâm hộp bao
        w = tlbr[2] - tlbr[0]
        h = tlbr[3] - tlbr[1]
        return np.array([tlbr[0] + w / 2.0, tlbr[1] + h / 2.0])
