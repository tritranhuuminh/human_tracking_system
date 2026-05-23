# src/trackers/b_track.py
import numpy as np
from filterpy.kalman import KalmanFilter
from src.trackers.kpt_utils import update_kpt_ema  # Nạp hàm tách riêng

class STrack:
    def __init__(self, tlbr, confidence, keypoints, gamma=0.9):
        self.tlbr = np.asarray(tlbr, dtype=np.float32)
        self.confidence = float(confidence)
        self.keypoints = np.asarray(keypoints, dtype=np.float32)
        self.gamma = gamma
        self.track_id = None
        self.is_activated = False
        self.time_since_update = 0

        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.eye(8)
        for i in range(4): self.kf.F[i, i + 4] = 1.0
        self.kf.H = np.zeros((4, 8))
        for i in range(4): self.kf.H[i, i] = 1.0
        self.kf.P *= 10.0
        self.kf.R *= 1.0
        self.kf.Q *= 0.01

        cx, cy, w, h = self._tlbr_to_cxcyah(self.tlbr)
        self.kf.x = np.array([cx, cy, w / h, h, 0, 0, 0, 0]).reshape(8, 1)

    def _tlbr_to_cxcyah(self, tlbr):
        w = tlbr[2] - tlbr[0]
        h = tlbr[3] - tlbr[1]
        cx = tlbr[0] + w / 2.0
        cy = tlbr[1] + h / 2.0
        return cx, cy, w, h

    def _cxcyah_to_tlbr(self, cxcyah):
        cx, cy, a, h = cxcyah[0], cxcyah[1], cxcyah[2], cxcyah[3]
        w = a * h
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        self.tlbr = self._cxcyah_to_tlbr(self.kf.x[:4].flatten())

    def update(self, new_det, en_kpt_ema=True):
        """Cập nhật Kalman và gọi hàm mượt xương biệt lập."""
        self.time_since_update = 0
        self.confidence = new_det.confidence
        
        cx, cy, w, h = self._tlbr_to_cxcyah(new_det.tlbr)
        z = np.array([cx, cy, w / h, h]).reshape(4, 1)
        self.kf.update(z)
        self.tlbr = self._cxcyah_to_tlbr(self.kf.x[:4].flatten())
        
        # Lấy vận tốc thực tế
        v_cx = float(self.kf.x[4][0])
        v_cy = float(self.kf.x[5][0])
        speed = np.sqrt(v_cx**2 + v_cy**2)
        
        # Gọi hàm tách riêng từ kpt_utils
        self.keypoints = update_kpt_ema(self.keypoints, new_det.keypoints, speed, en_kpt_ema)
