import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class ObjectTracker:
    def __init__(self, max_age: int = 30, n_init: int = 3, max_iou_distance: float = 0.7):
        try:
            self.tracker = DeepSort(
                max_iou_distance=max_iou_distance,
                max_age=max_age,
                n_init=n_init,
                embedder='mobilenet',
                half=True,
                bgr=True,
                embedder_gpu=True,
            )
        except Exception as exc:
            raise RuntimeError(f'Failed to initialize Deep SORT tracker: {exc}')

    def update_tracks(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        raw_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            raw_detections.append(([x1, y1, width, height], det['confidence'], det['label']))

        try:
            tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        except Exception as exc:
            raise RuntimeError(f'Failed to update tracker: {exc}')

        track_results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tlbr = track.to_tlbr()
            track_results.append({
                'track_id': track.track_id,
                'bbox': [int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3])],
                'class_name': track.get_det_class() or 'object',
                'confidence': float(track.get_det_conf() or 0.0),
                'center': [int((tlbr[0] + tlbr[2]) / 2), int((tlbr[1] + tlbr[3]) / 2)],
                'state': track.is_confirmed(),
            })
        return track_results
