import cv2
import os
from pathlib import Path

class VideoUtils:
    @staticmethod
    def resize_frame(frame, width=1024, height=576):
        if frame is None:
            return None
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def get_video_writer(output_path: str, fps: float, frame_size: tuple[int, int]):
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        return cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            frame_size,
        )

    @staticmethod
    def draw_detection(frame, bbox, label, confidence, track_id, color=(0, 255, 0)):
        if frame is None:
            return frame
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f'{label} ID:{track_id} {int(confidence * 100)}%'
        cv2.putText(frame, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame
