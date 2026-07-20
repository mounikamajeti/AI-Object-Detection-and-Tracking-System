import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path: str, device: str = None, confidence_threshold: float = 0.4):
        self.model_path = model_path
        self.device = device or self._get_device()
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = []
        self.load_model()

    def _get_device(self) -> str:
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            return 'cpu'

    def load_model(self) -> None:
        try:
            self.model = YOLO(self.model_path)
            self.model.fuse()
            self.class_names = self.model.names
        except Exception as exc:
            raise RuntimeError(f'Failed to load YOLO model from {self.model_path}: {exc}')

    def detect_objects(self, frame: np.ndarray) -> list[dict]:
        if frame is None or frame.size == 0:
            return []
        try:
            results = self.model(frame, device=self.device, stream=False, imgsz=640, conf=self.confidence_threshold)
            detections = []
            if isinstance(results, list):
                results = results[0]
            for result in results.boxes:
                xyxy = result.xyxy.cpu().numpy().flatten().tolist()
                confidence = float(result.conf.cpu().numpy()[0])
                class_id = int(result.cls.cpu().numpy()[0])
                label = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
                detections.append({
                    'bbox': xyxy,
                    'confidence': confidence,
                    'class_id': class_id,
                    'label': label,
                })
            return detections
        except Exception as exc:
            raise RuntimeError(f'Error running object detection: {exc}')
