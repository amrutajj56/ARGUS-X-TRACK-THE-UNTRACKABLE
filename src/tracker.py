"""Detection and short-term multi-object association."""
import cv2
from ultralytics import YOLO


class TrackingEngine:
    """Runs object detection and ByteTrack; persistent IDs are added separately."""
    def __init__(self, model_path="yolov8n.pt", confidence=0.35, device="cpu"):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    def process_frame(self, frame):
        results = self.model.track(
            source=frame, persist=True, tracker="bytetrack.yaml",
            conf=self.confidence, device=self.device, verbose=False
        )
        result = results[0]
        annotated = frame.copy()
        tracks = []
        if result.boxes is None:
            return annotated, tracks
        boxes = result.boxes
        ids = boxes.id
        id_values = ids.int().cpu().tolist() if ids is not None else [None] * len(boxes)
        for bbox, track_id, confidence, class_id in zip(
            boxes.xyxy.cpu().tolist(), id_values,
            boxes.conf.cpu().tolist(), boxes.cls.int().cpu().tolist()
        ):
            x1, y1, x2, y2 = [int(v) for v in bbox]
            name = self.model.names.get(int(class_id), str(class_id))
            tracks.append({
                "track_id": track_id, "bbox": [x1, y1, x2, y2],
                "confidence": float(confidence), "class_id": int(class_id),
                "class_name": name
            })
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (45, 130, 210), 1)
        return annotated, tracks
