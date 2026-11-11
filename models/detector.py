from pathlib import Path

from ultralytics import YOLO

from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    IMAGE_SIZE,
    DEVICE,
)


class VehicleDetector:
    """YOLO11 vehicle detection module."""

    def __init__(self):
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}"
            )

        self.model = YOLO(str(MODEL_PATH))

        print("YOLO11 vehicle detector loaded successfully.")

    def detect_frame(self, frame):
        """Detect vehicles in a single video frame."""

        results = self.model.predict(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )

        return results[0]

    def get_class_names(self):
        """Return the trained vehicle class names."""

        return self.model.names
