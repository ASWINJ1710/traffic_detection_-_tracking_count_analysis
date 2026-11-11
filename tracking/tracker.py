
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_THRESHOLD,
    IMAGE_SIZE,
    DEVICE,
    TRACKER,
    CLASS_NAMES,
)


class VehicleTracker:

    def __init__(self):

        self.model = YOLO(str(MODEL_PATH))

        # Store previous class predictions for each track
        self.last_classes = {}

        # Avoid printing the same initial prediction repeatedly
        self.logged_tracks = set()

        print("YOLO11 ByteTrack tracker initialized.")
        print("Model:", MODEL_PATH)
        print("Classes:", CLASS_NAMES)

    def track_frame(self, frame):

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=TRACKER,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]

        self._inspect_predictions(result)

        return result

    def _inspect_predictions(self, result):

        if result.boxes is None or len(result.boxes) == 0:
            return

        boxes = result.boxes

        if boxes.id is None:
            return

        track_ids = boxes.id.cpu().numpy().astype(int)
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        coordinates = boxes.xyxy.cpu().numpy()

        for track_id, class_id, confidence, bbox in zip(
            track_ids,
            class_ids,
            confidences,
            coordinates,
        ):

            class_name = CLASS_NAMES.get(
                int(class_id),
                "unknown"
            )

            x1, y1, x2, y2 = bbox

            width = int(x2 - x1)
            height = int(y2 - y1)

            previous_class = self.last_classes.get(track_id)

            # First prediction for this tracking ID
            if track_id not in self.logged_tracks:

                print(
                    f"[DETECTION] ID={track_id} | "
                    f"Class={class_name} | "
                    f"Confidence={confidence:.3f} | "
                    f"Box={width}x{height}"
                )

                self.logged_tracks.add(track_id)

            # Detect class changes during tracking
            elif previous_class != class_id:

                previous_name = CLASS_NAMES.get(
                    previous_class,
                    "unknown"
                )

                print(
                    f"[CLASS CHANGE] ID={track_id} | "
                    f"{previous_name} -> {class_name} | "
                    f"Confidence={confidence:.3f}"
                )

            # Save latest prediction
            self.last_classes[track_id] = int(class_id)

    def reset(self):
        """Reset tracking and diagnostic state between videos."""

        self.model.predictor = None

        self.last_classes.clear()
        self.logged_tracks.clear()

        print("Tracker and diagnostic state reset.")
