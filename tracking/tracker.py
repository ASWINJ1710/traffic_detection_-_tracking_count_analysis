
from collections import defaultdict

import torch
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

        self.last_classes = {}
        self.logged_tracks = set()

        self.class_scores = defaultdict(
            lambda: defaultdict(float)
        )

        self.stable_classes = {}

        self.score_decay = 0.90
        self.switch_threshold = 1.20

        print("YOLO11 ByteTrack tracker initialized.")
        print("Model:", MODEL_PATH)
        print("Classes:", CLASS_NAMES)
        print("Class smoothing enabled.")

    # ------------------------------------------
    # TRACK FRAME
    # ------------------------------------------

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

        self._smooth_predictions(result)

        self._inspect_predictions(result)

        return result

    # ------------------------------------------
    # CLASS SMOOTHING
    # ------------------------------------------

    def _smooth_predictions(self, result):

        if result.boxes is None or len(result.boxes) == 0:
            return

        boxes = result.boxes

        if boxes.id is None:
            return

        track_ids = boxes.id.cpu().numpy().astype(int)
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        smoothed_classes = class_ids.copy()

        for index, (track_id, class_id, confidence) in enumerate(
            zip(track_ids, class_ids, confidences)
        ):

            track_id = int(track_id)
            class_id = int(class_id)

            if class_id not in CLASS_NAMES:
                continue

            # Reduce influence of older predictions
            for previous_class in list(
                self.class_scores[track_id]
            ):

                self.class_scores[track_id][
                    previous_class
                ] *= self.score_decay

            # Add current prediction
            self.class_scores[track_id][class_id] += float(
                confidence
            )

            scores = self.class_scores[track_id]

            best_class = max(scores, key=scores.get)

            current_class = self.stable_classes.get(track_id)

            if current_class is None:

                self.stable_classes[track_id] = best_class

            elif best_class != current_class:

                current_score = scores.get(
                    current_class,
                    0.0,
                )

                best_score = scores[best_class]

                if best_score > (
                    current_score * self.switch_threshold
                ):

                    self.stable_classes[track_id] = best_class

            smoothed_classes[index] = self.stable_classes[
                track_id
            ]

        # Safely update YOLO prediction classes
        with torch.inference_mode():
            boxes.cls.copy_(
                torch.as_tensor(
                    smoothed_classes,
                    dtype=boxes.cls.dtype,
                    device=boxes.cls.device,
                )
            )

    # ------------------------------------------
    # DIAGNOSTIC LOGGING
    # ------------------------------------------

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
                "unknown",
            )

            x1, y1, x2, y2 = bbox

            width = int(x2 - x1)
            height = int(y2 - y1)

            previous_class = self.last_classes.get(track_id)

            if track_id not in self.logged_tracks:

                print(
                    f"[DETECTION] ID={track_id} | "
                    f"Class={class_name} | "
                    f"Confidence={confidence:.3f} | "
                    f"Box={width}x{height}"
                )

                self.logged_tracks.add(track_id)

            elif previous_class != class_id:

                previous_name = CLASS_NAMES.get(
                    previous_class,
                    "unknown",
                )

                print(
                    f"[STABLE CLASS CHANGE] "
                    f"ID={track_id} | "
                    f"{previous_name} -> {class_name}"
                )

            self.last_classes[track_id] = int(class_id)

    # ------------------------------------------
    # RESET
    # ------------------------------------------

    def reset(self):

        self.model.predictor = None

        self.last_classes.clear()
        self.logged_tracks.clear()
        self.class_scores.clear()
        self.stable_classes.clear()

        print("Tracker and classification history reset.")
