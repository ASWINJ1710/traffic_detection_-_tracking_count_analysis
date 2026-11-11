
from collections import defaultdict

from config import CLASS_NAMES, COUNTING_LINE_POSITION


class VehicleLineCounter:

    def __init__(self, line_position=None):

        self.line_position = (
            line_position
            if line_position is not None
            else COUNTING_LINE_POSITION
        )

        self.previous_positions = {}

        # Confidence-weighted predictions for each track
        self.class_scores = defaultdict(
            lambda: defaultdict(float)
        )

        self.class_votes = defaultdict(
            lambda: defaultdict(int)
        )

        self.counted_ids = set()

        # Store crossing details until processing finishes
        self.pending_events = []

        self.events = []

        self.vehicle_counts = {
            name: 0
            for name in CLASS_NAMES.values()
        }

        self.direction_counts = {
            "UP": 0,
            "DOWN": 0
        }

    def update(
        self,
        result,
        frame_shape,
        timestamp_seconds=0.0
    ):

        height = frame_shape[0]

        line_y = int(height * self.line_position)

        if result.boxes is None or len(result.boxes) == 0:
            return self.get_summary()

        boxes = result.boxes

        if boxes.id is None:
            return self.get_summary()

        coordinates = boxes.xyxy.cpu().numpy()

        track_ids = boxes.id.cpu().numpy().astype(int)

        class_ids = boxes.cls.cpu().numpy().astype(int)

        confidences = boxes.conf.cpu().numpy()

        for bbox, track_id, class_id, confidence in zip(
            coordinates,
            track_ids,
            class_ids,
            confidences
        ):

            # ----------------------------------
            # VALIDATE CLASS
            # ----------------------------------

            class_name = CLASS_NAMES.get(int(class_id))

            if class_name is None:
                continue

            # ----------------------------------
            # CONFIDENCE-WEIGHTED CLASS VOTING
            # ----------------------------------

            confidence = float(confidence)

            self.class_scores[track_id][class_name] += (
                confidence ** 2
            )

            self.class_votes[track_id][class_name] += 1

            # ----------------------------------
            # VEHICLE CENTER POSITION
            # ----------------------------------

            x1, y1, x2, y2 = bbox

            center_y = int((y1 + y2) / 2)

            previous_y = self.previous_positions.get(track_id)

            self.previous_positions[track_id] = center_y

            if previous_y is None:
                continue

            # ----------------------------------
            # CHECK LINE CROSSING
            # ----------------------------------

            crossed_down = (
                previous_y < line_y
                and center_y >= line_y
            )

            crossed_up = (
                previous_y > line_y
                and center_y <= line_y
            )

            if not crossed_down and not crossed_up:
                continue

            if track_id in self.counted_ids:
                continue

            direction = (
                "DOWN" if crossed_down else "UP"
            )

            self.counted_ids.add(track_id)

            # Save event; final class will be assigned
            # after all frames have been processed.
            self.pending_events.append({
                "track_id": int(track_id),
                "direction": direction,
                "timestamp_seconds": round(
                    float(timestamp_seconds), 2
                )
            })

            print(
                f"[CROSSING] ID={track_id} | "
                f"{direction} | "
                f"{timestamp_seconds:.2f}s"
            )

        return self.get_summary()

    def finalize(self):

        """
        Determine final class using predictions from
        the complete tracked vehicle history.
        """

        self.events = []

        self.vehicle_counts = {
            name: 0
            for name in CLASS_NAMES.values()
        }

        self.direction_counts = {
            "UP": 0,
            "DOWN": 0
        }

        for pending in self.pending_events:

            track_id = pending["track_id"]

            scores = self.class_scores.get(track_id, {})

            if not scores:
                continue

            final_class = max(
                scores,
                key=scores.get
            )

            event = {
                "track_id": track_id,
                "vehicle_class": final_class,
                "direction": pending["direction"],
                "timestamp_seconds": pending[
                    "timestamp_seconds"
                ]
            }

            self.events.append(event)

            self.vehicle_counts[final_class] += 1

            self.direction_counts[
                pending["direction"]
            ] += 1

            print(
                f"[FINAL CLASS] ID={track_id} | "
                f"{final_class.upper()} | "
                f"Votes={dict(self.class_votes[track_id])}"
            )

        return self.get_summary()

    def get_summary(self):

        return {
            "vehicle_counts": self.vehicle_counts.copy(),

            "direction_counts": self.direction_counts.copy(),

            "total_vehicles": len(self.counted_ids)
        }

    def reset(self):

        self.previous_positions.clear()

        self.class_scores.clear()

        self.class_votes.clear()

        self.counted_ids.clear()

        self.pending_events.clear()

        self.events.clear()

        self.vehicle_counts = {
            name: 0
            for name in CLASS_NAMES.values()
        }

        self.direction_counts = {
            "UP": 0,
            "DOWN": 0
        }
