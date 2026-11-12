
from collections import defaultdict

from config import CLASS_NAMES, COUNTING_LINE_POSITION


class VehicleLineCounter:
    """
    Vehicle counting using ByteTrack IDs and a virtual horizontal line.

    Features:
    - Hysteresis zone to reduce false crossings.
    - Bottom-centre or geometric-centre reference point.
    - UP/DOWN direction detection.
    - Confidence-weighted class classification.
    - Live and final class-wise statistics.
    - One count per tracking ID.
    """

    def __init__(
        self,
        line_position=None,
        crossing_margin=12,
        position_mode="bottom",
    ):

        self.line_position = (
            line_position
            if line_position is not None
            else COUNTING_LINE_POSITION
        )

        self.crossing_margin = max(0, int(crossing_margin))

        if position_mode not in ("bottom", "center"):
            raise ValueError(
                "position_mode must be 'bottom' or 'center'"
            )

        self.position_mode = position_mode

        # Position and stable side for each tracking ID
        self.previous_positions = {}
        self.last_stable_sides = {}

        # Class prediction history
        self.class_scores = defaultdict(
            lambda: defaultdict(float)
        )

        self.class_votes = defaultdict(
            lambda: defaultdict(int)
        )

        # Unique vehicle tracking
        self.counted_ids = set()

        # Crossing event storage
        self.pending_events = []
        self.events = []

        # Statistics
        self.vehicle_counts = self._empty_vehicle_counts()

        self.direction_counts = {
            "UP": 0,
            "DOWN": 0,
        }

    # --------------------------------------------------
    # INITIALIZE EMPTY COUNTS
    # --------------------------------------------------

    @staticmethod
    def _empty_vehicle_counts():

        return {
            name: 0
            for name in CLASS_NAMES.values()
        }

    # --------------------------------------------------
    # DETERMINE VEHICLE SIDE
    # --------------------------------------------------

    def _get_vehicle_side(self, position_y, line_y):

        upper_boundary = line_y - self.crossing_margin
        lower_boundary = line_y + self.crossing_margin

        if position_y < upper_boundary:
            return -1

        if position_y > lower_boundary:
            return 1

        return 0

    # --------------------------------------------------
    # GET BEST VEHICLE CLASS
    # --------------------------------------------------

    def _get_current_class(self, track_id):

        scores = self.class_scores.get(track_id, {})

        if not scores:
            return None

        return max(scores, key=scores.get)

    # --------------------------------------------------
    # UPDATE CLASSIFICATION HISTORY
    # --------------------------------------------------

    def _update_class_history(
        self,
        track_id,
        class_id,
        confidence,
    ):

        class_name = CLASS_NAMES.get(int(class_id))

        if class_name is None:
            return

        confidence = float(confidence)

        self.class_scores[track_id][class_name] += (
            confidence ** 2
        )

        self.class_votes[track_id][class_name] += 1

    # --------------------------------------------------
    # GET VEHICLE REFERENCE POINT
    # --------------------------------------------------

    def _get_reference_point(self, bbox):

        x1, y1, x2, y2 = bbox

        center_x = int((x1 + x2) / 2)

        if self.position_mode == "bottom":
            reference_y = int(y2)
        else:
            reference_y = int((y1 + y2) / 2)

        return center_x, reference_y

    # --------------------------------------------------
    # UPDATE COUNTER
    # --------------------------------------------------

    def update(
        self,
        result,
        frame_shape,
        timestamp_seconds=0.0,
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
            confidences,
        ):

            if int(class_id) not in CLASS_NAMES:
                continue

            # Update classification history
            self._update_class_history(
                track_id,
                class_id,
                confidence,
            )

            # Obtain reference position
            center_x, reference_y = self._get_reference_point(
                bbox
            )

            self.previous_positions[track_id] = (
                center_x,
                reference_y,
            )

            current_side = self._get_vehicle_side(
                reference_y,
                line_y,
            )

            # Initialize stable side
            if track_id not in self.last_stable_sides:

                if current_side != 0:
                    self.last_stable_sides[track_id] = current_side

                continue

            # Ignore movement inside crossing zone
            if current_side == 0:
                continue

            previous_side = self.last_stable_sides[track_id]

            # Update stable side
            self.last_stable_sides[track_id] = current_side

            # No side change
            if current_side == previous_side:
                continue

            # Prevent duplicate counting
            if track_id in self.counted_ids:
                continue

            direction = (
                "DOWN"
                if previous_side == -1 and current_side == 1
                else "UP"
            )

            self.counted_ids.add(track_id)

            event = {
                "track_id": int(track_id),
                "direction": direction,
                "timestamp_seconds": round(
                    float(timestamp_seconds),
                    2,
                ),
            }

            self.pending_events.append(event)

            print(
                f"[CROSSING] "
                f"ID={track_id} | "
                f"{direction} | "
                f"{timestamp_seconds:.2f}s"
            )

        self._refresh_live_counts()

        return self.get_summary()

    # --------------------------------------------------
    # REFRESH LIVE STATISTICS
    # --------------------------------------------------

    def _refresh_live_counts(self):

        self.vehicle_counts = self._empty_vehicle_counts()

        self.direction_counts = {
            "UP": 0,
            "DOWN": 0,
        }

        for event in self.pending_events:

            track_id = event["track_id"]

            vehicle_class = self._get_current_class(
                track_id
            )

            if vehicle_class is None:
                continue

            self.vehicle_counts[vehicle_class] += 1

            self.direction_counts[
                event["direction"]
            ] += 1

    # --------------------------------------------------
    # FINALIZE VEHICLE CLASSIFICATIONS
    # --------------------------------------------------

    def finalize(self):

        self.events = []

        self.vehicle_counts = self._empty_vehicle_counts()

        self.direction_counts = {
            "UP": 0,
            "DOWN": 0,
        }

        for pending in self.pending_events:

            track_id = pending["track_id"]

            final_class = self._get_current_class(
                track_id
            )

            if final_class is None:
                continue

            event = {
                "track_id": track_id,
                "vehicle_class": final_class,
                "direction": pending["direction"],
                "timestamp_seconds": pending[
                    "timestamp_seconds"
                ],
            }

            self.events.append(event)

            self.vehicle_counts[final_class] += 1

            self.direction_counts[
                pending["direction"]
            ] += 1

            print(
                f"[FINAL CLASS] "
                f"ID={track_id} | "
                f"{final_class.upper()} | "
                f"Votes={dict(self.class_votes[track_id])}"
            )

        return self.get_summary()

    # --------------------------------------------------
    # RETURN SUMMARY
    # --------------------------------------------------

    def get_summary(self):

        return {
            "vehicle_counts": self.vehicle_counts.copy(),

            "direction_counts": self.direction_counts.copy(),

            "total_vehicles": len(self.pending_events),
        }

    # --------------------------------------------------
    # RESET COUNTER
    # --------------------------------------------------

    def reset(self):

        self.previous_positions.clear()

        self.last_stable_sides.clear()

        self.class_scores.clear()

        self.class_votes.clear()

        self.counted_ids.clear()

        self.pending_events.clear()

        self.events.clear()

        self.vehicle_counts = self._empty_vehicle_counts()

        self.direction_counts = {
            "UP": 0,
            "DOWN": 0,
        }

        print("Vehicle line counter reset.")
