
import json
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np


class VehicleSpeedEstimator:
    """
    Vehicle speed estimator using:
    - Perspective homography
    - Bottom-centre vehicle position
    - Multi-frame linear regression
    - Per-track history
    - Outlier rejection
    """

    def __init__(
        self,
        calibration_path="outputs/speed_calibration.json",
        history_seconds=0.6,
        min_samples=6,
        max_speed_kmh=180.0,
    ):
        self.calibration_path = Path(calibration_path)

        if not self.calibration_path.exists():
            raise FileNotFoundError(
                f"Calibration file not found: {self.calibration_path}"
            )

        with open(self.calibration_path, "r") as file:
            calibration = json.load(file)

        self.calibrations = {}

        for road_name in ("left_road", "right_road"):
            if road_name not in calibration:
                continue

            road = calibration[road_name]

            if road.get("assumed_dimensions", True):
                raise ValueError(
                    f"{road_name} uses assumed dimensions. "
                    "Enter actual measured dimensions and recalibrate."
                )

            required = (
                "homography",
                "source_points",
                "lane_width_metres",
                "road_length_metres",
            )

            if any(key not in road for key in required):
                raise ValueError(
                    f"Incomplete calibration for {road_name}."
                )

            homography = np.asarray(
                road["homography"],
                dtype=np.float64,
            )

            polygon = np.asarray(
                road["source_points"],
                dtype=np.float32,
            )

            width = float(road["lane_width_metres"])
            length = float(road["road_length_metres"])

            if homography.shape != (3, 3):
                raise ValueError(
                    f"Invalid homography for {road_name}."
                )

            if polygon.shape != (4, 2):
                raise ValueError(
                    f"Invalid road boundary points for {road_name}."
                )

            if width <= 0 or length <= 0:
                raise ValueError(
                    f"Invalid physical dimensions for {road_name}."
                )

            if not np.all(np.isfinite(homography)):
                raise ValueError(
                    f"Non-finite homography for {road_name}."
                )

            self.calibrations[road_name] = {
                "homography": homography,
                "polygon": polygon,
                "width": width,
                "length": length,
            }

        if not self.calibrations:
            raise ValueError(
                "No valid road calibration was found."
            )

        self.history_seconds = float(history_seconds)
        self.min_samples = int(min_samples)
        self.max_speed_kmh = float(max_speed_kmh)

        self.track_history = defaultdict(deque)

    def get_world_position(self, point):
        pixel_point = (
            float(point[0]),
            float(point[1]),
        )

        for road_name, road in self.calibrations.items():
            inside = cv2.pointPolygonTest(
                road["polygon"],
                pixel_point,
                False,
            )

            if inside < 0:
                continue

            pixel_array = np.array(
                [[[pixel_point[0], pixel_point[1]]]],
                dtype=np.float32,
            )

            world_point = cv2.perspectiveTransform(
                pixel_array,
                road["homography"].astype(np.float32),
            )[0][0]

            x = float(world_point[0])
            y = float(world_point[1])

            if not np.isfinite(x) or not np.isfinite(y):
                return None, None

            if not (
                0 <= x <= road["width"]
                and 0 <= y <= road["length"]
            ):
                return None, None

            return road_name, (x, y)

        return None, None

    def update(self, track_id, bbox, timestamp):
        x1, y1, x2, y2 = map(float, bbox)

        bottom_center = (
            (x1 + x2) / 2.0,
            y2,
        )

        road_name, world_position = self.get_world_position(
            bottom_center
        )

        if world_position is None:
            return None

        history = self.track_history[int(track_id)]

        # Keep only samples from the same road.
        if history and history[-1][2] != road_name:
            history.clear()

        # Ignore duplicate or backward timestamps.
        if history and timestamp <= history[-1][0]:
            return None

        history.append(
            (
                float(timestamp),
                world_position,
                road_name,
            )
        )

        # Keep a short rolling history.
        while (
            history
            and timestamp - history[0][0]
            > self.history_seconds + 0.5
        ):
            history.popleft()

        if len(history) < self.min_samples:
            return None

        samples = [
            sample
            for sample in history
            if timestamp - sample[0] <= self.history_seconds
        ]

        if len(samples) < self.min_samples:
            return None

        times = np.asarray(
            [sample[0] for sample in samples],
            dtype=np.float64,
        )

        positions = np.asarray(
            [sample[1] for sample in samples],
            dtype=np.float64,
        )

        elapsed = times[-1] - times[0]

        if elapsed < 0.3:
            return None

        # Fit X and Y independently against time.
        relative_times = times - times.mean()

        try:
            velocity_x = np.polyfit(
                relative_times,
                positions[:, 0],
                1,
            )[0]

            velocity_y = np.polyfit(
                relative_times,
                positions[:, 1],
                1,
            )[0]

        except (ValueError, np.linalg.LinAlgError):
            return None

        speed_mps = float(
            np.sqrt(velocity_x ** 2 + velocity_y ** 2)
        )

        speed_kmh = speed_mps * 3.6

        if not np.isfinite(speed_kmh):
            return None

        if speed_kmh < 0 or speed_kmh > self.max_speed_kmh:
            return None

        return {
            "track_id": int(track_id),
            "road": road_name,
            "speed_kmh": round(speed_kmh, 1),
            "position": world_position,
            "samples": len(samples),
            "calibrated": True,
        }

    def reset(self):
        self.track_history.clear()
