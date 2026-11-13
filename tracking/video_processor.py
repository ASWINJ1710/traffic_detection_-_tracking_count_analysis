
import csv
import json
from pathlib import Path

import cv2

from tracking.tracker import VehicleTracker
from analytics.line_counter import VehicleLineCounter
from analytics.speed_estimator import VehicleSpeedEstimator

from config import COUNTING_LINE_POSITION, CLASS_NAMES


# ==================================================
# DRAW SPEED LABEL
# ==================================================

def draw_speed_label(
    frame,
    bbox,
    label,
    color=(0, 255, 255),
):

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = map(int, bbox)

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = max(
        0.65,
        min(1.1, width / 3500),
    )

    thickness = max(
        2,
        int(font_scale * 2.5),
    )

    padding = max(
        7,
        int(font_scale * 9),
    )

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    box_width = text_width + padding * 2

    box_height = (
        text_height + baseline + padding * 2
    )

    label_x = max(
        0,
        min(x1, width - box_width),
    )

    label_y = max(
        0,
        min(y1 + 8, height - box_height),
    )

    cv2.rectangle(
        frame,
        (label_x, label_y),
        (
            label_x + box_width,
            label_y + box_height,
        ),
        (25, 25, 25),
        -1,
    )

    cv2.rectangle(
        frame,
        (label_x, label_y),
        (
            label_x + box_width,
            label_y + box_height,
        ),
        color,
        2,
    )

    cv2.putText(
        frame,
        label,
        (
            label_x + padding,
            label_y + padding + text_height,
        ),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ==================================================
# MAIN VIDEO PROCESSOR
# ==================================================

def process_video(
    input_path,
    output_path,
    csv_path,
    frame_callback=None,
):

    input_path = Path(input_path)
    output_path = Path(output_path)
    csv_path = Path(csv_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Video not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open input video: {input_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            "Unable to create output video."
        )

    # Existing vehicle detection and tracking.
    tracker = VehicleTracker()

    # Existing vehicle counting.
    counter = VehicleLineCounter(
        line_position=COUNTING_LINE_POSITION
    )

    # Experimental perspective-based speed estimation.
    # Current road dimensions are assumptions, not verified measurements.
    physical_speed_estimator = VehicleSpeedEstimator(
        history_seconds=0.8,
        min_samples=5,
        allow_assumed_dimensions=True,
    )

    # Latest available speed information for CSV.
    latest_speed_by_track = {}

    frame_number = 0
    logged_event_count = 0

    line_y = int(
        height * COUNTING_LINE_POSITION
    )

    print("\nStarting video processing...")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print("YOLO detection: ENABLED")
    print("ByteTrack tracking: ENABLED")
    print("Vehicle counting: ENABLED")
    print("Random speed simulation: DISABLED")
    print("Approximate perspective speed estimation: ENABLED")
    print("Pixel-based fallback speed estimation: ENABLED")
    print("Speed calibration: ASSUMED 3.5m x 10m")
    print("IMPORTANT: km/h values are experimental estimates, not verified speeds.")

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            timestamp = frame_number / fps

            # --------------------------------------
            # VEHICLE DETECTION AND TRACKING
            # --------------------------------------

            result = tracker.track_frame(
                frame
            )

            # --------------------------------------
            # VEHICLE COUNTING
            # --------------------------------------

            counter.update(
                result,
                frame.shape,
                timestamp,
            )

            boxes = result.boxes

            current_labels = {}

            # --------------------------------------
            # --------------------------------------
            # APPROXIMATE PERSPECTIVE SPEED
            # --------------------------------------

            if (
                boxes is not None
                and boxes.id is not None
                and boxes.cls is not None
            ):

                track_ids = boxes.id.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                bounding_boxes = boxes.xyxy.cpu().tolist()

                for track_id, class_id, bbox in zip(
                    track_ids,
                    class_ids,
                    bounding_boxes,
                ):

                    track_id = int(track_id)
                    class_id = int(class_id)

                    vehicle_class = CLASS_NAMES.get(
                        class_id,
                        "unknown",
                    ).lower()

                    approximate_info = physical_speed_estimator.update(
                        track_id,
                        bbox,
                        timestamp,
                    )

                    if approximate_info is not None:

                        estimated_speed = approximate_info["speed_kmh"]

                        latest_speed_by_track[track_id] = {
                            "road": approximate_info["road"],
                            "speed_kmh": estimated_speed,
                            "vehicle_class": vehicle_class,
                            "speed_type": "approximate_perspective",
                            "calibration_status": (
                                approximate_info["calibration_status"]
                            ),
                        }

                        label = f"~{estimated_speed:.1f} km/h"
                        color = (0, 165, 255)

                    else:
                        label = "Measuring..."
                        color = (0, 165, 255)

                    current_labels[track_id] = (
                        label,
                        color,
                    )

            # DRAW YOLO DETECTIONS
            # --------------------------------------

            annotated_frame = result.plot(
                line_width=2,
                font_size=0.55,
                labels=True,
                conf=True,
            )

            # --------------------------------------
            # DRAW SPEED LABELS
            # --------------------------------------

            if (
                boxes is not None
                and boxes.id is not None
            ):

                track_ids = (
                    boxes.id.int().cpu().tolist()
                )

                bounding_boxes = (
                    boxes.xyxy.cpu().tolist()
                )

                for track_id, bbox in zip(
                    track_ids,
                    bounding_boxes,
                ):

                    track_id = int(track_id)

                    label_info = current_labels.get(
                        track_id
                    )

                    if label_info is None:
                        continue

                    label, color = label_info

                    draw_speed_label(
                        annotated_frame,
                        bbox,
                        label,
                        color,
                    )

            # --------------------------------------
            # COUNTING LINE
            # --------------------------------------

            cv2.line(
                annotated_frame,
                (0, line_y),
                (width, line_y),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            # --------------------------------------
            # VEHICLE COUNT SUMMARY
            # --------------------------------------

            summary = counter.get_summary()

            cv2.putText(
                annotated_frame,
                f"Total: {summary['total_vehicles']}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            y_position = 65

            for class_name, count in summary[
                "vehicle_counts"
            ].items():

                cv2.putText(
                    annotated_frame,
                    f"{class_name.title()}: {count}",
                    (20, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                y_position += 22

            # --------------------------------------
            # FRAME INFORMATION
            # --------------------------------------

            cv2.putText(
                annotated_frame,
                f"Frame: {frame_number}/{total_frames}",
                (20, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            writer.write(
                annotated_frame
            )

            # --------------------------------------
            # STREAMLIT LIVE PREVIEW
            # --------------------------------------

            if frame_callback is not None:

                if frame_number % max(
                    1,
                    int(fps / 5),
                ) == 0:

                    frame_callback(
                        annotated_frame
                    )

            # --------------------------------------
            # LOG CROSSING EVENTS
            # --------------------------------------

            while logged_event_count < len(
                counter.pending_events
            ):

                event = counter.pending_events[
                    logged_event_count
                ]

                print(
                    f"[CROSSING EVENT] "
                    f"Frame={frame_number} | "
                    f"ID={event['track_id']} | "
                    f"{event['direction']} | "
                    f"{event['timestamp_seconds']}s"
                )

                logged_event_count += 1

            # --------------------------------------
            # PROCESSING PROGRESS
            # --------------------------------------

            if frame_number % 100 == 0:

                print(
                    f"Processed "
                    f"{frame_number}/{total_frames} frames"
                )

    finally:

        cap.release()
        writer.release()

    # ----------------------------------------------
    # FINALIZE VEHICLE CLASSIFICATIONS
    # ----------------------------------------------

    print("\nFinalizing vehicle classifications...")

    summary = counter.finalize()

    # ----------------------------------------------
    # SAVE CROSSING EVENTS TO CSV
    # ----------------------------------------------

    fieldnames = [
        "track_id",
        "vehicle_class",
        "direction",
        "timestamp_seconds",
        "road_side",
        "speed_kmh",
        "speed_type",
        "calibration_status",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        csv_writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        csv_writer.writeheader()

        clean_events = []

        for event in counter.events:

            track_id = int(
                event["track_id"]
            )

            speed_info = latest_speed_by_track.get(
                track_id,
                {},
            )

            clean_event = {
                "track_id": track_id,
                "vehicle_class": event["vehicle_class"],
                "direction": event["direction"],
                "timestamp_seconds": event["timestamp_seconds"],
                "road_side": speed_info.get("road", ""),
                "speed_kmh": speed_info.get("speed_kmh", ""),
                "speed_type": speed_info.get(
                    "speed_type",
                    "unavailable",
                ),
                "calibration_status": speed_info.get(
                    "calibration_status",
                    "NOT_AVAILABLE",
                ),
            }

            clean_events.append(
                clean_event
            )

        csv_writer.writerows(
            clean_events
        )

    # ----------------------------------------------
    # FINAL REPORT
    # ----------------------------------------------

    print("\n" + "=" * 45)
    print("VIDEO PROCESSING COMPLETED")
    print("=" * 45)

    print(
        "Frames processed:",
        frame_number,
    )

    print(
        "Vehicle counts:",
        summary["vehicle_counts"],
    )

    print(
        "Direction counts:",
        summary["direction_counts"],
    )

    print(
        "Total vehicles:",
        summary["total_vehicles"],
    )

    approximate_speed_count = sum(
        1
        for speed_info in latest_speed_by_track.values()
        if speed_info.get("speed_type") == "approximate_perspective"
    )

    print(
        "Vehicles with approximate perspective speeds:",
        approximate_speed_count,
    )

    print(
        "Output video:",
        output_path,
    )

    print(
        "Events CSV:",
        csv_path,
    )

    print("=" * 45)

    return summary
