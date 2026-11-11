
import csv
from pathlib import Path

import cv2

from tracking.tracker import VehicleTracker
from analytics.line_counter import VehicleLineCounter

from config import COUNTING_LINE_POSITION


def process_video(input_path, output_path, csv_path):

    input_path = Path(input_path)
    output_path = Path(output_path)
    csv_path = Path(csv_path)

    # ----------------------------------
    # CHECK INPUT VIDEO
    # ----------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Video not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------
    # OPEN VIDEO
    # ----------------------------------

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(
            "Unable to open input video."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

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
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Unable to create output video."
        )

    # ----------------------------------
    # INITIALIZE TRACKING AND COUNTING
    # ----------------------------------

    tracker = VehicleTracker()

    counter = VehicleLineCounter(
        line_position=COUNTING_LINE_POSITION
    )

    frame_number = 0

    logged_event_count = 0

    print("\nStarting video processing...")

    try:

        # ----------------------------------
        # PROCESS VIDEO FRAME BY FRAME
        # ----------------------------------

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            timestamp = frame_number / fps

            # VEHICLE DETECTION AND TRACKING
            result = tracker.track_frame(frame)

            # UPDATE LINE COUNTER
            counter.update(
                result,
                frame.shape,
                timestamp
            )

            # ----------------------------------
            # DRAW DETECTION RESULTS
            # ----------------------------------

            annotated_frame = result.plot()

            line_y = int(
                height * COUNTING_LINE_POSITION
            )

            cv2.line(
                annotated_frame,
                (0, line_y),
                (width, line_y),
                (0, 0, 255),
                3
            )

            # ----------------------------------
            # DISPLAY CURRENT COUNT
            # ----------------------------------

            summary = counter.get_summary()

            cv2.putText(
                annotated_frame,
                f"Total: {summary['total_vehicles']}",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            y_position = 85

            for class_name, count in summary[
                "vehicle_counts"
            ].items():

                cv2.putText(
                    annotated_frame,
                    f"{class_name.title()}: {count}",
                    (30, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                y_position += 30

            # SAVE ANNOTATED FRAME
            writer.write(annotated_frame)

            # ----------------------------------
            # LOG NEW CROSSING EVENTS
            # ----------------------------------

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

            # ----------------------------------
            # PROCESSING PROGRESS
            # ----------------------------------

            if frame_number % 100 == 0:

                print(
                    f"Processed "
                    f"{frame_number}/{total_frames} frames"
                )

    finally:

        cap.release()
        writer.release()

    # ----------------------------------
    # FINALIZE VEHICLE CLASSIFICATIONS
    # ----------------------------------

    print("\nFinalizing vehicle classifications...")

    summary = counter.finalize()

    # ----------------------------------
    # SAVE FINAL CROSSING EVENTS TO CSV
    # ----------------------------------

    fieldnames = [
        "track_id",
        "vehicle_class",
        "direction",
        "timestamp_seconds"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        csv_writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        csv_writer.writeheader()

        clean_events = [
            {
                key: event[key]
                for key in fieldnames
            }
            for event in counter.events
        ]

        csv_writer.writerows(clean_events)

    # ----------------------------------
    # FINAL REPORT
    # ----------------------------------

    print("\n" + "=" * 45)

    print("VIDEO PROCESSING COMPLETED")

    print("=" * 45)

    print("Frames processed:", frame_number)

    print(
        "Vehicle counts:",
        summary["vehicle_counts"]
    )

    print(
        "Direction counts:",
        summary["direction_counts"]
    )

    print(
        "Total vehicles:",
        summary["total_vehicles"]
    )

    print("Output video:", output_path)

    print("Events CSV:", csv_path)

    print("=" * 45)

    return summary
