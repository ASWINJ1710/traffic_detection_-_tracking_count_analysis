
import cv2
import json
import numpy as np
from pathlib import Path

IMAGE_PATH = Path("outputs/calibration_frame.jpg")
OUTPUT_PATH = Path("outputs/speed_calibration.json")

LANE_WIDTH_METRES = 10.0
ROAD_LENGTH_METRES = 15.0

DISPLAY_WIDTH = 1280

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

original_height, original_width = image.shape[:2]

scale = DISPLAY_WIDTH / original_width
display_height = int(original_height * scale)

base_image = cv2.resize(
    image, (DISPLAY_WIDTH, display_height)
)

road_names = ["left", "right"]

point_labels = [
    "Near-left corner",
    "Near-right corner",
    "Far-right corner",
    "Far-left corner",
]

all_points = {
    "left": [],
    "right": [],
}

current_road_index = 0


def draw_screen():
    frame = base_image.copy()

    road = road_names[current_road_index]
    selected = all_points[road]

    cv2.putText(
        frame,
        f"Calibrating {road.upper()} ROAD",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Point {len(selected) + 1}/4",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    if len(selected) < 4:
        cv2.putText(
            frame,
            "Click: " + point_labels[len(selected)],
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

    for name in road_names:
        pts = all_points[name]

        for i, point in enumerate(pts):
            cv2.circle(frame, point, 7, (0, 255, 0), -1)

            cv2.putText(
                frame,
                f"{name[0].upper()}{i + 1}",
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        if len(pts) == 4:
            polygon = np.array(pts, dtype=np.int32)

            cv2.polylines(
                frame,
                [polygon],
                True,
                (0, 255, 255),
                2,
            )

    cv2.putText(
        frame,
        "R = Reset | N = Next road | S = Save | ESC = Exit",
        (20, display_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    return frame


def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    road = road_names[current_road_index]

    if len(all_points[road]) < 4:
        all_points[road].append((x, y))

        print(
            f"{road.upper()} ROAD - "
            f"{point_labels[len(all_points[road]) - 1]}: "
            f"{(x, y)}"
        )

        if len(all_points[road]) == 4:
            print(f"{road.upper()} ROAD: 4 points completed.")

            if current_road_index == 0:
                print("Press N to select the RIGHT ROAD.")


def create_calibration(points):
    # Convert displayed coordinates into original image coordinates.
    pixel_points = (
        np.array(points, dtype=np.float32) / scale
    )

    world_points = np.array(
        [
            [0, 0],
            [LANE_WIDTH_METRES, 0],
            [LANE_WIDTH_METRES, ROAD_LENGTH_METRES],
            [0, ROAD_LENGTH_METRES],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(
        pixel_points,
        world_points,
    )

    return {
        "homography": homography.tolist(),

        # Save original image coordinates for road-boundary checking.
        "source_points": pixel_points.tolist(),

        "lane_width_metres": LANE_WIDTH_METRES,
        "road_length_metres": ROAD_LENGTH_METRES,
        "assumed_dimensions": True,
    }


cv2.namedWindow("Two-Road Speed Calibration")

cv2.setMouseCallback(
    "Two-Road Speed Calibration",
    mouse_callback,
)

while True:
    cv2.imshow(
        "Two-Road Speed Calibration",
        draw_screen(),
    )

    key = cv2.waitKey(20) & 0xFF

    if key == ord("r"):
        all_points = {
            "left": [],
            "right": [],
        }

        current_road_index = 0
        print("All calibration points reset.")

    elif key == ord("n"):
        if len(all_points["left"]) == 4:
            current_road_index = 1
            print("Select 4 points for the RIGHT ROAD.")
        else:
            print("Complete all 4 LEFT ROAD points first.")

    elif key == ord("s"):
        if (
            len(all_points["left"]) == 4
            and len(all_points["right"]) == 4
        ):
            calibration = {
                "left_road": create_calibration(
                    all_points["left"]
                ),

                "right_road": create_calibration(
                    all_points["right"]
                ),

                "image_width": original_width,
                "image_height": original_height,
            }

            OUTPUT_PATH.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(OUTPUT_PATH, "w") as file:
                json.dump(calibration, file, indent=4)

            print("\nBoth road calibrations saved!")
            print(f"File: {OUTPUT_PATH}")
            print("Original pixel boundaries saved for both roads.")

            break

        else:
            print("Complete both road calibrations before saving.")
            print("Press N after finishing the left road.")

    elif key == 27:
        print("Calibration cancelled.")
        break

cv2.destroyAllWindows()
