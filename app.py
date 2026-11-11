
import csv
import shutil
import subprocess
import uuid
import html
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

from config import (
    BASE_DIR,
    VIDEO_DIR,
    OUTPUT_DIR,
    CLASS_NAMES,
    SUPPORTED_VIDEO_FORMATS,
)

from tracking.video_processor import process_video


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Traffic Analytics System",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Traffic Video Tracking & Analytics System")
st.caption(
    "YOLO11n Vehicle Detection | ByteTrack Tracking | "
    "Line Crossing Analytics"
)


# --------------------------------------------------
# DIRECTORIES
# --------------------------------------------------

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# HELPER: READ CSV WITHOUT PANDAS OR PYARROW
# --------------------------------------------------

def read_csv_file(csv_path):

    if not Path(csv_path).exists():
        return []

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# --------------------------------------------------
# HELPER: HTML TABLE
# --------------------------------------------------

def display_html_table(rows):

    if not rows:
        st.info("No vehicle crossing events recorded.")
        return

    headers = list(rows[0].keys())

    table_html = """
    <style>
    .traffic-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }

    .traffic-table th {
        background-color: #1f2937;
        color: white;
        padding: 12px;
        text-align: left;
    }

    .traffic-table td {
        padding: 10px;
        border-bottom: 1px solid #dddddd;
    }

    .traffic-table tr:nth-child(even) {
        background-color: #f3f4f6;
    }
    </style>

    <table class="traffic-table">
    <thead>
    <tr>
    """

    for header in headers:
        table_html += f"<th>{html.escape(header)}</th>"

    table_html += "</tr></thead><tbody>"

    for row in rows:
        table_html += "<tr>"

        for header in headers:
            value = html.escape(str(row.get(header, "")))
            table_html += f"<td>{value}</td>"

        table_html += "</tr>"

    table_html += "</tbody></table>"

    st.markdown(table_html, unsafe_allow_html=True)


# --------------------------------------------------
# HELPER: CONVERT VIDEO TO H264
# --------------------------------------------------

def convert_video_to_h264(input_path):

    input_path = Path(input_path)

    output_path = input_path.with_name(
        input_path.stem + "_browser.mp4"
    )

    # Find FFmpeg
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return None

    command = [
        ffmpeg_path,
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode == 0 and output_path.exists():
            if output_path.stat().st_size > 0:
                return output_path

    except Exception as error:
        st.warning(f"Video conversion failed: {error}")

    return None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Project Information")

st.sidebar.write("**Detection Model:** YOLO11n")
st.sidebar.write("**Tracking Algorithm:** ByteTrack")
st.sidebar.write("**Detection Classes:**")

for class_name in CLASS_NAMES.values():
    st.sidebar.write(f"- {class_name.title()}")

st.sidebar.write("**Counting:** Line Crossing")

st.sidebar.divider()

st.sidebar.info(
    "Upload a traffic video to detect and track vehicles."
)


# --------------------------------------------------
# VIDEO UPLOAD
# --------------------------------------------------

st.header("1. Upload Traffic Video")

uploaded_video = st.file_uploader(
    "Choose a traffic video",
    type=[
        extension.replace(".", "")
        for extension in SUPPORTED_VIDEO_FORMATS
    ],
)

if uploaded_video is not None:

    st.success(f"Uploaded: {uploaded_video.name}")

    if st.button(
        "▶ Start Traffic Analysis",
        type="primary",
        use_container_width=True,
    ):

        unique_id = uuid.uuid4().hex[:8]

        input_path = VIDEO_DIR / f"input_{unique_id}.mp4"
        output_path = OUTPUT_DIR / f"tracking_{unique_id}.mp4"
        csv_path = OUTPUT_DIR / f"events_{unique_id}.csv"

        with open(input_path, "wb") as file:
            file.write(uploaded_video.getbuffer())

        progress = st.progress(0)
        status = st.empty()

        try:
            status.info("Processing video with YOLO11n and ByteTrack...")

            progress.progress(20)

            summary = process_video(
                str(input_path),
                str(output_path),
                str(csv_path),
            )

            progress.progress(85)

            status.info("Preparing browser-compatible video...")

            browser_video = convert_video_to_h264(output_path)

            progress.progress(100)

            if browser_video is not None:
                final_video = str(browser_video)
                status.success("Traffic analysis completed successfully.")
            else:
                final_video = str(output_path)
                status.warning(
                    "Analysis completed, but H.264 conversion was unavailable. "
                    "The original processed video is available for download."
                )

            st.session_state["summary"] = summary
            st.session_state["video_path"] = final_video
            st.session_state["original_video_path"] = str(output_path)
            st.session_state["csv_path"] = str(csv_path)

        except Exception as error:
            status.error(f"Processing failed: {error}")
            st.exception(error)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

if "summary" in st.session_state:

    summary = st.session_state["summary"]

    video_path = Path(st.session_state["video_path"])
    csv_path = Path(st.session_state["csv_path"])

    vehicle_counts = summary.get(
        "vehicle_counts", {}
    )

    direction_counts = summary.get(
        "direction_counts", {}
    )

    total_vehicles = summary.get(
        "total_vehicles", 0
    )

    st.divider()

    st.header("2. Traffic Analytics Dashboard")

    # ----------------------------------------------
    # KPI METRICS
    # ----------------------------------------------

    st.subheader("Vehicle Crossing Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Vehicles",
        total_vehicles,
    )

    col2.metric(
        "UP Direction",
        direction_counts.get("UP", 0),
    )

    col3.metric(
        "DOWN Direction",
        direction_counts.get("DOWN", 0),
    )

    # ----------------------------------------------
    # CLASS COUNTS
    # ----------------------------------------------

    st.subheader("Vehicle Class Counts")

    class_columns = st.columns(4)

    for index, class_name in enumerate(CLASS_NAMES.values()):

        count = vehicle_counts.get(class_name, 0)

        class_columns[index].metric(
            class_name.title(),
            count,
        )

    # ----------------------------------------------
    # CHARTS
    # ----------------------------------------------

    st.divider()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.subheader("Vehicles by Class")

        class_names = list(CLASS_NAMES.values())

        class_values = [
            vehicle_counts.get(name, 0)
            for name in class_names
        ]

        bar_chart = go.Figure(
            data=[
                go.Bar(
                    x=class_names,
                    y=class_values,
                    text=class_values,
                    textposition="auto",
                )
            ]
        )

        bar_chart.update_layout(
            xaxis_title="Vehicle Class",
            yaxis_title="Crossing Count",
            template="plotly_white",
        )

        st.plotly_chart(
            bar_chart,
            use_container_width=True,
        )

    with chart_col2:

        st.subheader("Direction Distribution")

        directions = ["UP", "DOWN"]

        direction_values = [
            direction_counts.get("UP", 0),
            direction_counts.get("DOWN", 0),
        ]

        if sum(direction_values) > 0:

            pie_chart = go.Figure(
                data=[
                    go.Pie(
                        labels=directions,
                        values=direction_values,
                        hole=0.4,
                    )
                ]
            )

            pie_chart.update_layout(
                template="plotly_white",
            )

            st.plotly_chart(
                pie_chart,
                use_container_width=True,
            )

        else:
            st.info("No direction crossings recorded.")

    # ----------------------------------------------
    # PROCESSED VIDEO
    # ----------------------------------------------

    st.divider()

    st.header("3. Processed Traffic Video")

    if video_path.exists():

        st.video(str(video_path))

        with open(video_path, "rb") as file:

            st.download_button(
                label="Download Processed Video",
                data=file.read(),
                file_name=video_path.name,
                mime="video/mp4",
                use_container_width=True,
            )

    else:
        st.error("Processed video file not found.")

    # ----------------------------------------------
    # CROSSING EVENTS
    # ----------------------------------------------

    st.divider()

    st.header("4. Vehicle Crossing Events")

    events = read_csv_file(csv_path)

    display_html_table(events)

    if csv_path.exists():

        with open(csv_path, "rb") as file:

            st.download_button(
                label="Download Crossing Events CSV",
                data=file.read(),
                file_name="vehicle_crossing_events.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ----------------------------------------------
    # FOOTNOTE
    # ----------------------------------------------

    st.divider()

    st.caption(
        "Note: Counts represent unique tracked vehicles that crossed "
        "the virtual counting line. They do not represent every vehicle "
        "visible in the video."
    )

else:

    st.info(
        "Upload a traffic video and click Start Traffic Analysis "
        "to view the results."
    )
