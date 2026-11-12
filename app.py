
import csv
import shutil
import subprocess
import uuid
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

from config import (
    VIDEO_DIR,
    OUTPUT_DIR,
    CLASS_NAMES,
    SUPPORTED_VIDEO_FORMATS,
)

from tracking.video_processor import process_video


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Traffic Analytics | AI Monitoring",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# CUSTOM PROFESSIONAL UI
# ==================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stHeader"] {
        background-color: #0b1120;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263449;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 2px;
    }

    .main-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 24px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .section-description {
        color: #94a3b8;
        font-size: 13px;
        margin-top: -10px;
        margin-bottom: 18px;
    }

    .upload-card {
        background: linear-gradient(
            135deg,
            #17243a,
            #111c30
        );
        border: 1px solid #30435e;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 15px;
    }

    .upload-heading {
        font-size: 19px;
        font-weight: 700;
        color: #f8fafc;
    }

    .upload-description {
        font-size: 13px;
        color: #94a3b8;
    }

    .metric-card {
        background: #131f32;
        border: 1px solid #263b55;
        border-radius: 13px;
        padding: 17px;
        min-height: 112px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 29px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-caption {
        color: #64748b;
        font-size: 11px;
        margin-top: 3px;
    }

    .info-card {
        background: #131f32;
        border: 1px solid #263b55;
        border-radius: 13px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .status-text {
        color: #94a3b8;
        font-size: 12px;
    }

    hr {
        border-color: #263449;
    }

    div[data-testid="stMetric"] {
        background: #131f32;
        border: 1px solid #263b55;
        padding: 15px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    .stButton > button {
        border-radius: 9px;
        font-weight: 700;
        min-height: 45px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# DIRECTORIES
# ==================================================

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# SESSION STATE
# ==================================================

if "summary" not in st.session_state:
    st.session_state["summary"] = None

if "video_path" not in st.session_state:
    st.session_state["video_path"] = None

if "csv_path" not in st.session_state:
    st.session_state["csv_path"] = None

if "input_video_name" not in st.session_state:
    st.session_state["input_video_name"] = None


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def read_csv_file(csv_path):

    csv_path = Path(csv_path)

    if not csv_path.exists():
        return []

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        return list(csv.DictReader(file))


def convert_video_to_h264(input_path):

    input_path = Path(input_path)

    output_path = input_path.with_name(
        input_path.stem + "_browser.mp4"
    )

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

        if (
            result.returncode == 0
            and output_path.exists()
            and output_path.stat().st_size > 0
        ):
            return output_path

    except Exception:
        return None

    return None


def get_average_speed(events):

    speeds = []

    for event in events:

        try:

            speed = float(event.get("speed_kmh", ""))

            if speed >= 0:
                speeds.append(speed)

        except (ValueError, TypeError):
            continue

    if not speeds:
        return None

    return sum(speeds) / len(speeds)


def get_class_count(vehicle_counts, class_name):

    return vehicle_counts.get(
        class_name,
        vehicle_counts.get(str(class_name).lower(), 0),
    )


def create_metric(label, value, caption=""):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## 🚦 Traffic AI")

    st.caption("Intelligent Traffic Monitoring System")

    st.divider()

    st.markdown("### SYSTEM INFORMATION")

    st.write("**Detection:** YOLO11n")
    st.write("**Tracking:** ByteTrack")
    st.write("**Counting:** Line Crossing")
    st.write("**Speed:** Perspective Calibration")

    st.divider()

    st.markdown("### DETECTION CLASSES")

    for class_name in CLASS_NAMES.values():
        st.write(f"• {str(class_name).title()}")

    st.divider()

    st.caption(
        "AI-powered vehicle detection, tracking, "
        "counting and traffic analysis."
    )


# ==================================================
# MAIN HEADER
# ==================================================

st.markdown(
    """
    <div class="main-title">
        🚦 Traffic Analytics Dashboard
    </div>

    <div class="main-subtitle">
        AI-powered vehicle detection, tracking,
        counting and speed monitoring
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================

# ==================================================
# SECTION 1: VIDEO UPLOAD + LIVE DETECTION
# ==================================================

st.markdown(
    '<div class="section-title">01 / Traffic Video Analysis</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">'
    'Upload traffic footage and watch AI detection in real time.'
    '</div>',
    unsafe_allow_html=True,
)

upload_col, preview_col = st.columns([1, 1.35], gap="large")

with upload_col:

    with st.container(border=True):

        st.markdown("### ?? Upload Traffic Video")

        st.markdown(
            "Supported formats: MP4, AVI, MOV and other configured formats."
        )

        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=[
                extension.replace(".", "")
                for extension in SUPPORTED_VIDEO_FORMATS
            ],
            label_visibility="collapsed",
        )

        if uploaded_video is not None:

            file_size_mb = len(uploaded_video.getbuffer()) / (
                1024 * 1024
            )

            st.success(f"Selected: {uploaded_video.name}")
            st.info(f"File size: {file_size_mb:.2f} MB")

            start_analysis = st.button(
                "? Start Traffic Analysis",
                type="primary",
                use_container_width=True,
            )

            if start_analysis:

                unique_id = uuid.uuid4().hex[:8]

                input_extension = Path(
                    uploaded_video.name
                ).suffix.lower()

                if not input_extension:
                    input_extension = ".mp4"

                input_path = VIDEO_DIR / (
                    f"input_{unique_id}{input_extension}"
                )

                output_path = OUTPUT_DIR / (
                    f"tracking_{unique_id}.mp4"
                )

                csv_path = OUTPUT_DIR / (
                    f"events_{unique_id}.csv"
                )

                with open(input_path, "wb") as file:
                    file.write(uploaded_video.getbuffer())

                st.session_state["summary"] = None
                st.session_state["video_path"] = None
                st.session_state["csv_path"] = None

                status = st.empty()

                # Create the preview placeholder before processing starts.
                preview_placeholder = st.empty()

                def update_live_preview(frame):
                    preview_placeholder.image(
                        frame[:, :, ::-1],
                        channels="RGB",
                        use_container_width=True,
                    )

                try:

                    with st.spinner(
                        "Running YOLO11 detection, ByteTrack tracking "
                        "and speed estimation..."
                    ):

                        status.info(
                            "Processing traffic video..."
                        )

                        summary = process_video(
                            str(input_path),
                            str(output_path),
                            str(csv_path),
                            frame_callback=update_live_preview,
                        )

                        status.info(
                            "Preparing browser-compatible output video..."
                        )

                        browser_video = convert_video_to_h264(
                            output_path
                        )

                        if browser_video is not None:
                            final_video = browser_video
                            status.success(
                                "Traffic analysis completed successfully!"
                            )
                        else:
                            final_video = output_path
                            status.warning(
                                "Analysis completed. Browser conversion "
                                "was unavailable; download the original video."
                            )

                        st.session_state["summary"] = summary
                        st.session_state["video_path"] = str(final_video)
                        st.session_state["csv_path"] = str(csv_path)
                        st.session_state["input_video_name"] = uploaded_video.name

                    st.rerun()

                except Exception as error:

                    status.error(
                        f"Traffic analysis failed: {error}"
                    )

                    st.exception(error)


with preview_col:

    with st.container(border=True):

        st.markdown("### ?? Live Detection Preview")

        preview_placeholder = st.empty()

        saved_video = st.session_state.get("video_path")
        saved_summary = st.session_state.get("summary")

        if saved_summary is not None and saved_video:
            if Path(saved_video).exists():
                st.success("Detection completed")
                st.video(saved_video)
            else:
                preview_placeholder.info(
                    "The output video is not available."
                )
        else:
            preview_placeholder.info(
                "Your annotated traffic video will appear here "
                "while AI detection is running."
            )


# ==================================================
# SECTION 2: PROCESSED OUTPUT VIDEO
# ==================================================

summary = st.session_state.get("summary")

if summary is not None:

    st.divider()

    st.markdown(
        '<div class="section-title">02 / Processed Output Video</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Watch the annotated video with vehicle detection, tracking IDs '
        'and speed labels.'
        '</div>',
        unsafe_allow_html=True,
    )

    video_path = Path(
        st.session_state["video_path"]
    )

    if video_path.exists():

        with st.container(border=True):

            st.video(str(video_path))

            col1, col2 = st.columns([3, 1])

            col1.caption(
                f"Processed file: {video_path.name}"
            )

            with open(video_path, "rb") as file:

                col2.download_button(
                    label="⬇ Download Video",
                    data=file.read(),
                    file_name=video_path.name,
                    mime="video/mp4",
                    use_container_width=True,
                )

    else:

        st.error("Processed video file not found.")


    # ==================================================
    # SECTION 3: ANALYTICS DASHBOARD
    # ==================================================

    st.divider()

    st.markdown(
        '<div class="section-title">03 / Traffic Analytics</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Vehicle statistics, traffic direction, speed estimates '
        'and crossing event analysis.'
        '</div>',
        unsafe_allow_html=True,
    )

    vehicle_counts = summary.get(
        "vehicle_counts", {}
    )

    direction_counts = summary.get(
        "direction_counts", {}
    )

    total_vehicles = summary.get(
        "total_vehicles", 0
    )

    csv_path = Path(
        st.session_state["csv_path"]
    )

    events = read_csv_file(csv_path)

    average_speed = get_average_speed(events)


    # --------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------

    st.markdown("#### Overall Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        create_metric(
            "Total Vehicles",
            total_vehicles,
            "Unique line crossings",
        )

    with col2:
        create_metric(
            "UP Direction",
            direction_counts.get("UP", 0),
            "Vehicles moving UP",
        )

    with col3:
        create_metric(
            "DOWN Direction",
            direction_counts.get("DOWN", 0),
            "Vehicles moving DOWN",
        )

    with col4:

        speed_display = (
            f"{average_speed:.1f} km/h"
            if average_speed is not None
            else "N/A"
        )

        create_metric(
            "Average Speed",
            speed_display,
            "Average of available crossing-event estimates",
        )


    # --------------------------------------------------
    # VEHICLE CLASS CARDS
    # --------------------------------------------------

    st.markdown("#### Vehicle Classification")

    class_names = list(CLASS_NAMES.values())

    class_columns = st.columns(
        max(1, len(class_names))
    )

    for index, class_name in enumerate(class_names):

        count = get_class_count(
            vehicle_counts,
            class_name,
        )

        with class_columns[index]:

            create_metric(
                str(class_name).title(),
                count,
                "Detected crossing vehicles",
            )


    # --------------------------------------------------
    # CHARTS
    # --------------------------------------------------

    st.divider()

    st.markdown("#### Traffic Insights")

    chart_col1, chart_col2 = st.columns(2)

    # VEHICLE DISTRIBUTION

    with chart_col1:

        st.markdown("**Vehicle Distribution**")

        class_values = [
            get_class_count(vehicle_counts, name)
            for name in class_names
        ]

        if sum(class_values) > 0:

            class_chart = go.Figure(
                data=[
                    go.Bar(
                        x=[
                            str(name).title()
                            for name in class_names
                        ],
                        y=class_values,
                        text=class_values,
                        textposition="auto",
                    )
                ]
            )

            class_chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="#131f32",
                plot_bgcolor="#131f32",
                font=dict(color="#e5e7eb"),
                xaxis_title="Vehicle Type",
                yaxis_title="Vehicle Count",
                margin=dict(l=20, r=20, t=25, b=30),
                height=340,
            )

            st.plotly_chart(
                class_chart,
                use_container_width=True,
            )

        else:

            st.info("No vehicle crossing data available.")


    # DIRECTION DISTRIBUTION

    with chart_col2:

        st.markdown("**Traffic Direction Distribution**")

        directions = ["UP", "DOWN"]

        direction_values = [
            direction_counts.get("UP", 0),
            direction_counts.get("DOWN", 0),
        ]

        if sum(direction_values) > 0:

            direction_chart = go.Figure(
                data=[
                    go.Pie(
                        labels=directions,
                        values=direction_values,
                        hole=0.55,
                    )
                ]
            )

            direction_chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="#131f32",
                plot_bgcolor="#131f32",
                font=dict(color="#e5e7eb"),
                margin=dict(l=20, r=20, t=25, b=25),
                height=340,
            )

            st.plotly_chart(
                direction_chart,
                use_container_width=True,
            )

        else:

            st.info("No direction crossing data available.")


    # --------------------------------------------------
    # SPEED ANALYSIS
    # --------------------------------------------------

    st.divider()

    st.markdown("#### Vehicle Speed Analysis")

    speed_records = []

    for event in events:

        try:

            speed = float(event.get("speed_kmh", ""))

            if speed >= 0:

                speed_records.append({
                    "track_id": event.get("track_id", ""),
                    "vehicle_class": event.get("vehicle_class", ""),
                    "speed_kmh": speed,
                })

        except (ValueError, TypeError):
            continue

    if speed_records:

        speed_chart = go.Figure(
            data=[
                go.Bar(
                    x=[
                        f"ID {record['track_id']}"
                        for record in speed_records
                    ],
                    y=[
                        record["speed_kmh"]
                        for record in speed_records
                    ],
                    text=[
                        f"{record['speed_kmh']:.1f}"
                        for record in speed_records
                    ],
                    textposition="auto",
                )
            ]
        )

        speed_chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="#131f32",
            plot_bgcolor="#131f32",
            font=dict(color="#e5e7eb"),
            xaxis_title="Vehicle Track ID",
            yaxis_title="Estimated Speed (km/h)",
            margin=dict(l=20, r=20, t=25, b=30),
            height=350,
        )

        st.plotly_chart(
            speed_chart,
            use_container_width=True,
        )

        st.caption(
            "Speed chart uses available crossing-event estimates only. "
            "These values depend on camera-specific calibration."
        )

    else:

        st.info(
            "No speed estimates are available in the crossing events."
        )


    # --------------------------------------------------
    # VEHICLE CROSSING EVENTS
    # --------------------------------------------------

    st.divider()

    st.markdown("#### Vehicle Crossing Event History")

    if events:

        st.dataframe(
            events,
            use_container_width=True,
            hide_index=True,
        )

        with open(csv_path, "rb") as file:

            st.download_button(
                label="⬇ Download Vehicle Events CSV",
                data=file.read(),
                file_name="vehicle_crossing_events.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:

        st.info(
            "No vehicle crossing events were recorded."
        )


    # --------------------------------------------------
    # FOOTNOTE
    # --------------------------------------------------

    st.divider()

    st.caption(
        "Vehicle counts represent unique tracked vehicles crossing "
        "the virtual counting line. Speed estimates depend on "
        "camera-specific perspective calibration and assumed road "
        "dimensions. They are not verified real-world speeds."
    )


else:

    st.divider()

    st.info(
        "Upload a traffic video above and click "
        "'Start Traffic Analysis' to generate the output video "
        "and analytics dashboard."
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748b;
        padding:25px 0 10px 0;
        font-size:12px;
    ">
        TRAFFIC ANALYTICS SYSTEM · YOLO11n · ByteTrack · OpenCV
    </div>
    """,
    unsafe_allow_html=True,
)
