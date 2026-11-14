
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

    /* Individual KPI card palettes */
    .metric-card {
        border-radius: 16px;
        padding: 19px;
        min-height: 124px;
        border: 1px solid rgba(255,255,255,.13);
        box-shadow: 0 7px 20px rgba(0,0,0,.18);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 26px rgba(0,0,0,.25);
    }
    .metric-blue   { background: linear-gradient(135deg,#173B67,#1D4F91); }
    .metric-green  { background: linear-gradient(135deg,#155B50,#167D68); }
    .metric-amber  { background: linear-gradient(135deg,#69421C,#98601F); }
    .metric-purple { background: linear-gradient(135deg,#51356F,#704B99); }
    .metric-cyan   { background: linear-gradient(135deg,#15516A,#167A91); }
    .metric-orange { background: linear-gradient(135deg,#713B2B,#A34E32); }
    .metric-red    { background: linear-gradient(135deg,#702D3A,#A83D4D); }
    .metric-teal   { background: linear-gradient(135deg,#174E5B,#167C83); }
    .metric-slate  { background: linear-gradient(135deg,#334155,#475569); }
    .metric-label {
        color: #E2E8F0;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .8px;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 28px;
        line-height: 1.2;
        font-weight: 800;
        margin-top: 11px;
        overflow-wrap: anywhere;
    }
    .metric-caption {
        color: #D0DCEB;
        font-size: 11px;
        margin-top: 6px;
    }

    .dashboard-panel {
        background: linear-gradient(145deg,#142238,#101b2e);
        border: 1px solid #2b405b;
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .section-title { letter-spacing: .2px; }
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border: 1px solid #2b405b;
        border-radius: 12px;
        overflow: hidden;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 700;
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


    /* Compact single-screen analytics dashboard (scoped to Section 3 only) */
    .st-key-analytics_dashboard {
        background: #0f1b2d;
        color: #e5e7eb;
        border: 1px solid #263b55;
        border-radius: 12px;
        padding: 10px 14px 12px 14px;
        margin-top: 8px;
    }
    .st-key-analytics_dashboard [data-testid="stVerticalBlock"] {
        gap: 0.55rem;
    }
    .st-key-analytics_dashboard .section-title {
        color: #f8fafc !important;
        font-size: 19px !important;
        margin-top: 2px !important;
        margin-bottom: 4px !important;
    }
    .st-key-analytics_dashboard h3,
    .st-key-analytics_dashboard h4,
    .st-key-analytics_dashboard p,
    .st-key-analytics_dashboard label,
    .st-key-analytics_dashboard [data-testid="stMarkdownContainer"] {
        color: #e2e8f0;
    }
    .st-key-analytics_dashboard .section-description {
        color: #94a3b8 !important;
        font-size: 11px !important;
        margin: 0 0 5px 0 !important;
    }
    .st-key-analytics_dashboard .metric-card {
        min-height: 76px !important;
        height: 76px;
        padding: 9px 11px !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 7px rgba(15, 23, 42, .08) !important;
    }
    .st-key-analytics_dashboard .metric-label {
        font-size: 9px !important;
        letter-spacing: .45px !important;
    }
    .st-key-analytics_dashboard .metric-value {
        font-size: 21px !important;
        line-height: 1.1 !important;
        margin-top: 6px !important;
    }
    .st-key-analytics_dashboard .metric-caption {
        font-size: 9px !important;
        margin-top: 3px !important;
    }
    .st-key-analytics_dashboard [data-testid="stMetric"] {
        padding: 8px !important;
        border-radius: 9px !important;
    }
    .st-key-analytics_dashboard [data-testid="stDataFrame"] {
        font-size: 11px;
    }
    .st-key-analytics_dashboard hr {
        margin: 5px 0 !important;
        border-color: #263b55 !important;
    }
    .st-key-analytics_dashboard [data-testid="stPlotlyChart"] {
        background: #111f33;
        border: 1px solid #2b405b;
        border-radius: 9px;
        padding: 3px;
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


def get_average_speed_kmh(events):

    speeds = []

    for event in events:

        try:

            if event.get("speed_type") not in (
                "",
                "approximate_perspective",
                None,
            ):
                continue

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


def create_metric(label, value, caption="", color="blue"):

    allowed_colors = {
        "blue", "green", "amber", "purple", "cyan",
        "orange", "red", "teal", "slate",
    }
    if color not in allowed_colors:
        color = "blue"

    # Escape text values before placing them in the HTML card.
    import html
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_caption = html.escape(str(caption))

    st.markdown(
        f"""
        <div class="metric-card metric-{color}">
            <div class="metric-label">{safe_label}</div>
            <div class="metric-value">{safe_value}</div>
            <div class="metric-caption">{safe_caption}</div>
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
    with st.container(key="analytics_dashboard"):
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

        average_speed = get_average_speed_kmh(events)


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
                color="blue",
            )

        with col2:
            create_metric(
                "UP Direction",
                direction_counts.get("UP", 0),
                "Vehicles moving UP",
                color="green",
            )

        with col3:
            create_metric(
                "DOWN Direction",
                direction_counts.get("DOWN", 0),
                "Vehicles moving DOWN",
                color="amber",
            )

        with col4:

            speed_display = (
                f"~{average_speed:.1f} km/h"
                if average_speed is not None
                else "N/A"
            )

            create_metric(
                "Average Approximate Speed",
                speed_display,
                "Perspective-based estimated speed",
                color="purple",
            )


        # --------------------------------------------------
        # TRAFFIC CONGESTION ANALYSIS
        # --------------------------------------------------

        st.divider()

        st.markdown("#### Traffic Congestion Monitoring")

        congestion_data = summary.get("congestion", {})
        congestion_history = summary.get("congestion_history", [])

        if congestion_data and congestion_data.get("zones"):

            congestion_level = congestion_data.get(
                "overall_level", "Unavailable"
            )

            congestion_score = congestion_data.get(
                "overall_score", 0.0
            )

            current_tracked = congestion_data.get(
                "total_tracked_vehicles", 0
            )

            most_congested = congestion_data.get(
                "most_congested_zone", "Unavailable"
            )

            congestion_col1, congestion_col2, congestion_col3 = st.columns(3)

            with congestion_col1:
                create_metric(
                    "Overall Congestion",
                    congestion_level,
                    f"Score: {congestion_score}/100",
                    color="red",
                )

            with congestion_col2:
                create_metric(
                    "Currently Tracked Vehicles",
                    current_tracked,
                    "Vehicles currently inside calibrated road zones",
                    color="cyan",
                )

            with congestion_col3:
                create_metric(
                    "Most Congested Zone",
                    most_congested.replace("_", " ").title(),
                    "Highest zone congestion score",
                    color="orange",
                )

            st.caption(
                "Congestion is a heuristic based on zone occupancy and "
                "tracked vehicle movement. Road dimensions and zone capacity "
                "are assumptions, not verified traffic measurements."
            )

            # ZONE-WISE CONGESTION TABLE

            st.markdown("**Zone-wise Congestion**")

            zone_rows = []

            for zone_id, zone in congestion_data["zones"].items():
                zone_rows.append({
                    "Road": zone.get("road", "").replace("_", " ").title(),
                    "Zone": zone.get("zone", ""),
                    "Vehicles": zone.get("vehicle_count", 0),
                    "Moving": zone.get("moving_vehicles", 0),
                    "Stationary": zone.get("stationary_vehicles", 0),
                    "Occupancy (%)": zone.get("occupancy_percent", 0),
                    "Score": zone.get("congestion_score", 0),
                    "Level": zone.get("level", "Unknown"),
                    "Congested Duration (s)": zone.get(
                        "congestion_duration_seconds", 0
                    ),
                })

            st.dataframe(
                zone_rows,
                use_container_width=True,
                hide_index=True,
            )

            # CONGESTION HISTORY CHART

            st.markdown("**Congestion History**")

            if congestion_history:

                history_chart = go.Figure()

                history_chart.add_trace(
                    go.Scatter(
                        x=[
                            item["timestamp_seconds"]
                            for item in congestion_history
                        ],
                        y=[
                            item["overall_score"]
                            for item in congestion_history
                        ],
                        mode="lines+markers",
                        name="Overall Congestion",
                        line=dict(color="#38BDF8", width=3),
                        marker=dict(color="#A78BFA", size=6),
                    )
                )

                history_chart.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111f33",
                    plot_bgcolor="#111f33",
                    font=dict(color="#e5e7eb"),
                    xaxis_title="Video Time (seconds)",
                    yaxis_title="Congestion Score",
                    yaxis=dict(range=[0, 100]),
                    margin=dict(l=20, r=20, t=25, b=35),
                    height=210,
                )

                st.plotly_chart(
                    history_chart,
                    use_container_width=True,
                )

            else:
                st.info("No congestion history samples available.")

        else:
            st.info(
                "Congestion results are unavailable. "
                "Process a video with congestion analysis enabled."
            )


        # --------------------------------------------------
        # VEHICLE CLASS CARDS
        # --------------------------------------------------

        st.markdown("#### Vehicle Classification")

        class_names = list(CLASS_NAMES.values())

        class_columns = st.columns(
            max(1, len(class_names))
        )

        class_palette = ["blue", "green", "amber", "purple", "cyan", "orange", "teal", "slate"]

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
                    color=class_palette[index % len(class_palette)],
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
                            marker=dict(color=["#3B82F6", "#10B981", "#F59E0B", "#A78BFA", "#22D3EE", "#F97316", "#14B8A6", "#94A3B8"][:len(class_values)]),
                        )
                    ]
                )

                class_chart.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111f33",
                    plot_bgcolor="#111f33",
                    font=dict(color="#e5e7eb"),
                    xaxis_title="Vehicle Type",
                    yaxis_title="Vehicle Count",
                    margin=dict(l=20, r=20, t=25, b=30),
                    height=205,
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
                            marker=dict(colors=["#10B981", "#F59E0B"], line=dict(color="#101B2E", width=3)),
                        )
                    ]
                )

                direction_chart.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111f33",
                    plot_bgcolor="#111f33",
                    font=dict(color="#e5e7eb"),
                    margin=dict(l=20, r=20, t=25, b=25),
                    height=205,
                )

                st.plotly_chart(
                    direction_chart,
                    use_container_width=True,
                )

            else:

                st.info("No direction crossing data available.")


        # --------------------------------------------------
        # COMPACT SPEED + CROSSING EVENTS ROW
        # --------------------------------------------------

        speed_records = []

        for event in events:
            try:
                if event.get("speed_type") not in (
                    "",
                    "approximate_perspective",
                    None,
                ):
                    continue

                speed = float(event.get("speed_kmh", ""))

                if speed >= 0:
                    speed_records.append({
                        "track_id": event.get("track_id", ""),
                        "vehicle_class": event.get("vehicle_class", ""),
                        "speed_kmh": speed,
                    })

            except (ValueError, TypeError):
                continue

        # Keep both panels in one compact horizontal row.
        speed_col, events_col = st.columns([0.95, 1.45], gap="small")

        with speed_col:
            st.markdown("#### Vehicle Speed")

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
                            marker=dict(color="#818CF8"),
                        )
                    ]
                )

                speed_chart.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111f33",
                    plot_bgcolor="#111f33",
                    font=dict(color="#e5e7eb", size=9),
                    xaxis_title="Track ID",
                    yaxis_title="Speed km/h",
                    margin=dict(l=8, r=8, t=12, b=28),
                    height=225,
                    showlegend=False,
                    bargap=0.35,
                )

                speed_chart.update_xaxes(
                    tickfont=dict(size=8),
                    gridcolor="#263b55",
                    zerolinecolor="#334155",
                )
                speed_chart.update_yaxes(
                    tickfont=dict(size=8),
                    gridcolor="#263b55",
                    zerolinecolor="#334155",
                )

                st.plotly_chart(
                    speed_chart,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                st.caption(
                    "Approximate perspective-based speed; not certified "
                    "real-world measurements."
                )
            else:
                st.info("No approximate speed estimates available.")

        with events_col:
            st.markdown("#### Vehicle Crossing Events")

            if events:
                st.dataframe(
                    events,
                    use_container_width=True,
                    hide_index=True,
                    height=225,
                )

                with open(csv_path, "rb") as file:
                    st.download_button(
                        label="⬇ Download Events CSV",
                        data=file.read(),
                        file_name="vehicle_crossing_events.csv",
                        mime="text/csv",
                        type="secondary",
                    )
            else:
                st.info("No vehicle crossing events were recorded.")


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
