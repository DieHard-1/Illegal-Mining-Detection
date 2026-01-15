"""
Illegal Mining Detection Dashboard
Physics-Guided, Unsupervised, Evidence-Driven Interface

Run with:
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Illegal Mining Detection System",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

TOTAL_MINES = 506  

# =============================================================================
# PATH CONFIG
# =============================================================================
BASE_DATA = "../data"
BASE_OUTPUTS = "../outputs"

STEP1_SENTINEL = os.path.join(BASE_DATA, "step1_sentinel")
STEP4_CHANGES = os.path.join(BASE_DATA, "step4_changes")
STEP5_REGIONS = os.path.join(BASE_DATA, "step5_regions")

ZONES_CSV = os.path.join(BASE_OUTPUTS, "validation", "step2_zone_assignments.csv")
VIOLATIONS_CSV = os.path.join(BASE_OUTPUTS, "final", "alerts", "violation_alerts.csv")
METRICS_JSON = os.path.join(BASE_OUTPUTS, "final", "metrics", "evaluation_metrics.json")
SUMMARY_IMG = os.path.join(BASE_OUTPUTS, "final", "reports", "summary_statistics.png")

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
.metric-box {background:#1f77b4;color:white;padding:15px;border-radius:8px;text-align:center;}
.info-box {background:#e8f4f8;padding:15px;border-radius:8px;border-left:4px solid #1f77b4;margin:10px 0;}
.warning-box {background:#fff4e6;padding:15px;border-radius:8px;border-left:4px solid #ff9800;margin:10px 0;}
.success-box {background:#e8f5e9;padding:15px;border-radius:8px;border-left:4px solid #4caf50;margin:10px 0;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_all_data():
    data = {}

    data["metrics"] = json.load(open(METRICS_JSON)) if os.path.exists(METRICS_JSON) else {}
    data["violations"] = pd.read_csv(VIOLATIONS_CSV) if os.path.exists(VIOLATIONS_CSV) else pd.DataFrame()
    data["zones"] = pd.read_csv(ZONES_CSV) if os.path.exists(ZONES_CSV) else pd.DataFrame()

    if os.path.exists(STEP5_REGIONS):
        data["available_mines"] = sorted(
            f.replace("_regions.json", "")
            for f in os.listdir(STEP5_REGIONS)
            if f.endswith("_regions.json")
        )
    else:
        data["available_mines"] = []

    return data


def load_mine_regions(mine_id):
    path = os.path.join(STEP5_REGIONS, f"{mine_id}_regions.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

# =============================================================================
# HELPERS
# =============================================================================
def explain_growth_consistency():
    st.markdown("""
    <div class="info-box">
    <strong>Growth Consistency</strong><br>
    Measures how often excavated area increases over time.
    <ul>
      <li>&gt; 70% → persistent mining</li>
      <li>40–70% → mixed / seasonal</li>
      <li>&lt; 40% → likely noise</li>
    </ul>
    Mining grows; noise oscillates.
    </div>
    """, unsafe_allow_html=True)


def first_violation_index(df, zone):
    """First period with confirmed excavation in NO-GO zone."""
    if zone != "NOGO":
        return None
    for i, area in enumerate(df["effective_area_m2"].values):
        if area > 0:
            return i
    return None

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    data = load_all_data()

    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "🏠 Home",
            "🔍 Mine Explorer",
            "📈 Temporal Growth & Legality",
            "🗺️ Spatial Evidence",
            "⚠️ Violations Dashboard",
            "📊 System Metrics"
        ]
    )

    # =========================================================================
    # HOME
    # =========================================================================
    if page == "🏠 Home":
        st.title("🏔️ Illegal Mining Detection System")

        det = data["metrics"].get("detection_metrics", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Mines", TOTAL_MINES)
        c2.metric("Mines with Excavation", det.get("mines_with_activity", "NA"))
        c3.metric("NO-GO Violations", det.get("mines_with_violations", "NA"))
        c4.metric("Detection Rate", f"{det.get('detection_rate', 0):.1%}")

        st.markdown("""
        <div class="info-box">
        This system detects mining as a <b>persistent physical land transformation</b>.
        Every alert is backed by pixel-level change, spatial coherence,
        temporal growth, and explicit legal rules.
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # MINE EXPLORER
    # =========================================================================
    elif page == "🔍 Mine Explorer":
        st.title("🔍 Mine Explorer")

        mine_id = st.selectbox("Select Mine", data["available_mines"])
        regions = load_mine_regions(mine_id)

        zone = "UNKNOWN"
        if not data["zones"].empty:
            z = data["zones"][data["zones"]["mine_id"] == mine_id]
            if not z.empty:
                zone = z.iloc[0]["zone_type"].upper()

        excavated = regions and regions.get("total_growth_m2", 0) > 0
        violation = zone == "NOGO" and excavated

        state = "NO ACTIVITY"
        if excavated and zone == "LEGAL":
            state = "LEGAL MINING"
        if excavated and zone == "NOGO":
            state = "ILLEGAL MINING"

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Zone", zone)
        c2.metric("Mining State", state)
        c3.metric("Excavation Detected", "YES" if excavated else "NO")

        if regions:
            c4.metric("Total Growth (m²)", f"{regions.get('total_growth_m2',0):,.0f}")
            c5.metric("Growth Consistency", f"{regions.get('growth_consistency',0):.1%}")

        explain_growth_consistency()

    # =========================================================================
    # TEMPORAL GROWTH + FIRST VIOLATION MARKER
    # =========================================================================
    elif page == "📈 Temporal Growth & Legality":
        st.title("📈 Excavation Growth with Legal Context")

        mine_id = st.selectbox("Select Mine", data["available_mines"])
        regions = load_mine_regions(mine_id)

        if not regions or "temporal_regions" not in regions:
            st.warning("No temporal growth data.")
            return

        df = pd.DataFrame(regions["temporal_regions"])
        df["period"] = range(len(df))
        df["effective_area_m2"] = np.maximum.accumulate(df["total_area_m2"].values)

        zone = "UNKNOWN"
        if not data["zones"].empty:
            z = data["zones"][data["zones"]["mine_id"] == mine_id]
            if not z.empty:
                zone = z.iloc[0]["zone_type"].upper()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["period"],
            y=df["total_area_m2"]/1e3,
            name="Raw Detected Area",
            mode="lines+markers",
            line=dict(color="orange", dash="dot")
        ))

        fig.add_trace(go.Scatter(
            x=df["period"],
            y=df["effective_area_m2"]/1e3,
            name="Confirmed Excavation Envelope",
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color="red", width=3)
        ))

        fv = first_violation_index(df, zone)
        if fv is not None:
            fig.add_vline(x=fv, line_width=3, line_dash="dash", line_color="black")
            fig.add_annotation(
                x=fv,
                y=df["effective_area_m2"].max()/1e3,
                text="First NO-GO Violation",
                showarrow=True,
                arrowhead=2,
                yshift=10
            )

        fig.update_layout(
            xaxis_title="Time Period",
            yaxis_title="Area (×1000 m²)",
            title=f"Mine {mine_id} – Excavation Growth"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        Orange curve may fluctuate due to clouds or masking.
        Red envelope enforces monotonic mining growth.
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # SPATIAL EVIDENCE 
    # =========================================================================
    elif page == "🗺️ Spatial Evidence":
        st.title("🗺️ Spatial Excavation Evidence")

        mine_id = st.selectbox("Select Mine", data["available_mines"])
        change_dir = os.path.join(STEP4_CHANGES, mine_id)

        if not os.path.exists(change_dir):
            st.warning("No change detection data.")
            return

        files = sorted(f for f in os.listdir(change_dir) if f.endswith(".npz"))
        if not files:
            st.warning("No excavation masks available.")
            return

        idx = st.slider("Select Time Comparison", 0, len(files)-1)
        change = np.load(os.path.join(change_dir, files[idx]))
        mask = change["excavation_mask"]

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(mask, cmap="Reds")
            ax.set_title("Detected Excavation Pixels (Red)")
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            st.markdown("""
            <div class="info-box">
            Each red pixel is an excavation candidate from Step 4.
            Only spatially coherent and persistent regions survive Step 5.
            </div>
            """, unsafe_allow_html=True)
            st.metric("Excavated Pixels", int(mask.sum()))

    # =========================================================================
    # VIOLATIONS
    # =========================================================================
    elif page == "⚠️ Violations Dashboard":
        st.title("⚠️ Illegal Mining Alerts")

        if data["violations"].empty:
            st.success("No violations detected.")
            return

        st.dataframe(data["violations"], use_container_width=True)

    # =========================================================================
    # SYSTEM METRICS
    # =========================================================================
    elif page == "📊 System Metrics":
        st.title("📊 System-Level Metrics")

        det = data["metrics"].get("detection_metrics", {})
        temp = data["metrics"].get("temporal_consistency", {})

        c1, c2, c3 = st.columns(3)
        c1.metric("Detection Rate", f"{det.get('detection_rate',0):.1%}")
        c2.metric("Violation Rate (NO-GO)", f"{det.get('violation_rate_in_nogo',0):.1%}")
        c3.metric("Avg Growth Consistency", f"{temp.get('avg_growth_consistency',0):.1%}")

        if os.path.exists(SUMMARY_IMG):
            st.image(Image.open(SUMMARY_IMG), use_container_width=True)

# =============================================================================
if __name__ == "__main__":
    main()
