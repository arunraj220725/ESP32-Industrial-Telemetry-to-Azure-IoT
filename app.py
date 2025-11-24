import streamlit as st
import time
import random

st.set_page_config(page_title="Simple IoT Dashboard", layout="wide")

# ---------------------------
# Session State Initialization
# ---------------------------
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["time", "current", "voltage", "accel"])

if "running" not in st.session_state:
    st.session_state.running = False


# ---------------------------
# Fake Telemetry Generator
# ---------------------------
def generate_fake_data():
    return {
        "time": time.strftime("%H:%M:%S"),
        "current": round(random.uniform(1.0, 4.5), 2)
        "accel": round(random.uniform(5.0, 15.0), 2),
    }


# ---------------------------
# UI Layout
# ---------------------------
st.title("🏭 Simple Industrial IoT Dashboard")

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Monitoring"):

with col2:
    if st.button("Stop Monitoring"):
        st.session_state.running = False


# ---------------------------
# Update Loop
# ---------------------------
if st.session_state.running:
    new_row = generate_fake_data()
    st.session_state.data.loc[len(st.session_state.data)] = new_row

    # Keep only last 50 samples
    if len(st.session_state.data) > 50:
        st.session_state.data = st.session_state.data.tail(50)


# ---------------------------
# Display Latest Values
# ---------------------------
if len(st.session_state.data) > 0:
    latest = st.session_state.data.iloc[-1]

    st.subheader("📡 Latest Telemetry")
    c1, c2, c3 = st.columns(3)

    c1.metric("Current", f"{latest['current']} A")
    c2.metric("Voltage", f"{latest['voltage']} V")
    c3.metric("Vibration", f"{latest['accel']} m/s²")


# ---------------------------
# Plot Live Graph
# ---------------------------
if len(st.session_state.data) > 1:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=st.session_state.data["time"],
        y=st.session_state.data["current"],
        mode="lines+markers",
        name="Current"
    ))

    fig.add_trace(go.Scatter(
        x=st.session_state.data["time"],
        y=st.session_state.data["voltage"],
        mode="lines+markers",
        name="Voltage"
    ))

    fig.update_layout(
        title="Live Telemetry Chart",
        xaxis_title="Time",
        yaxis_title="Value",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# Auto-refresh
# ---------------------------
if st.session_state.running:
    time.sleep(1)
    st.rerun()
