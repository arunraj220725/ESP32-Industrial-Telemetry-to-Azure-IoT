import streamlit as st
import threading
from azure.eventhub import EventHubConsumerClient
import json
import logging
import time
from datetime import datetime
import queue
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
CONNECTION_STRING = "Endpoint=sb://ihsuprodpnres011dednamespace.servicebus.windows.net/;SharedAccessKeyName=iothubowner;SharedAccessKey=6ctipeK26laX1vp2RHXrpwnaccSjGkd06AIoTBjVbUo=;EntityPath=iothub-ehub-ble-scanne-55883412-d2d85e52dd"
CONSUMER_GROUP = "$Default"

# Initialize session state
if 'message_queue' not in st.session_state:
    st.session_state.message_queue = queue.Queue()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = "Not Started"
if 'latest_data' not in st.session_state:
    st.session_state.latest_data = {}
if 'total_received' not in st.session_state:
    st.session_state.total_received = 0
if 'receive_thread' not in st.session_state:
    st.session_state.receive_thread = None
if 'consumer_started' not in st.session_state:
    st.session_state.consumer_started = False
if 'last_processed_time' not in st.session_state:
    st.session_state.last_processed_time = 0
if 'data_frame' not in st.session_state:
    st.session_state.data_frame = pd.DataFrame()
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'stream_data' not in st.session_state:
    st.session_state.stream_data = {
        'current': [],
        'voltage': [],
        'vibration': [],
        'timestamps': []
    }

# Alert thresholds (industrial standards)
ALERT_THRESHOLDS = {
    'current_high': 4.0,  # Amps
    'current_low': 1.0,   # Amps
    'voltage_high': 280.0, # Volts
    'voltage_low': 180.0,  # Volts
    'accel_mag_high': 11.0 # m/s²
}

def on_event(partition_context, event):
    try:
        if event.body:
            # Try to parse as JSON
            try:
                message_body = event.body_as_json()
                save_message_to_temp(message_body)
                
            except json.JSONDecodeError:
                # Try as string
                raw_message = event.body_as_str()
                if "TELEMETRY: Successfully sent:" in raw_message:
                    try:
                        start_idx = raw_message.find('{')
                        end_idx = raw_message.rfind('}') + 1
                        if start_idx != -1 and end_idx != -1:
                            json_str = raw_message[start_idx:end_idx]
                            message_body = json.loads(json_str)
                            save_message_to_temp(message_body)
                    except Exception as e:
                        print(f"❌ Failed to extract JSON: {e}")
        
        partition_context.update_checkpoint(event)
        
    except Exception as e:
        print(f"❌ Error in on_event: {e}")

def save_message_to_temp(message):
    """Save message to temporary file"""
    try:
        import tempfile
        import os
        
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, "iot_messages.json")
        
        messages = []
        if os.path.exists(temp_file):
            try:
                with open(temp_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        messages = json.loads(content)
            except:
                messages = []
        
        messages.append({
            'timestamp': time.time(),
            'data': message
        })
        
        if len(messages) > 1000:
            messages = messages[-1000:]
        
        with open(temp_file, 'w') as f:
            json.dump(messages, f)
            
    except Exception as e:
        print(f"❌ Error saving message to temp: {e}")

def load_messages_from_temp():
    """Load messages from temporary file"""
    try:
        import tempfile
        import os
        
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, "iot_messages.json")
        
        if os.path.exists(temp_file):
            with open(temp_file, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"❌ Error loading messages from temp: {e}")
        return []

def on_error(partition_context, error):
    print(f"❌ Partition {partition_context.partition_id} error: {error}")

def receive_events():
    client = None
    try:
        client = EventHubConsumerClient.from_connection_string(
            conn_str=CONNECTION_STRING,
            consumer_group=CONSUMER_GROUP,
        )
        
        with client:
            client.receive(
                on_event=on_event,
                on_error=on_error,
                starting_position="-1",
                max_wait_time=60,
            )
            
    except Exception as e:
        print(f"❌ Error in receive_events: {e}")
    finally:
        if client:
            client.close()

def start_consumer():
    if st.session_state.receive_thread and st.session_state.receive_thread.is_alive():
        return
    
    st.session_state.receive_thread = threading.Thread(target=receive_events, daemon=True)
    st.session_state.receive_thread.start()
    st.session_state.consumer_started = True
    st.session_state.connection_status = "Data Link Active – Receiving Telemetry"

def check_alerts(data):
    """Check for alert conditions"""
    alerts = []
    
    if 'current' in data:
        if data['current'] > ALERT_THRESHOLDS['current_high']:
            alerts.append(f"⚠️ HIGH CURRENT: {data['current']:.2f}A (Threshold: {ALERT_THRESHOLDS['current_high']}A)")
        elif data['current'] < ALERT_THRESHOLDS['current_low']:
            alerts.append(f"⚠️ LOW CURRENT: {data['current']:.2f}A (Threshold: {ALERT_THRESHOLDS['current_low']}A)")
    
    if 'voltage' in data:
        if data['voltage'] > ALERT_THRESHOLDS['voltage_high']:
            alerts.append(f"⚠️ HIGH VOLTAGE: {data['voltage']:.1f}V (Threshold: {ALERT_THRESHOLDS['voltage_high']}V)")
        elif data['voltage'] < ALERT_THRESHOLDS['voltage_low']:
            alerts.append(f"⚠️ LOW VOLTAGE: {data['voltage']:.1f}V (Threshold: {ALERT_THRESHOLDS['voltage_low']}V)")
    
    if all(k in data for k in ['accel_x', 'accel_y', 'accel_z']):
        magnitude = (data['accel_x']**2 + data['accel_y']**2 + data['accel_z']**2)**0.5
        if magnitude > ALERT_THRESHOLDS['accel_mag_high']:
            alerts.append(f"⚠️ HIGH VIBRATION: {magnitude:.2f}m/s² (Threshold: {ALERT_THRESHOLDS['accel_mag_high']}m/s²)")
    
    return alerts

def create_vibration_stream_graph(df):
    """Create stream graph for vibration analysis"""
    if len(df) < 2:
        return go.Figure()
    
    # Create stream graph for acceleration components
    fig = go.Figure()
    
    # Add traces for each acceleration component
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
    
    if 'accel_x' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['accel_x'],
            mode='lines',
            name='Accel X',
            stackgroup='one',
            line=dict(width=0.5, color=colors[0]),
            fillcolor=colors[0]
        ))
    
    if 'accel_y' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['accel_y'],
            mode='lines',
            name='Accel Y',
            stackgroup='one',
            line=dict(width=0.5, color=colors[1]),
            fillcolor=colors[1]
        ))
    
    if 'accel_z' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['accel_z'],
            mode='lines',
            name='Accel Z',
            stackgroup='one',
            line=dict(width=0.5, color=colors[2]),
            fillcolor=colors[2]
        ))
    
    fig.update_layout(
        title="Vibration Signal Stream Graph",
        xaxis_title="Time (s)",
        yaxis_title="Acceleration (m/s²)",
        showlegend=True,
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='rgba(240,240,240,0.1)'
    )
    
    return fig

def create_current_voltage_stream_graph(df):
    """Create stream graph for current and voltage"""
    if len(df) < 2:
        return go.Figure()
    
    fig = go.Figure()
    
    # Normalize data for better visualization
    if 'current' in df.columns and 'voltage' in df.columns:
        current_norm = (df['current'] - df['current'].min()) / (df['current'].max() - df['current'].min())
        voltage_norm = (df['voltage'] - df['voltage'].min()) / (df['voltage'].max() - df['voltage'].min())
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=current_norm,
            mode='lines',
            name='Current (Normalized)',
            stackgroup='one',
            line=dict(width=0.5, color='#FFA726'),
            fillcolor='rgba(255, 167, 38, 0.6)'
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=voltage_norm,
            mode='lines',
            name='Voltage (Normalized)',
            stackgroup='one',
            line=dict(width=0.5, color='#42A5F5'),
            fillcolor='rgba(66, 165, 245, 0.6)'
        ))
    
    fig.update_layout(
        title="Current & Voltage Stream Analysis",
        xaxis_title="Time (s)",
        yaxis_title="Normalized Values",
        showlegend=True,
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='rgba(240,240,240,0.1)'
    )
    
    return fig

def create_spectrum_stream_graph(df):
    """Create frequency spectrum stream graph"""
    if len(df) < 10:
        return go.Figure()
    
    fig = go.Figure()
    
    # Calculate FFT for vibration analysis (simplified)
    if all(k in df.columns for k in ['accel_x', 'accel_y', 'accel_z']):
        # Use magnitude for spectrum analysis
        magnitude = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
        
        # Simple rolling window for spectrum-like visualization
        window_size = min(10, len(magnitude))
        spectrum_low = magnitude.rolling(window=window_size).mean()
        spectrum_high = magnitude.rolling(window=window_size).std()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=spectrum_low,
            mode='lines',
            name='Low Frequency',
            stackgroup='one',
            line=dict(width=0.5, color='#66BB6A'),
            fillcolor='rgba(102, 187, 106, 0.6)'
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=spectrum_high,
            mode='lines',
            name='High Frequency',
            stackgroup='one',
            line=dict(width=0.5, color='#AB47BC'),
            fillcolor='rgba(171, 71, 188, 0.6)'
        ))
    
    fig.update_layout(
        title="Vibration Spectrum Analysis",
        xaxis_title="Time (s)",
        yaxis_title="Frequency Components",
        showlegend=True,
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='rgba(240,240,240,0.1)'
    )
    
    return fig

def create_time_series_chart(df, column, title, color):
    """Create time series chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df[column],
        mode='lines',
        name=title,
        line=dict(color=color, width=2)
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=title,
        height=300,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig

def create_3d_acceleration_chart(df):
    """Create 3D acceleration scatter plot"""
    if len(df) < 2:
        return go.Figure()
    
    fig = go.Figure(data=[go.Scatter3d(
        x=df['accel_x'],
        y=df['accel_y'],
        z=df['accel_z'],
        mode='markers',
        marker=dict(
            size=8,
            color=df.index,
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title="3D Acceleration Pattern",
        scene=dict(
            xaxis_title='Accel X',
            yaxis_title='Accel Y',
            zaxis_title='Accel Z'
        ),
        height=400,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig

def process_messages():
    """Load and process messages from temporary file"""
    try:
        messages_from_file = load_messages_from_temp()
        
        if not messages_from_file:
            return 0
        
        new_messages = []
        for msg in messages_from_file:
            if len(st.session_state.messages) == 0 or msg['timestamp'] > st.session_state.last_processed_time:
                new_messages.append(msg)
        
        if new_messages:
            # Process new messages
            for msg in new_messages:
                data = msg['data']
                st.session_state.messages.append(data)
                st.session_state.latest_data = data
                st.session_state.total_received += 1
                
                # Check for alerts
                new_alerts = check_alerts(data)
                for alert in new_alerts:
                    if alert not in st.session_state.alerts:
                        st.session_state.alerts.append(alert)
            
            # Update DataFrame for plotting
            update_dataframe()
            
            # Update last processed time
            st.session_state.last_processed_time = new_messages[-1]['timestamp']
            
            return len(new_messages)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error processing messages: {e}")
        return 0

def update_dataframe():
    """Update DataFrame with latest messages for plotting"""
    if not st.session_state.messages:
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.messages)
    
    # Keep only last 200 points for performance
    if len(df) > 200:
        df = df.tail(200)
    
    # Reset index to use as time axis
    df = df.reset_index(drop=True)
    
    st.session_state.data_frame = df

# --- Streamlit UI ---
st.set_page_config(
    page_title="Industrial IoT Dashboard", 
    layout="wide",
    page_icon="🏭"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .alert-card {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff0000;
        margin-bottom: 0.5rem;
    }
    .stream-graph-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏭 M_PRESS Machine Performance & Structural Vibration Dashboard</h1>', unsafe_allow_html=True)

# Process messages at startup
processed_count = process_messages()

# Control Panel
with st.container():
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])
    
    with col1:
        if st.button("🚀 Start Monitoring", type="primary", use_container_width=True):
            start_consumer()
            st.success("Machine Monitoring Activated")
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True):
            processed = process_messages()
            if processed > 0:
                st.success(f"Refreshed {processed} messages!")
    
    with col3:
        if st.button("📊 Update Plots", use_container_width=True):
            update_dataframe()
            st.success("Plots Updated!")
    
    with col4:
        if st.button("🗑️ Clear Data", use_container_width=True):
            st.session_state.messages = []
            st.session_state.latest_data = {}
            st.session_state.total_received = 0
            st.session_state.data_frame = pd.DataFrame()
            st.session_state.alerts = []
            st.rerun()
    
    with col5:
        if st.button("🔔 Clear Alerts", use_container_width=True):
            st.session_state.alerts = []

# Status and Alerts Row
col1, col2 = st.columns([2, 1])

with col1:
    # Connection Status
    status = st.session_state.connection_status
    if "Connected" in status and "Error" not in status:
        st.success(f"🔗 **Status:** {status}")
    elif "Error" in status:
        st.error(f"🔗 **Status:** {status}")
    else:
        st.warning(f"🔗 **Status:** {status}")
    
    st.metric("📨 Total Telemetry Packets", st.session_state.total_received)

with col2:
    # Alerts
    if st.session_state.alerts:
        st.error(f"🚨 Active Alerts: {len(st.session_state.alerts)}")
    else:
        st.success("✅ No Active Fault/Alarms")

# Display Alerts
if st.session_state.alerts:
    with st.expander("🚨 Active Alerts", expanded=True):
        for alert in st.session_state.alerts[-10:]:  # Show last 10 alerts
            st.markdown(f'<div class="alert-card">{alert}</div>', unsafe_allow_html=True)

# Main Dashboard
if st.session_state.messages:
    latest = st.session_state.latest_data
    
    # Real-time Metrics Row
    st.subheader("📊 Live Press Parameters")
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        if 'current' in latest:
            st.metric("⚡ Current", f"{latest['current']:.2f} A", 
                     delta=f"{(latest['current'] - 2.5):.2f}A" if 'current' in st.session_state.messages[-2] else None)
    
    with metric_cols[1]:
        if 'voltage' in latest:
            st.metric("🔌 Line Supply Voltage", f"{latest['voltage']:.1f} V",
                     delta=f"{(latest['voltage'] - 230):.1f}V" if 'voltage' in st.session_state.messages[-2] else None)
    
    with metric_cols[2]:
        if all(k in latest for k in ['accel_x', 'accel_y', 'accel_z']):
            magnitude = (latest['accel_x']**2 + latest['accel_y']**2 + latest['accel_z']**2)**0.5
            st.metric("📈 Structural Vibration (RMS)", f"{magnitude:.2f} m/s²")
    
    with metric_cols[3]:
        st.metric("⏱️ Telemetry Rate", f"{processed_count}/min")
    
   
    
    # Traditional Time Series Charts
    st.subheader("📈 Press Component Condition AnalysisAnalysis")
    if not st.session_state.data_frame.empty:
        ts_cols = st.columns(3)
        
        with ts_cols[0]:
            if 'current' in st.session_state.data_frame.columns:
                fig_current_ts = create_time_series_chart(
                    st.session_state.data_frame, 'current', 'Current Trend', 'blue'
                )
                st.plotly_chart(fig_current_ts, use_container_width=True)
        
        with ts_cols[1]:
            if 'voltage' in st.session_state.data_frame.columns:
                fig_voltage_ts = create_time_series_chart(
                    st.session_state.data_frame, 'voltage', 'Voltage Trend', 'green'
                )
                st.plotly_chart(fig_voltage_ts, use_container_width=True)
        
        with ts_cols[2]:
            if all(k in st.session_state.data_frame.columns for k in ['accel_x', 'accel_y', 'accel_z']):
                magnitude = np.sqrt(
                    st.session_state.data_frame['accel_x']**2 + 
                    st.session_state.data_frame['accel_y']**2 + 
                    st.session_state.data_frame['accel_z']**2
                )
                temp_df = st.session_state.data_frame.copy()
                temp_df['magnitude'] = magnitude
                fig_magnitude_ts = create_time_series_chart(
                    temp_df, 'magnitude', 'Vibration Magnitude', 'red'
                )
                st.plotly_chart(fig_magnitude_ts, use_container_width=True)
    
    # 3D Acceleration Plot
    if all(k in st.session_state.data_frame.columns for k in ['accel_x', 'accel_y', 'accel_z']):
        st.subheader("🎯 3-Axis Press Vibration Signature")
        fig_3d = create_3d_acceleration_chart(st.session_state.data_frame)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # Raw Data Section
    with st.expander("📋 Raw Telemetry Snapshot"):
        if st.session_state.messages:
            st.dataframe(pd.DataFrame(st.session_state.messages[-20:]), use_container_width=True)

else:
    # Welcome/Setup Screen
    st.info("🎯 **Welcome to Industrial Vibration Monitoring Dashboard**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        ### To get started:
        1. Click **🚀 Start Monitoring** to begin receiving data
        2. Click **🔄 Refresh Data** to load received messages
        3. Monitor real-time metrics and alerts
        4. Analyze vibration patterns with stream graphs
        
        ### Vibration Analysis Features:
        - 🌊 **Stream Graphs** for signal visualization
        - 📡 **Frequency Spectrum** analysis
        - 📈 **Component-wise** time series
        - 🎯 **3D Vibration** patterns
        - 🚨 **Smart alert** system
        """)
    
    with col2:
        st.write("### Expected Data Format:")
        st.json({
            "current": 2.85,
            "voltage": 220.0, 
            "accel_x": 0.09,
            "accel_y": 0.09,
            "accel_z": 9.64
        })

# Auto-refresh
if st.session_state.consumer_started:
    time.sleep(2)
    st.rerun()