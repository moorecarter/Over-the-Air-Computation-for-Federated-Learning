import streamlit as st
import pandas as pd
import numpy as np
import zmq
import time
import threading
import json
from streamlit.runtime.scriptrunner import add_script_run_ctx
import altair as alt
import base64
#NEED TO ADD THE RX_BUFFER IMPL ON THE BACKEND SO THAT WE CAN DISPLAY THE RX_BUFFER IN THE CHART


# SERVER_IP = "10.216.218.166"
SERVER_IP = "localhost"
SERVER_PORT = 5555

st.set_page_config(
    page_title="FLOTA",
    layout="wide",
    page_icon=":satellite:"  
)

custom_css = """
<style>
    /* Import both fonts directly from Google */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');


    /* 2. Apply Inter to all general text and headers */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
    }

    /* 3. Apply JetBrains Mono to all technical stuff (Code, Metrics, DataFrames) */
    code, pre, .stDataFrame, [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* pill hover styling */
    div[data-testid="stPills"] button:hover {
        border: 1px solid #00FFFF !important;
    }
    /* Targets the main Streamlit app background */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% -20%, #2B3A67 0%, #1E1B3A 50%, #1E1E1E 95%) !important;    
    }
    
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    
</style>
"""

# Data schema 

# TOPIC: config
# --------------
# Sent once at startup
# {
#   "total_params": 390472,
#   "batch_size": 32,
#   "local_epochs": 3,
#   "num_clients": 3,
#   "model": "small_cnn",
#   "dataset": "bloodmnist",
#   "freq_hz": 2.41e9,
#   "sample_rate": 1e6
# }

# TOPIC: status
# --------------
# Sent during round
# {
#   "state": "idle" | "training" | "transmitting" | "receiving",
#   "round": 3,
# }

# TOPIC: metrics
# --------------
# Sent after each round evaluation
# {
#   "accuracy": 0.826,
#   "loss": 0.54,
#   "csi_per_client": [
#     {"magnitude": 0.0123, "phase_deg": -45.2},
#     {"magnitude": 0.0118, "phase_deg": 12.7},
#     {"magnitude": 0.0121, "phase_deg": -89.1}
#   ],
#   "rx_buffer": [complex64],
#   "time_offsets_ns": [0.0, 1000.0, -500.0],
#   "snr_db": [25.3, 24.1, 26.0]
# }

# Initialize session state variables for before message arrives

if 'config' not in st.session_state:
    print("Config not in session state, initializing...")
    st.session_state['config'] = {
        'total_params': 0,
        'batch_size': 0,
        'local_epochs': 0,
        'num_rounds': 0,
        'num_clients': 0,
        'model_name': '[Model Name]',
        'dataset_name': '[Dataset Name]',
        'freq_hz': 0,
        'sample_rate': 0,
        'start_time': 0,
    }
if 'status' not in st.session_state:
    st.session_state['status'] = {
        'state': 'none',
        'round': 0,
    }

if 'metrics' not in st.session_state:
    st.session_state['metrics'] = {
        'accuracy': 0,
        'loss': 0,
        'rx_buffer': np.array([], dtype=np.complex64),
        'csi_per_client': [],
        'time_offsets_ns': [],
        'snr_db': [],
    }
if 'metrics_accum' not in st.session_state:
    st.session_state['metrics_accum'] = {
        'accuracy': pd.DataFrame(columns=['round', 'accuracy']),
        'loss': pd.DataFrame(columns=['round', 'loss']),
        'rx_buffer': pd.DataFrame(columns=['round', 'magnitude']),
        'csi_per_client': pd.DataFrame(columns=['round', 'client', 'magnitude', 'phase_deg']),
        'time_offsets_ns': pd.DataFrame(columns=['round', 'client', 'time_offset_ns']),
        'snr_db': pd.DataFrame(columns=['round', 'client', 'snr_db']),
    }

if 'restart' not in st.session_state:
    st.session_state['restart'] = False

def zmq_listener():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{SERVER_IP}:{SERVER_PORT}")
    print(f"Connected to {SERVER_IP}:{SERVER_PORT}")
    # Subscribe to all topics and filter in code so we do not
    # silently miss messages due to an exact-prefix mismatch.
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    # Block until the initial config arrives.
    socket.setsockopt(zmq.RCVTIMEO, -1)  # block indefinitely until first message

    while True:
        try:
            # Either returns a message within the timeout or raises zmq.Again
            message = socket.recv_multipart()
            topic = message[0].decode('utf-8')
            payload = message[1]
            data = json.loads(payload.decode('utf-8'))
            if topic == "config":
                print("Config: ", data)
                st.session_state['config']['total_params'] = data['total_params']
                st.session_state['config']['batch_size'] = data['batch_size']
                st.session_state['config']['local_epochs'] = data['local_epochs']
                st.session_state['config']['num_rounds'] = data['num_rounds']
                st.session_state['config']['num_clients'] = data['num_clients']
                st.session_state['config']['model_name'] = data['model']
                st.session_state['config']['dataset_name'] = data['dataset']
                st.session_state['config']['freq_hz'] = data['freq_hz']
                st.session_state['config']['sample_rate'] = data['sample_rate']
                st.session_state['config']['start_time'] = time.time()
                st.session_state['status']['state'] = 'idle'
                # Once we've seen the initial config, enable a timeout
                socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 s
            elif topic == "status" and st.session_state['status']['state'] != 'none':
                print("Status: ", data)
                st.session_state['status']['state'] = data['state']
                st.session_state['status']['round'] = data['round']
            elif topic == "metrics" and st.session_state['status']['state'] != 'none':
                print("Metrics: ", data)

                # Update latest metrics snapshot (tolerant to optional fields)
                st.session_state['metrics']['accuracy'] = data.get('accuracy', 0.0)
                st.session_state['metrics']['loss'] = data.get('loss', 0.0)
                st.session_state['metrics']['csi_per_client'] = data.get('csi_per_client', []) or []
                st.session_state['metrics']['time_offsets_ns'] = data.get('time_offsets_ns', []) or []
                st.session_state['metrics']['snr_db'] = data.get('snr_db', []) or []
                rx_mag = data.get('rx_buffer_mag', [])
                st.session_state['metrics']['rx_buffer_mag'] = np.array(rx_mag, dtype=float)
                current_round = st.session_state['status'].get('round', 0)
                if isinstance(data, dict) and 'round' in data:
                    current_round = data['round']

                # Accumulate metrics (per round)
                acc_df = st.session_state['metrics_accum']['accuracy']
                new_acc_row = pd.DataFrame(
                    [{'round': current_round, 'accuracy': data['accuracy']}]
                )
                st.session_state['metrics_accum']['accuracy'] = (
                    new_acc_row
                    if acc_df.empty
                    else pd.concat([acc_df, new_acc_row], ignore_index=True)
                )

                loss_df = st.session_state['metrics_accum']['loss']
                new_loss_row = pd.DataFrame(
                    [{'round': current_round, 'loss': data['loss']}]
                )
                st.session_state['metrics_accum']['loss'] = (
                    new_loss_row
                    if loss_df.empty
                    else pd.concat([loss_df, new_loss_row], ignore_index=True)
                )

                # Accumulate per‑client metrics
                num_clients = st.session_state['config'].get('num_clients', 0)

                # CSI per client
                csi_list = data.get('csi_per_client', []) or []
                csi_rows = []
                for idx, csi in enumerate(csi_list):
                    csi_rows.append(
                        {
                            'round': current_round,
                            'client': idx if not num_clients else idx % num_clients,
                            'magnitude': csi.get('magnitude', np.nan),
                            'phase_deg': csi.get('phase_deg', np.nan),
                        }
                    )
                if csi_rows:
                    csi_df = st.session_state['metrics_accum']['csi_per_client']
                    new_csi_df = pd.DataFrame(csi_rows)
                    st.session_state['metrics_accum']['csi_per_client'] = (
                        new_csi_df
                        if csi_df.empty
                        else pd.concat([csi_df, new_csi_df], ignore_index=True)
                    )

                # Time offsets per client
                offsets_list = data.get('time_offsets_ns', []) or []
                offset_rows = []
                for idx, offset in enumerate(offsets_list):
                    offset_rows.append(
                        {
                            'round': current_round,
                            'client': idx if not num_clients else idx % num_clients,
                            'time_offset_ns': offset,
                        }
                    )
                if offset_rows:
                    offsets_df = st.session_state['metrics_accum']['time_offsets_ns']
                    new_offsets_df = pd.DataFrame(offset_rows)
                    st.session_state['metrics_accum']['time_offsets_ns'] = (
                        new_offsets_df
                        if offsets_df.empty
                        else pd.concat(
                            [offsets_df, new_offsets_df], ignore_index=True
                        )
                    )

                # SNR per client
                snr_list = data.get('snr_db', []) or []
                snr_rows = []
                for idx, snr in enumerate(snr_list):
                    snr_rows.append(
                        {
                            'round': current_round,
                            'client': idx if not num_clients else idx % num_clients,
                            'snr_db': snr,
                        }
                    )
                if snr_rows:
                    snr_df = st.session_state['metrics_accum']['snr_db']
                    new_snr_df = pd.DataFrame(snr_rows)
                    st.session_state['metrics_accum']['snr_db'] = (
                        new_snr_df
                        if snr_df.empty
                        else pd.concat([snr_df, new_snr_df], ignore_index=True)
                    )
        except zmq.Again:
            continue

def get_video_data(video_path):
    with open(video_path, 'rb') as v_file:
        video_bytes = v_file.read()
        encoded_video = base64.b64encode(video_bytes).decode('utf-8')
    return encoded_video

if 'video_cache' not in st.session_state:
    st.session_state['video_cache'] = {'idle': get_video_data('images/IDLE.mp4'),
    'training': get_video_data('images/TRAIN.mp4'),
    'transmitting': get_video_data('images/OTA.mp4'),
    'writeback': get_video_data('images/WRITEBACK.mp4'),
    }

# Start thread for ZMQ listener
if 'thread_started' not in st.session_state:
    listener_thread = threading.Thread(target=zmq_listener, daemon=True)
    add_script_run_ctx(listener_thread) 
    listener_thread.start()
    st.session_state['thread_started'] = True

st.markdown(custom_css, unsafe_allow_html=True)

@st.fragment(run_every="1s")
def live_timer():
    st.metric(label="Training Time", value=f"{int(time.time() - st.session_state['config']['start_time'])} s")

st.title("📡 Over-the-Air Computation for Federated Learning ")

with st.container(height="content", border=True, horizontal_alignment="center", vertical_alignment="center"):
    left_spacer, center_col, right_spacer = st.columns([1, 3, 1])
    with center_col:
        current_state = st.session_state['status']['state']
        state_to_file = {
            'idle': 'images/IDLE.webp',
            'training': 'images/TRAIN.webp',
            'transmitting': 'images/OTA.webp',
            'writeback': 'images/WRITEBACK.webp'
        }
        
        st.image(state_to_file.get(current_state, 'images/IDLE.webp'), width=1000)
        # 4. Inject the encoded string into your HTML using a data URI
        # st.markdown(f"""
        #     <div>
        #         <video width="1000" autoplay loop muted playsinline>
        #             <source src="data:video/mp4;base64,{st.session_state['video_cache'].get(current_state, st.session_state['video_cache']['idle'])}" type="video/mp4">
        #         </video>
        #     </div>
        # """, unsafe_allow_html=True)
        if int(st.session_state['config']['num_rounds']) > 0 and int(st.session_state['status']['round']) == int(st.session_state['config']['num_rounds']):
            st.success("Training completed!", icon="🎉")
            st.session_state['status']['state'] = 'idle'
        if int(st.session_state['config']['num_rounds']) == 0:
            st.info("Training not started!", icon="ℹ️")
        if int(st.session_state['config']['num_rounds']) > 0 and int(st.session_state['status']['round']) == 0 and st.session_state['status']['state'] == 'idle':
            st.warning("Initializing training...", icon="🔧")
        if st.session_state['status']['state'] != 'idle' and st.session_state['status']['round'] > 0:
            st.info(f"Training {st.session_state['status']['round']}/{st.session_state['config']['num_rounds']} rounds", icon="🧠")
        if st.session_state['status']['state'] == 'timeout':
            st.error(f"Training timed out at round {st.session_state['status']['round']}! Please check the server and try again.", icon="🚨")


if st.session_state['status']['state'] != 'idle' and st.session_state['status']['round'] > 0:
    progress_raw = st.session_state['status']['round'] / st.session_state['config']['num_rounds']
    # Clamp to [0, 1] to satisfy Streamlit's progress API
    progress_value = max(0.0, min(1.0, float(progress_raw)))
    st.progress(
        value=progress_value,
        text=f"Training Progress: {(st.session_state['status']['round'])/(st.session_state['config']['num_rounds']) * 100:.2f}%"
    )


left_col, right_col = st.columns([1, 2])

# 2. Build the Left Column (Fixed Controls & Summary)
with left_col:    
    st.subheader("🔍 Select Metric")

    metric_selector = st.selectbox(
        label = "Select Metric to Display",
        label_visibility="collapsed",
        options=("Global Model Accuracy", "USRP Data Transfer", "Channel State Information (CSI)"),
        index=0
    )
    
    st.divider()
    st.subheader("🧾 Summary")
    metric_selector2 = st.pills(label="Select Metric", options=("Training Data", "Model Hyperparameters", "OTA Communication"), default="Training Data", selection_mode="single")
    if metric_selector2 == "Training Data":            
        if st.session_state['status']['state'] != 'idle' and int(st.session_state['config']['num_rounds']) > 0 and int(st.session_state['status']['round']) < int(st.session_state['config']['num_rounds']):
            live_timer()
        else:
            st.metric(label = "Training Time", value="Not Training")
        st.metric(label="Current Round", value=f"{st.session_state['status']['round']}")

        # Global accuracy value (as percentage)
        current_acc = float(st.session_state['metrics']['accuracy'])
        acc_history = st.session_state['metrics_accum']['accuracy']['accuracy']
        if len(acc_history) >= 2:
            delta_acc = float(acc_history.iloc[-1] - acc_history.iloc[-2]) * 100.0
            delta_text_acc = f"{delta_acc:.2f}%"
        else:
            delta_text_acc = "0.00%"

        st.metric(
            label="Global Model Accuracy",
            value=f"{current_acc * 100.0:.2f}%",
            delta=delta_text_acc,
        )
        current_loss = float(st.session_state['metrics']['loss'])
        loss_history = st.session_state['metrics_accum']['loss']['loss']
        if len(loss_history) >= 2:
            delta_loss = float(loss_history.iloc[-1] - loss_history.iloc[-2])
            delta_text_loss = f"{delta_loss:.2f}"
        else:
            delta_text_loss = "0.00"
        st.metric(
            label="Global Model Loss",
            value=f"{current_loss:.2f}",
            delta=delta_text_loss,
        )

    elif metric_selector2 == "Model Hyperparameters":
        if not 'model_name' in st.session_state['config'] or not 'dataset_name' in st.session_state['config']:
            st.session_state['config']['model_name'] = "[No Model Selected]"
            st.session_state['config']['dataset_name'] = "[No Dataset Selected]"
        st.markdown(f"##### Training Model: {st.session_state['config']['model_name']} on Dataset: {st.session_state['config']['dataset_name']}") 
        st.metric(label="Total Parameters", value=f"{st.session_state['config']['total_params']}") 
        st.metric(label="Payload Size (CMPLX64)", value=f"%.2f MB" % (st.session_state['config']['total_params'] * 8 / 1e6)) 
        st.metric(label="Local Batch Size", value=f"{st.session_state['config']['batch_size']}")

    elif metric_selector2 == "OTA Communication":
        st.markdown("##### TX/RX: USRP X310 (UBX-160)")
        st.metric(label="Center Frequency", value=f"%.2f GHz" % (st.session_state['config']['freq_hz'] / 1e9)) 

        # time_offsets_ns and snr_db are lists in the latest metrics snapshot
        offsets = st.session_state['metrics']['time_offsets_ns'] or [0.0]
        snrs = st.session_state['metrics']['snr_db'] or [0.0]

        time_sync_offset = max(offsets) - min(offsets)
        mean_snr = float(np.mean(snrs))

        st.metric(label="Max. Client Time Sync Offset", value=f"{time_sync_offset:.2f} μs")
        st.metric(label="Avg. SNR", value=f"{mean_snr:.2f} dB")

# 3. Build the Right Column (Scrollable Charts)
with right_col:
    st.subheader(f"📊 Live View: {metric_selector}")
    
    # The magic happens here: a container with a fixed height becomes scrollable
    with st.container(height="content"):
        if st.session_state['status']['state'] == 'idle' and int(st.session_state['config']['num_rounds']) != int(st.session_state['status']['round']) and int(st.session_state['status']['round']) != 0:
            st.info("Waiting for data...")
        else:
            if metric_selector == "Global Model Accuracy":
                acc_df = st.session_state['metrics_accum']['accuracy']
                loss_df = st.session_state['metrics_accum']['loss']
                if not acc_df.empty and not loss_df.empty:
                    chart_df = pd.concat([acc_df.set_index('round')['accuracy'], loss_df.set_index('round')['loss']], axis=1)
                    st.line_chart(chart_df, x_label="Round", color=["#00edfa", "#ff00ff"], y=["accuracy", "loss"])
                else:
                    st.info("Waiting for first round to complete...")
                st.caption("Global accuracy and loss aggr   egating over successive communication rounds. Accuracy is in percentage, loss is in float.")
                
            elif metric_selector == "USRP Data Transfer":
                # RX buffer magnitude only
                rx_mag = st.session_state['metrics'].get('rx_buffer_mag', np.array([]))

                if isinstance(rx_mag, np.ndarray) and rx_mag.size > 0:
                    df = pd.DataFrame(
                        {
                            "sample_idx": np.arange(len(rx_mag)),
                            "magnitude": rx_mag,
                        }
                    )
                    st.line_chart(
                        df,
                        x="sample_idx",
                        y="magnitude",
                        x_label="Sample Index",
                        y_label="Magnitude",
                        color=["#00edfa"],
                    )
                else:
                    st.info("Waiting for data transfer...")
                st.caption("Aggregated RX waveform magnitude from the USRP for the latest round.")

            elif metric_selector == "Channel State Information (CSI)":
                csi_df = st.session_state['metrics_accum']['csi_per_client']
                if not csi_df.empty:
                    all_csi = csi_df.copy()

                    # Convert magnitude/phase (deg) to complex plane coordinates
                    radians = np.deg2rad(all_csi['phase_deg'])
                    all_csi['Real'] = all_csi['magnitude'] * np.cos(radians)
                    all_csi['Imaginary'] = all_csi['magnitude'] * np.sin(radians)

                    chart = (
                        alt.Chart(all_csi)
                        .mark_circle(size=150)
                        .encode(
                            x='Real',
                            y='Imaginary',
                            color=alt.Color('client:N', legend=None),
                            tooltip=['client', 'round', 'Real', 'Imaginary'],
                        )
                        .interactive()
                    )

                    st.altair_chart(chart, width='stretch')
                else:
                    st.info("Waiting for CSI estimation...")
                st.caption("Live CSI on the complex plane across all rounds seen so far. Each point represents a client's phase and magnitude.")

# Refresh cadence:

if st.session_state['status']['state'] != 'idle':
    time.sleep(1.0)
    st.rerun()
else:
    time.sleep(2.0)
    st.rerun()

    