"""
=============================================================================
Project Title: Automatic Speech Recognition (ASR) Tool Using Python and OpenAI Whisper
Description  : A modern, functional web application built with Streamlit,
               OpenAI Whisper, Librosa, and SoundFile for college/academic submission.
Author       : Computer Science & Engineering Student Project
Tech Stack   : Python 3, OpenAI Whisper, Streamlit, PyTorch, Librosa, SoundFile
=============================================================================
"""

import os
import io
import time
import tempfile
import numpy as np
import streamlit as st
import soundfile as sf
import librosa
import torch
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ASR Tool - OpenAI Whisper",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for a polished, modern, academic presentation
st.markdown("""
<style>
    /* Main container adjustments */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    
    /* Card-like containers */
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Result Box */
    .result-container {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-left: 5px solid #2563EB;
        border-radius: 6px;
        padding: 16px;
        margin-top: 10px;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #1E293B;
    }

    /* Badge */
    .lang-badge {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1D4ED8;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
        margin-right: 8px;
    }
    .time-badge {
        display: inline-block;
        background-color: #DCFCE7;
        color: #15803D;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 9999px;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 24px 0 10px 0;
        color: #94A3B8;
        font-size: 0.85rem;
        border-top: 1px solid #E2E8F0;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "transcription_data" not in st.session_state:
    st.session_state["transcription_data"] = None
if "audio_info" not in st.session_state:
    st.session_state["audio_info"] = None
if "audio_bytes" not in st.session_state:
    st.session_state["audio_bytes"] = None
if "audio_array" not in st.session_state:
    st.session_state["audio_array"] = None


def reset_application():
    """Resets all uploaded audio and transcription session data."""
    st.session_state["uploader_key"] += 1
    st.session_state["transcription_data"] = None
    st.session_state["audio_info"] = None
    st.session_state["audio_bytes"] = None
    st.session_state["audio_array"] = None


# -----------------------------------------------------------------------------
# 3. Model Loading with Caching (CPU-Optimized)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_whisper_model(model_name: str = "base"):
    """
    Loads and caches the OpenAI Whisper model in memory.
    Cached so model weights are loaded only once per session.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)
    return model, device


# -----------------------------------------------------------------------------
# 4. Audio Preprocessing using Librosa and SoundFile
# -----------------------------------------------------------------------------
def preprocess_audio_file(uploaded_file):
    """
    Validates, decodes, and resamples uploaded audio to a 16 kHz mono float32 numpy array.
    Uses SoundFile for primary fast decoding and Librosa for resampling/fallbacks.
    Returns: (audio_float32, sample_rate, duration_seconds, raw_bytes)
    """
    raw_bytes = uploaded_file.read()
    if len(raw_bytes) == 0:
        raise ValueError("The uploaded file is empty. Please provide a valid audio file.")

    # Save to a temporary file for robust multi-backend audio parsing
    file_suffix = "." + uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        tmp_file.write(raw_bytes)
        tmp_path = tmp_file.name

    try:
        # Step 1: Decode audio signal using soundfile or librosa
        decoded = False
        audio_data = None
        sr = None

        # Attempt SoundFile first (WAV, MP3, FLAC, OGG)
        try:
            audio_data, sr = sf.read(tmp_path, dtype="float32")
            decoded = True
        except Exception:
            pass

        # Fallback to Librosa if SoundFile fails (handles various codecs/containers)
        if not decoded or audio_data is None:
            try:
                audio_data, sr = librosa.load(tmp_path, sr=None, mono=False)
                decoded = True
            except Exception as lib_err:
                raise ValueError(
                    f"Unsupported or corrupted audio stream. Unable to decode '{uploaded_file.name}'. "
                    f"Error details: {str(lib_err)}"
                )

        # Step 2: Convert Multi-channel / Stereo to Mono
        if audio_data.ndim > 1:
            if audio_data.shape[0] < audio_data.shape[1]:
                audio_data = np.mean(audio_data, axis=0)
            else:
                audio_data = np.mean(audio_data, axis=1)

        # Step 3: Resample to 16,000 Hz (Whisper standard acoustic input frequency)
        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Step 4: Ensure float32 array in normalized range [-1.0, 1.0]
        audio_data = audio_data.astype(np.float32)
        max_abs = np.max(np.abs(audio_data))
        if max_abs > 1.0:
            audio_data = audio_data / max_abs

        duration_seconds = float(len(audio_data) / sr)

        if duration_seconds < 0.1:
            raise ValueError("The audio file is too short (less than 0.1 seconds) to process.")

        return audio_data, sr, duration_seconds, raw_bytes

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def plot_waveform(audio_array: np.ndarray, sample_rate: int = 16000):
    """Generates a clean waveform figure using Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 2.2), dpi=100)
    time_axis = np.linspace(0, len(audio_array) / sample_rate, num=len(audio_array))
    
    # Downsample points for efficient plotting if file is long
    max_points = 10000
    if len(audio_array) > max_points:
        step = len(audio_array) // max_points
        time_axis = time_axis[::step]
        audio_array_sub = audio_array[::step]
    else:
        audio_array_sub = audio_array

    ax.plot(time_axis, audio_array_sub, color="#2563EB", linewidth=0.8, alpha=0.85)
    ax.fill_between(time_axis, audio_array_sub, color="#93C5FD", alpha=0.35)
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_xlim(0, time_axis[-1] if len(time_axis) > 0 else 1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (seconds)", fontsize=9, color="#475569")
    ax.set_ylabel("Amplitude", fontsize=9, color="#475569")
    ax.tick_params(colors="#64748B", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#E2E8F0")
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# 5. Sidebar Navigation & Control Panel
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=64)
    st.title("Settings & Controls")
    
    # Hardware status check
    cuda_available = torch.cuda.is_available()
    device_label = "GPU (CUDA)" if cuda_available else "CPU (Laptop Optimized)"
    st.info(f"**Hardware Device**: {device_label}")

    # Model Selection (Base model is default as specified in instructions)
    model_choice = st.selectbox(
        "Whisper Model Architecture",
        options=["base", "tiny", "small"],
        index=0,
        help="The 'base' model offers the best balance of speed, low memory usage, and transcription accuracy on CPUs."
    )

    st.markdown("---")
    st.subheader("Supported Audio Formats")
    st.markdown("""
    - **WAV** (`audio/wav`)
    - **MP3** (`audio/mp3`)
    - **M4A** (`audio/m4a`, `audio/mp4`)
    """)

    st.markdown("---")
    st.subheader("Reset / Clear")
    if st.button("🗑️ Clear Audio & Results", use_container_width=True, help="Reset the uploader and clear previous transcriptions"):
        reset_application()
        st.rerun()

    st.markdown("---")
    st.caption("College Mini Project • Academic Submission")
    st.caption("Department of Computer Science & Engineering")


# -----------------------------------------------------------------------------
# 6. Header Section
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Automatic Speech Recognition (ASR) Tool</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Convert spoken speech into accurate text using <b>OpenAI Whisper</b>, '
    '<b>Librosa</b>, and <b>SoundFile</b>. Optimized for local CPU execution without requiring paid API keys.'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 7. Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_transcribe, tab_architecture, tab_help = st.tabs([
    "🎙️ Speech-to-Text Studio",
    "🧠 How ASR Works",
    "📖 User Guide & Testing"
])


# =============================================================================
# TAB 1: SPEECH-TO-TEXT STUDIO
# =============================================================================
with tab_transcribe:
    col_upload, col_preview = st.columns([1.1, 0.9], gap="large")

    with col_upload:
        st.subheader("1. Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose a spoken audio file (.wav, .mp3, .m4a)",
            type=["wav", "mp3", "m4a"],
            key=f"audio_uploader_{st.session_state['uploader_key']}",
            help="Select an audio recording containing speech."
        )

        if uploaded_file is not None:
            try:
                if (st.session_state["audio_info"] is None or 
                    st.session_state["audio_info"].get("filename") != uploaded_file.name):
                    
                    with st.spinner("Decoding audio with SoundFile and Librosa..."):
                        audio_array, sr, duration, raw_bytes = preprocess_audio_file(uploaded_file)
                        
                        st.session_state["audio_array"] = audio_array
                        st.session_state["audio_bytes"] = raw_bytes
                        st.session_state["audio_info"] = {
                            "filename": uploaded_file.name,
                            "filesize_kb": round(len(raw_bytes) / 1024, 2),
                            "duration_sec": round(duration, 2),
                            "sample_rate": sr,
                            "samples_count": len(audio_array)
                        }
                        st.session_state["transcription_data"] = None

            except ValueError as val_err:
                st.error(f"❌ Error: {str(val_err)}")
                st.session_state["audio_array"] = None
                st.session_state["audio_bytes"] = None
                st.session_state["audio_info"] = None
            except Exception as ex:
                st.error(f"❌ An unexpected error occurred while loading audio: {str(ex)}")
                st.session_state["audio_array"] = None
                st.session_state["audio_bytes"] = None
                st.session_state["audio_info"] = None

    with col_preview:
        st.subheader("2. Audio Playback & Inspection")
        if st.session_state["audio_info"] is not None and st.session_state["audio_bytes"] is not None:
            info = st.session_state["audio_info"]
            st.audio(st.session_state["audio_bytes"], format=f"audio/{info['filename'].split('.')[-1]}")

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{info["duration_sec"]}s</div>'
                    f'<div class="metric-lbl">Duration</div></div>',
                    unsafe_allow_html=True
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{info["filesize_kb"]} KB</div>'
                    f'<div class="metric-lbl">File Size</div></div>',
                    unsafe_allow_html=True
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{info["sample_rate"]} Hz</div>'
                    f'<div class="metric-lbl">Sample Rate</div></div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("Upload an audio file on the left to preview playback and inspection metrics.")

    # Waveform Display Section
    if st.session_state["audio_array"] is not None:
        with st.expander("📊 View Audio Signal Waveform", expanded=False):
            fig = plot_waveform(st.session_state["audio_array"], sample_rate=16000)
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("---")

    # Transcribe Action Button
    st.subheader("3. Transcribe Speech")
    
    can_transcribe = st.session_state["audio_array"] is not None
    transcribe_clicked = st.button(
        "🚀 Start Transcription",
        type="primary",
        disabled=not can_transcribe,
        use_container_width=False,
        help="Transcribe the loaded audio file into text using OpenAI Whisper"
    )

    if transcribe_clicked and can_transcribe:
        try:
            with st.spinner(f"Transcribing audio using Whisper '{model_choice}' model on {device_label}..."):
                start_time = time.time()
                
                # Load cached model
                model, device = load_whisper_model(model_choice)
                
                # Run Whisper transcription directly on float32 array
                result = model.transcribe(
                    st.session_state["audio_array"],
                    fp16=(device == "cuda"),
                    verbose=False
                )
                
                elapsed_time = round(time.time() - start_time, 2)
                
                st.session_state["transcription_data"] = {
                    "text": result.get("text", "").strip(),
                    "language": result.get("language", "unknown"),
                    "segments": result.get("segments", []),
                    "elapsed_time": elapsed_time,
                    "model_used": model_choice
                }
                st.success("Transcription complete!")

        except Exception as e:
            st.error(f"❌ Transcription failed: {str(e)}")
            st.info("Tip: Ensure the audio contains valid speech and your system has sufficient free memory.")

    # -------------------------------------------------------------------------
    # Transcription Output Display & Download
    # -------------------------------------------------------------------------
    if st.session_state["transcription_data"] is not None:
        t_data = st.session_state["transcription_data"]
        detected_lang = t_data["language"]
        
        lang_display = detected_lang.upper()
        if hasattr(whisper, "tokenizer") and hasattr(whisper.tokenizer, "LANGUAGES"):
            lang_display = whisper.tokenizer.LANGUAGES.get(detected_lang, detected_lang).title()

        st.markdown("### 📝 Transcription Results")
        
        st.markdown(
            f'<span class="lang-badge">🌐 Detected Language: {lang_display} ({detected_lang})</span>'
            f'<span class="time-badge">⚡ Processed in: {t_data["elapsed_time"]}s</span>'
            f'<span class="lang-badge" style="background-color:#F1F5F9; color:#475569;">🧠 Model: {t_data["model_used"]}</span>',
            unsafe_allow_html=True
        )

        transcribed_text = t_data["text"]
        if transcribed_text:
            st.markdown(
                f'<div class="result-container">{transcribed_text}</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ No discernible speech was detected in the audio file.")

        st.markdown("<br>", unsafe_allow_html=True)

        col_down1, col_down2 = st.columns([1, 1])

        download_content = (
            f"===========================================================\n"
            f"Automatic Speech Recognition (ASR) - Transcription Report\n"
            f"===========================================================\n"
            f"File Name        : {st.session_state['audio_info']['filename']}\n"
            f"Audio Duration   : {st.session_state['audio_info']['duration_sec']} seconds\n"
            f"Detected Language: {lang_display} ({detected_lang})\n"
            f"Whisper Model    : {t_data['model_used']}\n"
            f"Execution Time   : {t_data['elapsed_time']} seconds\n"
            f"===========================================================\n\n"
            f"TRANSCRIPTION:\n"
            f"{transcribed_text}\n\n"
            f"===========================================================\n"
            f"DETAILED TIMESTAMPS / SEGMENTS:\n"
            f"===========================================================\n"
        )
        for seg in t_data.get("segments", []):
            start = round(seg.get("start", 0), 2)
            end = round(seg.get("end", 0), 2)
            seg_text = seg.get("text", "").strip()
            download_content += f"[{start:05.2f}s -> {end:05.2f}s] {seg_text}\n"

        clean_filename = st.session_state['audio_info']['filename'].rsplit('.', 1)[0]
        
        with col_down1:
            st.download_button(
                label="📥 Download Transcription (.TXT)",
                data=download_content,
                file_name=f"transcription_{clean_filename}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col_down2:
            show_segments = st.checkbox("Show Timestamped Segments Breakdown", value=False)

        if show_segments and t_data.get("segments"):
            st.markdown("#### Timestamped Segments")
            for seg in t_data["segments"]:
                s_start = f"{seg.get('start', 0):.2f}"
                s_end = f"{seg.get('end', 0):.2f}"
                s_text = seg.get('text', '').strip()
                st.write(f"⏱️ **`{s_start}s - {s_end}s`**: {s_text}")


# =============================================================================
# TAB 2: EDUCATIONAL / HOW ASR WORKS
# =============================================================================
with tab_architecture:
    st.subheader("Understanding Automatic Speech Recognition (ASR)")
    st.markdown("""
    Automatic Speech Recognition (ASR) is a subfield of computer science, machine learning, 
    and natural language processing (NLP) that enables computer systems to identify and 
    translate spoken language into readable text.
    """)

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("### The ASR Pipeline")
        st.markdown("""
        1. **Audio Acquisition & Ingestion**:
           - Sound files in `.wav`, `.mp3`, or `.m4a` format are decoded.
        2. **Signal Preprocessing (Librosa & SoundFile)**:
           - Multi-channel stereo recordings are averaged to **mono**.
           - Audio is resampled to a standardized **16,000 Hz (16 kHz)** sampling rate.
           - Audio amplitude is normalized to standard float32 values between `[-1.0, 1.0]`.
        3. **Acoustic Feature Extraction**:
           - Continuous sound waves are converted into an 80-channel **log-Mel spectrogram**,
             which visually represents frequency distributions over time matching human auditory perception.
        """)

    with col_a2:
        st.markdown("### OpenAI Whisper Architecture")
        st.markdown("""
        - **Encoder-Decoder Transformer**: Whisper utilizes a sequence-to-sequence Transformer architecture.
        - **Audio Encoder**: Encodes the 80-channel log-Mel spectrogram into high-dimensional hidden representations.
        - **Autoregressive Text Decoder**: Predicts corresponding text tokens sequentially, including:
          - Language identification tokens (e.g., `<|en|>`).
          - Phrase-level timestamp tokens.
          - Text transcription tokens.
        - **Weak Supervision Training**: Trained on 680,000 hours of multilingual, multi-task supervised audio data.
        """)

    st.markdown("---")
    st.markdown("### Model Size Comparison")
    st.markdown("""
    | Model | Parameters | Required VRAM / RAM | Relative Speed | Target Use Case |
    | :--- | :--- | :--- | :--- | :--- |
    | **tiny** | 39 M | ~1 GB | ~32x | Ultra-fast embedded devices |
    | **base (Selected)** | **74 M** | **~1 GB** | **~16x** | **Optimal for student laptops & CPU execution** |
    | **small** | 244 M | ~2 GB | ~6x | High-accuracy offline transcription |
    | **medium** | 769 M | ~5 GB | ~2x | High-complexity multi-lingual environments |
    | **large** | 1550 M | ~10 GB | ~1x | State-of-the-art production servers |
    """)


# =============================================================================
# TAB 3: USER GUIDE & TESTING INSTRUCTIONS
# =============================================================================
with tab_help:
    st.subheader("Step-by-Step User Instructions")
    st.markdown("""
    1. **Upload Audio**: Click on **Browse files** or drag and drop an audio file (`.wav`, `.mp3`, or `.m4a`).
    2. **Inspect & Play**: Listen to the audio using the built-in player and verify the audio duration and sample rate.
    3. **Start Transcription**: Click the **🚀 Start Transcription** button.
    4. **View Output**: The detected language and transcribed speech will appear within seconds.
    5. **Export Text**: Click **📥 Download Transcription (.TXT)** to save the transcribed report to your local drive.
    6. **Reset**: Use the **🗑️ Clear Audio & Results** button in the sidebar to reset the session anytime.
    """)

    st.markdown("---")
    st.subheader("Test Cases Included in the Project")
    st.markdown("""
    The repository contains pre-recorded test cases in the `sample_audio/` directory:
    - `short_sample.wav`: Tests quick inference on short spoken sentences (~5 sec).
    - `long_sample.wav`: Tests longer speech segment processing (~18 sec).
    - `sample_audio.mp3`: Tests MP3 format decoding and transcription.
    - `noisy_sample.wav`: Tests Whisper's acoustic robustness in the presence of noise.
    - `invalid_file.wav`: Tests error handling for corrupted or non-audio files.
    """)


# -----------------------------------------------------------------------------
# 8. Footer Section
# -----------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    <b>Automatic Speech Recognition (ASR) Tool</b> • College Computer Science Engineering Project<br>
    Developed with Python, OpenAI Whisper, Streamlit, PyTorch, Librosa & SoundFile • Runs 100% Locally on CPU
</div>
""", unsafe_allow_html=True)
