Automatic Speech Recognition (ASR) Tool Using Python and OpenAI Whisper

An academic computer science engineering project implementing an Automatic Speech Recognition (ASR) web application powered by OpenAI Whisper, Streamlit, Librosa, and SoundFile. This tool runs entirely on local CPU hardware without requiring paid third-party API keys or an active internet connection.


Table of Contents
Project Title
Abstract
Introduction
Problem Statement
Objectives
Features
Tools and Technologies Used
System Requirements
Project Folder Structure
Installation Instructions
Prerequisites
Virtual Environment Setup
Dependency Installation
How to Run the Application
How to Use the Application
Testing Procedure
Automated Test Suite
Manual Test Cases
Expected Output
Limitations
Future Enhancements
GitHub Repository Setup & Push Instructions
Assignment Requirement Checklist
Conclusion
Project Title

Automatic Speech Recognition (ASR) Tool Using Python and OpenAI Whisper

Abstract

Automatic Speech Recognition (ASR) represents a foundational pillar in human-computer interaction, bridging auditory communications with computational text processing. This project presents a standalone, offline web application capable of transcribing spoken audio into high-fidelity textual transcriptions. By combining OpenAI's open-source Whisper model with a Streamlit interface and robust audio signal processing via Librosa and SoundFile, the system enables audio ingestion across standard formats (WAV, MP3, M4A), audio waveform visualization, automated language identification, and text export. The system is explicitly configured for standard CPU architectures, ensuring broad accessibility and reproducibility for academic evaluations.

Introduction

Speech is the most intuitive and primary method of human interaction. Traditional automated speech recognition architectures often depended on complex, multi-stage pipelines (e.g., Hidden Markov Models, separate acoustic models, pronunciation lexicons, and n-gram language models).

With recent breakthroughs in deep learning and self-supervised sequence-to-sequence Transformer architectures, modern ASR models unify acoustic feature extraction and text prediction into an end-to-end network. This project leverages OpenAI's open-source Whisper model to provide a self-contained, user-centric transcription suite with an interactive web UI.

Problem Statement

While high-quality speech-to-text models exist, real-world deployment in educational, research, and resource-constrained environments faces three key bottlenecks:

API Paywalls & Privacy: Cloud-based services (e.g., Google Cloud Speech, AWS Transcribe, OpenAI Whisper API) require recurring billing, API tokens, and stream sensitive voice recordings to remote servers.
Hardware Constraints: Large neural models frequently require high-end, dedicated GPUs with large VRAM capacities, which are unavailable on standard student laptops.
Complex Interfaces: Command-line utilities are inaccessible to non-technical users who require intuitive playback, real-time waveform inspection, and one-click report downloads.

This project addresses these challenges by delivering an open-source, local, CPU-optimized web solution.

Objectives
Build an Interactive Web Interface: Design an intuitive, modern Streamlit UI for seamless audio uploads and transcription display.
Support Multi-Format Audio Ingestion: Accept popular audio formats including .wav, .mp3, and .m4a.
Implement Robust Local ASR: Transcribe speech locally using the Whisper base model without external API requests.
Provide Signal Inspection: Display playback controls, duration, sample rate, file size metrics, and graphical waveform visualizations.
Deliver Multi-Language Identification: Automatically identify the spoken language from the input acoustic signal.
Enable Formatted Export: Provide one-click text file export containing both continuous transcriptions and timestamped segment logs.
Ensure Fault Tolerance: Handle corrupted files, silent streams, and unsupported codecs gracefully with informative user feedback.
Features
Modern User Interface: Built using Streamlit with custom CSS, visual metrics cards, and responsive layout.
Universal Audio Upload: Ingests WAV, MP3, and M4A audio files up to 200 MB.
Interactive Audio Player: Built-in audio playback component to preview speech before and after transcription.
Audio Preprocessing Pipeline: Automatic stereo-to-mono downmixing, amplitude normalization, and 16,000 Hz resampling via SoundFile and Librosa.
Signal Waveform Visualization: Visual representation of the audio signal amplitude across time.
OpenAI Whisper Integration: Utilizes the 74M-parameter base model optimized for CPU inference with fp16=False fallback.
Language Detection: Automatically detects the spoken language (e.g., English, Spanish, French, Hindi) and displays the ISO code and friendly language name.
Timestamped Segments: Expandable segment breakdown displaying sentence-by-sentence timestamps ([00.00s -> 04.86s]).
One-Click TXT Export: Download an organized summary report containing file metadata, full transcription, and timestamped lines.
Clear / Reset Controls: One-click reset button to purge uploaded audio, cached signals, and output results.
Educational Explanations: Dedicated documentation tab explaining the internal workings of the Mel-spectrogram and Transformer encoder-decoder.
Tools and Technologies Used
Category	Technology	Purpose
Language	Python 3.10+ / 3.13	Core programming language
ASR Neural Model	OpenAI Whisper (base)	End-to-end speech recognition and language identification
Machine Learning	PyTorch (torch, torchaudio)	Tensor execution and CPU inference backend
Frontend Framework	Streamlit	Web interface, reactive widgets, and state management
Audio Processing	SoundFile & Librosa	Audio decoding, stereo-to-mono downmixing, and 16 kHz resampling
Visualization	Matplotlib	Signal waveform generation
Version Control	Git & GitHub	Source code tracking and collaboration
System Requirements
Minimum Hardware Requirements
Processor: Intel Core i3 (7th Gen+) / AMD Ryzen 3 or equivalent.
Memory (RAM): 4 GB minimum (8 GB recommended for smooth multi-tasking).
Storage: ~500 MB free disk space (for Python dependencies and model weights).
GPU: Not required (Runs 100% locally on CPU).
Software Requirements
Operating System: Windows 10/11, macOS (11+), or Ubuntu Linux (20.04+).
Python: Version 3.10, 3.11, 3.12, or 3.13.
Web Browser: Google Chrome, Mozilla Firefox, Microsoft Edge, or Safari.
Project Folder Structure
ASR-Tool/
├── app.py                 # Main Streamlit web application
├── test_asr.py            # Automated test suite for college verification
├── requirements.txt       # Project dependencies and library versions
├── README.md              # Comprehensive documentation and setup guide
├── .gitignore             # Git ignore file for cache, venv, and checkpoints
├── sample_audio/          # Test audio suite
│   ├── short_sample.wav   # Short spoken audio test (~5 sec)
│   ├── long_sample.wav    # Longer audio test (~18 sec)
│   ├── sample_audio.mp3   # MP3 format validation audio
│   ├── noisy_sample.wav   # Spoken audio with additive background noise
│   └── invalid_file.wav   # Corrupted/invalid file for error handling tests
└── screenshots/           # Application screenshots and visual assets
    └── app_preview.png    # Dashboard interface preview
Installation Instructions
Prerequisites

Ensure Python is installed on your computer. Verify by opening a terminal:

bash
python --version
Virtual Environment Setup

It is recommended to isolate project dependencies inside a clean virtual environment.

Windows (PowerShell):

powershell
# Navigate into the project folder
cd ASR-Tool
# Create a virtual environment named 'venv'
python -m venv venv
# Activate the virtual environment
.\venv\Scripts\Activate.ps1

macOS / Linux:

bash
# Navigate into the project folder
cd ASR-Tool
# Create a virtual environment
python3 -m venv venv
# Activate the virtual environment
source venv/bin/activate
Dependency Installation

Install all required libraries using the provided requirements.txt:

bash
pip install --upgrade pip
pip install -r requirements.txt
How to Run the Application

Once dependencies are installed, launch the Streamlit application:

bash
streamlit run app.py

Streamlit will start its local development server and open the application in your default browser at:

http://localhost:8501
How to Use the Application
Open the Tool: Launch http://localhost:8501 in your browser.
Upload Audio: Click on Browse files or drag and drop an audio file (.wav, .mp3, or .m4a).
Listen & Inspect:
Play the audio file using the built-in media player.
Review metadata metrics (Duration, File Size, Sample Rate).
Expand the View Audio Signal Waveform tab to view the acoustic wave.
Select Whisper Model: From the sidebar, keep the default base model (or choose tiny/small).
Start Transcription: Click the 🚀 Start Transcription button.
Review Results:
View the recognized textual output.
Observe the detected language badge (e.g., English (en)).
Check the inference time badge (e.g., Processed in 2.57s).
Download Transcription: Click 📥 Download Transcription (.TXT) to save the complete transcript and timestamps to your computer.
Reset / Clear: Click 🗑️ Clear Audio & Results in the sidebar to reset the session.
Testing Procedure
Automated Test Suite

An automated test script (test_asr.py) is included to verify all functional requirements without manual intervention:

bash
python test_asr.py
Manual Test Cases

The application includes five test audio files inside the sample_audio/ directory:

Test Case	Audio File	Purpose	Expected Result
1. Short Audio	short_sample.wav	Verify quick turnaround on short phrases (~5s).	Transcribes "Artificial intelligence is transforming speech recognition technology." with high accuracy in ~2.5s.
2. Long Audio	long_sample.wav	Verify paragraph-length continuous speech (~18s).	Transcribes the full introductory speech with sentence punctuation in ~3.0s.
3. Different Format	sample_audio.mp3	Test non-WAV audio container decoding.	SoundFile decodes MP3 and transcribes the speech accurately.
4. Noisy Audio	noisy_sample.wav	Test Whisper acoustic robustness with background noise.	Transcribes words accurately despite additive Gaussian noise.
5. Invalid File	invalid_file.wav	Test error boundary handling on corrupted files.	Application rejects the invalid audio and presents a clear error message.
Expected Output

When running test_asr.py or uploading the sample files via app.py:

======================================================================
 AUTOMATIC SPEECH RECOGNITION (ASR) - AUTOMATED TEST SUITE
======================================================================
[INFO] Loading OpenAI Whisper 'base' model on CPU...
[INFO] Model loaded successfully in 1.13 seconds.
----------------------------------------------------------------------
Test #1: Short Audio Recording (WAV)
File: short_sample.wav (Size: 214524 bytes)
Preprocessed: Duration=4.86s, SR=16000Hz, Samples=77816
Detected Language: en
Transcribed Text : "Artificial intelligence is transforming speech recognition technology."
Inference Time   : 2.57s (Word Count: 7)
Status: [PASS] - Correct transcription generated.
----------------------------------------------------------------------
Test #2: Longer Audio Recording (WAV)
File: long_sample.wav (Size: 822114 bytes)
Preprocessed: Duration=18.64s, SR=16000Hz, Samples=298256
Detected Language: en
Transcribed Text : "Welcome to the Automatic Speech Recognition Tool Demonstration. This application converts spoken audio into written text using OpenAI Whisper and Modern Python libraries. It runs entirely on the local computer without requiring an external internet connection or paid API keys."
Inference Time   : 2.97s (Word Count: 40)
Status: [PASS] - Correct transcription generated.
----------------------------------------------------------------------
Test #3: Different Audio Format (MP3)
File: sample_audio.mp3 (Size: 60688 bytes)
Preprocessed: Duration=7.11s, SR=16000Hz, Samples=113810
Detected Language: en
Transcribed Text : "This audio file is encoded in MP3 format to verify multi-format speech recognition compatibility."
Inference Time   : 2.40s (Word Count: 14)
Status: [PASS] - Correct transcription generated.
----------------------------------------------------------------------
Test #4: Audio with Background Noise
File: noisy_sample.wav (Size: 214522 bytes)
Preprocessed: Duration=4.86s, SR=16000Hz, Samples=77816
Detected Language: en
Transcribed Text : "Artificial intelligence is transforming speech recognition technology."
Inference Time   : 2.30s (Word Count: 7)
Status: [PASS] - Correct transcription generated.
----------------------------------------------------------------------
Test #5: Invalid / Corrupted File Handling
File: invalid_file.wav (Size: 91 bytes)
Caught Expected ValueError: Unsupported or corrupted audio stream: Format not recognised.
Status: [PASS] - Gracefully rejected invalid file as expected.
======================================================================
[SUCCESS] ALL 5 TEST SCENARIOS PASSED SUCCESSFULLY!
======================================================================
Limitations
High Background Noise: Extremely degraded audio with high signal-to-noise ratios (SNR < 0 dB) may result in minor phonetic hallucinations.
Accents & Domain Jargon: Obscure medical or technical terms may require larger Whisper models (medium or large) for optimal precision.
M4A / AAC on Systems Without Native Codecs: Decoding M4A files on older OS environments may require installing FFmpeg if the system audio backend lacks AAC decoders.
CPU Latency on Multi-Hour Recordings: Transcribing audio over 30 minutes on an older dual-core CPU can take several minutes.
Future Enhancements
Live Microphone Input: Add real-time streaming audio recording via WebRTC or Streamlit audio-recorder component.
Speaker Diarization: Integrate PyAnnote.audio to label distinct speakers (e.g., "Speaker 1", "Speaker 2").
Subtitle Export (.SRT / .VTT): Export timestamped transcription files directly for video synchronization.
Multilingual Translation Mode: Leverage Whisper's translation task to automatically translate foreign speech directly into English.
Summarization Pipeline: Integrate a lightweight open-source LLM (such as LLaMA-3 or Gemma) to generate bullet-point summaries of transcribed lectures or meetings.
GitHub Repository Setup & Push Instructions

Follow these step-by-step commands in your terminal to initialize Git, commit the code, and push the repository to GitHub:

Step 1: Install Git (if not already installed)
Windows: Download from git-scm.com/download/win
 or run winget install --id Git.Git -e --source winget.
macOS: Run brew install git.
Linux (Ubuntu/Debian): Run sudo apt update && sudo apt install git -y.
Step 2: Open Terminal in the Project Directory
bash
cd ASR-Tool
Step 3: Initialize Git Repository
bash
git init
Step 4: Configure Git User Info (First-Time Setup)
bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
Step 5: Stage All Files
bash
git add .
Step 6: Commit the Code
bash
git commit -m "Initial commit: Complete Automatic Speech Recognition (ASR) Tool using Whisper and Streamlit"
Step 7: Create a New GitHub Repository
Log in to GitHub
.
Click the + icon in the top-right corner and select New repository.
Name the repository: ASR-Tool (or speech-to-text-whisper).
Set visibility to Public (or Private based on college guidelines).
Do NOT check "Add a README file" or ".gitignore" (as they are already created).
Click Create repository.
Step 8: Link Remote and Push to GitHub

Copy your repository URL from GitHub and execute:

bash
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/ASR-Tool.git
git push -u origin main
Assignment Requirement Checklist
 Web Interface: Modern, interactive Streamlit frontend with custom CSS.
 Multi-Format Ingestion: Supports .wav, .mp3, and .m4a files.
 Speech-to-Text: Powered by OpenAI Whisper (base model).
 Clear Text Display: Shows formatted transcription and timestamped phrases.
 Action Button: Dedicated "🚀 Start Transcription" button.
 Loading Spinner: Real-time progress and status indicator during processing.
 Error Handling: Graceful rejection of corrupt/empty audio with helpful guidance.
 Report Download: One-click TXT export with metadata and timestamps.
 Language Detection: Automatically detects and displays language code and name.
 Clear/Reset Option: Sidebar button to clear audio, cache, and results.
 100% Local & Free: Runs locally on CPU without paid APIs or cloud keys.
 Robust Audio Processing: Resampling and downmixing via Librosa and SoundFile.
 Complete Documentation: Detailed README with setup, testing, and theory.
Conclusion

This Automatic Speech Recognition project demonstrates an end-to-end, functional, and privacy-preserving machine learning application. By coupling the OpenAI Whisper model with an accessible Streamlit frontend and standard audio processing libraries (Librosa and SoundFile), the system successfully delivers accurate speech-to-text conversion on commodity laptop hardware. The implementation satisfies all functional requirements and testing benchmarks specified for an undergraduate computer science engineering capstone submission.
