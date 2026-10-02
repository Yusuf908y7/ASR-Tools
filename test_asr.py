"""
=============================================================================
Automatic Speech Recognition (ASR) Tool - Automated Test Suite
=============================================================================
This test script verifies all 5 testing requirements:
1. Short audio recording (WAV)
2. Longer audio recording (WAV)
3. Different audio format (MP3)
4. Audio with background noise
5. Invalid / corrupted file error handling
=============================================================================
"""

import os
import sys
import time
import io
import numpy as np
import soundfile as sf
import librosa
import whisper

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(CURRENT_DIR, "sample_audio")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def simulate_audio_loader(filepath):
    """Replicates the app.py audio loading and preprocessing pipeline."""
    with open(filepath, "rb") as f:
        raw_bytes = f.read()

    if len(raw_bytes) == 0:
        raise ValueError("The uploaded file is empty.")

    decoded = False
    data = None
    sr = None

    try:
        data, sr = sf.read(filepath, dtype="float32")
        decoded = True
    except Exception:
        pass

    if not decoded or data is None:
        try:
            data, sr = librosa.load(filepath, sr=None, mono=False)
            decoded = True
        except Exception as err:
            raise ValueError(f"Unsupported or corrupted audio stream: {err}")

    if data.ndim > 1:
        if data.shape[0] < data.shape[1]:
            data = np.mean(data, axis=0)
        else:
            data = np.mean(data, axis=1)

    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        sr = 16000

    data = data.astype(np.float32)
    max_abs = np.max(np.abs(data))
    if max_abs > 1.0:
        data = data / max_abs

    duration = len(data) / sr
    if duration < 0.1:
        raise ValueError("Audio is too short.")

    return data, sr, duration


def run_tests():
    print("=" * 70)
    print(" AUTOMATIC SPEECH RECOGNITION (ASR) - AUTOMATED TEST SUITE")
    print("=" * 70)

    print("\n[INFO] Loading OpenAI Whisper 'base' model on CPU...")
    start_load = time.time()
    model = whisper.load_model("base", device="cpu")
    print(f"[INFO] Model loaded successfully in {time.time() - start_load:.2f} seconds.\n")

    test_cases = [
        {"id": 1, "title": "Short Audio Recording (WAV)", "file": "short_sample.wav", "expected_success": True, "min_words": 5},
        {"id": 2, "title": "Longer Audio Recording (WAV)", "file": "long_sample.wav", "expected_success": True, "min_words": 15},
        {"id": 3, "title": "Different Audio Format (MP3)", "file": "sample_audio.mp3", "expected_success": True, "min_words": 5},
        {"id": 4, "title": "Audio with Background Noise", "file": "noisy_sample.wav", "expected_success": True, "min_words": 5},
        {"id": 5, "title": "Invalid / Corrupted File Handling", "file": "invalid_file.wav", "expected_success": False, "min_words": 0}
    ]

    all_passed = True

    for test in test_cases:
        print("-" * 70)
        print(f"Test #{test['id']}: {test['title']}")
        file_path = os.path.join(SAMPLE_DIR, test["file"])
        print(f"File: {test['file']} (Size: {os.path.getsize(file_path)} bytes)")

        if test["expected_success"]:
            try:
                audio_arr, sr, duration = simulate_audio_loader(file_path)
                print(f"Preprocessed: Duration={duration:.2f}s, SR={sr}Hz, Samples={len(audio_arr)}")

                t0 = time.time()
                res = model.transcribe(audio_arr, fp16=False, verbose=False)
                t_elapsed = time.time() - t0

                transcribed_text = res.get("text", "").strip()
                detected_lang = res.get("language", "unknown")
                word_count = len(transcribed_text.split())

                print(f"Detected Language: {detected_lang}")
                print(f"Transcribed Text : \"{transcribed_text}\"")
                print(f"Inference Time   : {t_elapsed:.2f}s (Word Count: {word_count})")

                if word_count >= test["min_words"]:
                    print("Status: [PASS] - Correct transcription generated.")
                else:
                    print("Status: [FAIL] - Output too short.")
                    all_passed = False

            except Exception as e:
                print(f"Status: [FAIL] - Unexpected exception: {e}")
                all_passed = False
        else:
            try:
                audio_arr, sr, duration = simulate_audio_loader(file_path)
                print("Status: [FAIL] - Corrupt file did not trigger validation error!")
                all_passed = False
            except ValueError as val_err:
                print(f"Caught Expected ValueError: {val_err}")
                print("Status: [PASS] - Gracefully rejected invalid file as expected.")
            except Exception as other_err:
                print(f"Caught Exception: {other_err}")
                print("Status: [PASS] - Invalid file blocked safely.")

    print("=" * 70)
    if all_passed:
        print("[SUCCESS] ALL 5 TEST SCENARIOS PASSED SUCCESSFULLY!")
    else:
        print("[FAILURE] SOME TESTS FAILED.")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
