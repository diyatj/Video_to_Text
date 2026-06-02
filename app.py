import streamlit as st
from faster_whisper import WhisperModel
import tempfile
import os
import subprocess

st.title("Video → Transcript (Large Files Supported)")

model = WhisperModel("base", device="cpu", compute_type="int8")

video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

if video_file:
    st.video(video_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.info("Extracting audio...")
    audio_path = video_path.replace(".mp4", ".wav")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-ar", "16000", "-ac", "1",
        "-vn", audio_path
    ])

    st.info("Transcribing...")
    segments, _ = model.transcribe(audio_path)
    transcript = " ".join([s.text for s in segments])

    st.subheader("Transcript")
    st.write(transcript)
