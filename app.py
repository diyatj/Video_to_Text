import streamlit as st
import whisper
import tempfile
import os
import subprocess

st.title("Video → Transcript (Large Files Supported)")

model = whisper.load_model("base")

video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

if video_file:
    st.video(video_file)

    # Save video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.info("Extracting audio...")

    audio_path = video_path.replace(".mp4", ".wav")

    # Extract audio using ffmpeg
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-ar", "16000", "-ac", "1",
        "-vn", audio_path
    ])

    st.info("Splitting audio into chunks...")

    chunk_dir = tempfile.mkdtemp()

    # Split into 5-minute chunks
    subprocess.run([
        "ffmpeg", "-i", audio_path,
        "-f", "segment",
        "-segment_time", "300",
        "-c", "copy",
        os.path.join(chunk_dir, "chunk_%03d.wav")
    ])

    transcript = ""

    st.info("Transcribing chunks...")

    for file in sorted(os.listdir(chunk_dir)):
        chunk_path = os.path.join(chunk_dir, file)

        result = model.transcribe(chunk_path)
        transcript += result["text"] + " "

    st.subheader("Transcript")
    st.write(transcript)
