import os
import json
import sys
import math
import subprocess
import yt_dlp
from openai import OpenAI

# -------------------------------------------------------------------
# 1. READ ENVIRONMENT VARIABLES FROM GITHUB ACTIONS
# -------------------------------------------------------------------
YOUTUBE_URL = os.getenv("TARGET_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not YOUTUBE_URL:
    print("❌ Error: TARGET_URL environment variable is missing.")
    sys.exit(1)

# API Engine Selection
if GROQ_API_KEY:
    print("🚀 Initializing Groq API Engine (Primary)...")
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
    LLM_MODEL = "llama-3.1-8b-instant"
    WHISPER_MODEL = "whisper-large-v3-turbo"
elif OPENAI_API_KEY:
    print("🚀 Initializing OpenAI API Engine (Primary)...")
    client = OpenAI(api_key=OPENAI_API_KEY)
    LLM_MODEL = "gpt-4o-mini"
    WHISPER_MODEL = "whisper-1"
else:
    print("❌ Error: Neither GROQ_API_KEY nor OPENAI_API_KEY is configured.")
    sys.exit(1)

MEDIA_DIR = "./media"
OUTPUT_DIR = "./output"
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 2. DOWNLOAD YOUTUBE MEDIA
# -------------------------------------------------------------------
def download_media(url):
    print(f"📥 Downloading source video: {url}")
    video_path = os.path.join(MEDIA_DIR, "input_video.mp4")
    audio_base = os.path.join(MEDIA_DIR, "input_audio")
    audio_path = f"{audio_base}.mp3"

    ydl_opts_base = {
        'geo_bypass': True,
        'nocheckcertificate': True,
        'quiet': False,
        'no_warnings': False,
        'user_agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36',
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'ios']
            }
        }
    }

    # 1. Download Video Stream
    ydl_opts_video = {
        **ydl_opts_base,
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best',
        'outtmpl': video_path,
        'merge_output_format': 'mp4',
        'overwrites': True,
    }

    # 2. Download & Post-Process Audio Stream (using 64k bitrate for smaller file size)
    ydl_opts_audio = {
        **ydl_opts_base,
        'format': 'bestaudio/best',
        'outtmpl': audio_base,
        'overwrites': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '64',
        }],
    }

    print("🎬 Downloading MP4 video stream...")
    with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
        ydl.download([url])

    print("🎵 Extracting MP3 audio stream...")
    with yt_dlp.YoutubeDL(ydl_opts_audio) as ydl:
        ydl.download([url])

    return video_path, audio_path

# -------------------------------------------------------------------
# 3. AUDIO SPLITTING HELPER (FOR FILES > 24MB)
# -------------------------------------------------------------------
def get_audio_duration(audio_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return float(result.stdout.strip())

def split_audio_into_chunks(audio_path, max_size_mb=24):
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    if file_size_mb <= max_size_mb:
        return [audio_path]

    total_duration = get_audio_duration(audio_path)
    num_chunks = math.ceil(file_size_mb / max_size_mb)
    chunk_duration = total_duration / num_chunks

    chunk_paths = []
    print(f"📦 Audio is {file_size_mb:.2f} MB. Splitting into {num_chunks} chunks (~{chunk_duration:.0f}s each)...")

    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_path = os.path.join(MEDIA_DIR, f"audio_chunk_{i+1}.mp3")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", audio_path,
            "-t", str(chunk_duration),
            "-c", "copy",
            chunk_path
        ]
        subprocess.run(cmd, check=True)
        chunk_paths.append((chunk_path, start_time))

    return chunk_paths

# -------------------------------------------------------------------
# 4. CLOUD TRANSCRIPTION (WHISPER API)
# -------------------------------------------------------------------
def transcribe_audio(audio_path):
    print("🎙 Transcribing audio via API...")
    audio_chunks = split_audio_into_chunks(audio_path)
    combined_segments = []

    for item in audio_chunks:
        if isinstance(item, tuple):
            chunk_path, time_offset = item
        else:
            chunk_path, time_offset = item, 0.0

        print(f"  └─ Uploading {os.path.basename(chunk_path)}...")
        with open(chunk_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(chunk_path), f.read()),
                model=WHISPER_MODEL,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        raw_segments = getattr(transcription, "segments", []) if hasattr(transcription, "segments") else transcription.get("segments", [])

        for seg in raw_segments:
            start = seg.get("start") if isinstance(seg, dict) else getattr(seg, "start")
            end = seg.get("end") if isinstance(seg, dict) else getattr(seg, "end")
            text = seg.get("text") if isinstance(seg, dict) else getattr(seg, "text")

            combined_segments.append({
                "start": round(start + time_offset, 2),
                "end": round(end + time_offset, 2),
                "text": text.strip()
            })

    return combined_segments

# -------------------------------------------------------------------
# 5. LLM VIRAL TIMESTAMP EXTRACTION
# -------------------------------------------------------------------
def get_viral_timestamps(transcript):
    print("🤖 Asking LLM to pick top viral moments...")

    prompt = f"""
    Analyze the following transcript and extract 1-3 highly engaging short clip segments (30-60 seconds each).
    Return a JSON object with a key 'clips' containing an array of objects with 'start', 'end', and 'title'.

    Transcript:
    {transcript}
    """

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert video editor picking viral clips. Output strict JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

# -------------------------------------------------------------------
# 6. FFMPEG CROP & RENDER (9:16 VERTICAL)
# -------------------------------------------------------------------
def render_vertical_clip(video_path, start_time, end_time, output_path):
    duration = end_time - start_time
    print(f"🎬 Rendering vertical clip ({start_time}s to {end_time}s)...")

    # Crop 16:9 source into centered 9:16 vertical frame (1080x1920)
    filter_complex = "crop=ih*(9/16):ih:(iw-ih*(9/16))/2:0,scale=1080:1920"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-vf", filter_complex,
        "-c:v", "libx264", "-crf", "22", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ]

    subprocess.run(cmd, check=True)
    print(f"✅ Rendered clip saved to: {output_path}")

# -------------------------------------------------------------------
# MAIN PIPELINE RUNNER
# -------------------------------------------------------------------
def main():
    video_file, audio_file = download_media(YOUTUBE_URL)
    transcript = transcribe_audio(audio_file)
    clip_data = get_viral_timestamps(transcript)

    for i, clip in enumerate(clip_data.get("clips", [])):
        output_file = os.path.join(OUTPUT_DIR, f"clip_{i+1}.mp4")
        render_vertical_clip(
            video_path=video_file,
            start_time=clip["start"],
            end_time=clip["end"],
            output_path=output_file
        )

if __name__ == "__main__":
    main()