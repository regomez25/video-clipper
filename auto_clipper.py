import os
import json
import sys
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

# API Engine Selection (Supports free-tier Groq API or OpenAI)
if GROQ_API_KEY:
    print("🚀 Using Groq API Engine...")
    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    LLM_MODEL = "llama-3.3-70b-versatile"
    WHISPER_MODEL = "whisper-large-v3-turbo"
elif OPENAI_API_KEY:
    print("🚀 Using OpenAI API Engine...")
    client = OpenAI(api_key=OPENAI_API_KEY)
    LLM_MODEL = "gpt-4o-mini"
    WHISPER_MODEL = "whisper-1"
else:
    print("❌ Error: Neither GROQ_API_KEY nor OPENAI_API_KEY is configured in GitHub Secrets.")
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

    # 2. Download & Post-Process Audio Stream
    ydl_opts_audio = {
        **ydl_opts_base,
        'format': 'bestaudio/best',
        'outtmpl': audio_base,  # Omit extension so FFmpeg converts to input_audio.mp3
        'overwrites': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
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
# 3. CLOUD TRANSCRIPTION (WHISPER API)
# -------------------------------------------------------------------
def transcribe_audio(audio_path):
    print("🎙️ Transcribing audio via API...")
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model=WHISPER_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    
    segments = []
    raw_segments = getattr(transcription, "segments", []) if hasattr(transcription, "segments") else transcription.get("segments", [])
    
    for seg in raw_segments:
        start = seg.get("start") if isinstance(seg, dict) else getattr(seg, "start")
        end = seg.get("end") if isinstance(seg, dict) else getattr(seg, "end")
        text = seg.get("text") if isinstance(seg, dict) else getattr(seg, "text")
        
        segments.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "text": text.strip()
        })
    return segments

# -------------------------------------------------------------------
# 4. LLM VIRAL TIMESTAMP EXTRACTION
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
        model="llama-3.3-70b-versatile",  # Supported Groq model
        messages=[
            {"role": "system", "content": "You are an expert video editor picking viral clips. Output strict JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)

# -------------------------------------------------------------------
# 5. FFMPEG CROP & RENDER (9:16 VERTICAL)
# -------------------------------------------------------------------
def render_vertical_clip(video_path, start_time, end_time, output_path):
    duration = end_time - start_time
    print(f"🎬 Rendering vertical clip ({start_time}s to {end_time}s)...")

    # Crop 16:9 1080p source into centered 9:16 vertical frame (1080x1920)
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