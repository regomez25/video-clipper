import os
import json
import sys
import subprocess
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
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
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
    audio_path = os.path.join(MEDIA_DIR, "input_audio.mp3")

    # Anti-Bot Flags optimized for Cloud CI/CD environments
    yt_dlp_common_args = [
        "--user-agent", "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3.1 Mobile/15E148 Safari/604.1",
        "--extractor-args", "youtube:player_client=ios,android_vr,web",
        "--no-check-certificates",
        "--geo-bypass"
    ]

    if os.path.exists("cookies.txt"):
        print("🔑 Found cookies.txt, applying session authentication...")
        yt_dlp_common_args.extend(["--cookies", "cookies.txt"])

    # 1. Download Video
    cmd_video = [
        "yt-dlp",
        *yt_dlp_common_args,
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "--merge-output-format", "mp4",
        "-o", video_path,
        "--force-overwrites",
        url
    ]
    subprocess.run(cmd_video, check=True)

    # 2. Extract Audio
    cmd_audio = [
        "yt-dlp",
        *yt_dlp_common_args,
        "-x", "--audio-format", "mp3",
        "-o", audio_path,
        "--force-overwrites",
        url
    ]
    subprocess.run(cmd_audio, check=True)

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
def get_viral_timestamps(transcript_data):
    print("🤖 Asking LLM to pick top viral moments...")
    prompt = f"""
    You are an expert short-form editor. Analyze this podcast transcript with timestamps.
    Identify the TOP 2 standalone high-value clips (between 30 and 45 seconds long).
    
    Return ONLY a valid JSON object in this exact format:
    {{
      "clips": [
        {{
          "start": 12.5,
          "end": 42.0,
          "title": "Startup_Advice"
        }}
      ]
    }}

    Transcript:
    {json.dumps(transcript_data)}
    """

    response = client.chat.completions.create(
        model=LLM_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
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