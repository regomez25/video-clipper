import os
import json
import sys
import math
import subprocess
import yt_dlp
from openai import OpenAI

# -------------------------------------------------------------------
# 1. READ ENVIRONMENT & SETUP OPENROUTER CLIENT
# -------------------------------------------------------------------
YOUTUBE_URL = os.getenv("TARGET_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Optional fallback for Whisper transcriptions
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Optional fallback for Groq Whisper transcription


print("OPENROUTER_API_KEY:",OPENROUTER_API_KEY);
print("OPENAI_API_KEY:",OPENAI_API_KEY);
print("GROQ_API_KEY:",GROQ_API_KEY);

if not YOUTUBE_URL:
    print("❌ Error: TARGET_URL environment variable is missing.")
    sys.exit(1)

if not OPENROUTER_API_KEY:
    print("❌ Error: OPENROUTER_API_KEY environment variable is missing.")
    sys.exit(1)

MEDIA_DIR = "./media"
OUTPUT_DIR = "./output"
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize OpenRouter Client
openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/video-clipper",
        "X-Title": "Auto Video Clipper"
    }
)

# Ordered fallback list of OpenRouter models (High quality -> Fast -> Free)
MODEL_FALLBACK_CASCADE = [
    "meta-llama/llama-3.3-70b-instruct",       # High quality Llama 3.3
    "deepseek/deepseek-r1-distill-llama-70b",   # Fast Reasoning
    "qwen/qwen-2.5-72b-instruct",               # Qwen alternative
    "google/gemini-2.5-flash:free",             # High context free fallback
    "openrouter/free"                            # Router auto-selects free models
]

# -------------------------------------------------------------------
# 2. DOWNLOAD YOUTUBE MEDIA
# -------------------------------------------------------------------
def download_media(url):
    print(f"📥 Downloading source video: {url}")
    video_path = os.path.join(MEDIA_DIR, "input_video.mp4")
    audio_base = os.path.join(MEDIA_DIR, "input_audio")
    audio_path = f"{audio_base}.mp3"

    cookie_file = "youtube_cookies.txt" if os.path.exists("youtube_cookies.txt") and os.path.getsize("youtube_cookies.txt") > 0 else None
    
    if cookie_file:
        print("🔑 Using provided YouTube cookies for authentication.")

    ydl_opts_base = {
        'geo_bypass': True,
        'nocheckcertificate': True,
        'quiet': False,
        'no_warnings': False,
        'cookiefile': cookie_file,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
        'extractor_args': {
            'youtube': {
                'player_client': ['tv', 'web_embedded', 'web']
            }
        }
    }

    # Extremely resilient format fallback chain
    ydl_opts_video = {
        **ydl_opts_base,
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/bestvideo+bestaudio/best/mp4',
        'outtmpl': video_path,
        'merge_output_format': 'mp4',
        'overwrites': True,
    }

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
# 3. AUDIO SPLITTING & TRANSCRIPTION
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
        return [(audio_path, 0.0)]

    total_duration = get_audio_duration(audio_path)
    num_chunks = math.ceil(file_size_mb / max_size_mb)
    chunk_duration = total_duration / num_chunks

    chunk_paths = []
    print(f"📦 Audio is {file_size_mb:.2f} MB. Splitting into {num_chunks} chunks...")

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

def transcribe_audio(audio_path):
    print("🎙 Transcribing audio...")
    audio_chunks = split_audio_into_chunks(audio_path)
    combined_segments = []

    # Build transcription client (Prefer Groq for free Whisper, then OpenAI)
    transcribe_client = None
    whisper_model = "whisper-large-v3-turbo"

    if GROQ_API_KEY:
        transcribe_client = OpenAI(
            api_key=GROQ_API_KEY, 
            base_url="https://api.groq.com/openai/v1"
        )
        print("  └─ Using Groq API for Whisper transcription")
    elif OPENAI_API_KEY:
        transcribe_client = OpenAI(api_key=OPENAI_API_KEY)
        whisper_model = "whisper-1"
        print("  └─ Using OpenAI API for Whisper transcription")
    else:
        raise RuntimeError("❌ No valid API key found for audio transcription (GROQ_API_KEY or OPENAI_API_KEY required).")

    for chunk_path, time_offset in audio_chunks:
        print(f"  └─ Uploading {os.path.basename(chunk_path)}...")

        with open(chunk_path, "rb") as f:
            transcription = transcribe_client.audio.transcriptions.create(
                file=(os.path.basename(chunk_path), f.read()),
                model=whisper_model,
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
# 4. OPENROUTER MULTI-MODEL REASONING
# -------------------------------------------------------------------
def format_transcript_for_llm(segments):
    return "\n".join([f"[{s['start']}s - {s['end']}s] {s['text']}" for s in segments])

def get_viral_timestamps(transcript_segments):
    print("🤖 Asking OpenRouter for top viral moments...")

    formatted_transcript = format_transcript_for_llm(transcript_segments)
    max_chars = 48000
    if len(formatted_transcript) > max_chars:
        formatted_transcript = formatted_transcript[:max_chars]

    prompt = f"""
    Analyze the transcript timestamps below and identify 1-3 engaging clip segments (30-60 seconds each).
    Return a JSON object with a key 'clips' containing an array of objects with keys: 'start', 'end', and 'title'.

    Transcript:
    {formatted_transcript}
    """

    last_error = None

    # Iterate through fallback models on OpenRouter
    for model_name in MODEL_FALLBACK_CASCADE:
        try:
            print(f"  └─ Requesting OpenRouter Model: {model_name}...")
            
            response = openrouter_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert video editor. Return valid JSON only containing viral clips with exact start and end timestamps."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            raw_content = response.choices[0].message.content
            
            # Clean possible markdown wrapping (e.g. ```json ... ```)
            clean_json = raw_content.replace("```json", "").replace("```", "").strip()
            parsed_data = json.loads(clean_json)

            print(f"✅ Success using OpenRouter model: {model_name}")
            return parsed_data

        except Exception as e:
            print(f"⚠️️ Model {model_name} failed: {e}")
            last_error = e

    raise RuntimeError(f"❌ All OpenRouter fallback models failed. Last error: {last_error}")

# -------------------------------------------------------------------
# 5. FFMPEG CROP & RENDER (9:16 VERTICAL)
# -------------------------------------------------------------------
def render_vertical_clip(video_path, start_time, end_time, output_path):
    duration = end_time - start_time
    print(f"🎬 Rendering vertical clip ({start_time}s to {end_time}s)...")

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
    transcript_segments = transcribe_audio(audio_file)
    clip_data = get_viral_timestamps(transcript_segments)

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