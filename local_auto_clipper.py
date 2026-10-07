import os
import json
import sys
import math
import subprocess
from openai import OpenAI

# -------------------------------------------------------------------
# 1. CONFIGURATION & PROVIDERS SETUP
# -------------------------------------------------------------------
YOUTUBE_URL = os.getenv("TARGET_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not YOUTUBE_URL:
    print("❌ Error: TARGET_URL environment variable is missing.")
    print("Usage: export TARGET_URL='https://www.youtube.com/watch?v=...' && python auto_clipper.py")
    sys.exit(1)

MEDIA_DIR = "./media"
OUTPUT_DIR = "./output"
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROVIDERS = []

# --- PROVIDER 1: GROQ ---
if GROQ_API_KEY:
    PROVIDERS.append({
        "name": "Groq",
        "client": OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"),
        "whisper_model": "whisper-large-v3-turbo",
        "preferred_models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen-2.5-32b"]
    })

# --- PROVIDER 2: OPENROUTER ---
if OPENROUTER_API_KEY:
    PROVIDERS.append({
        "name": "OpenRouter",
        "client": OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://github.com/video-clipper", "X-Title": "Auto Video Clipper"}
        ),
        "whisper_model": None,
        "preferred_models": [
            "meta-llama/llama-3.3-70b-instruct",
            "google/gemini-2.5-flash:free",
            "openrouter/free"
        ]
    })

# --- PROVIDER 3: OPENAI ---
if OPENAI_API_KEY:
    PROVIDERS.append({
        "name": "OpenAI",
        "client": OpenAI(api_key=OPENAI_API_KEY),
        "whisper_model": "whisper-1",
        "preferred_models": ["gpt-4o-mini", "gpt-4o"]
    })

if not PROVIDERS:
    print("❌ Error: No API keys provided. Set GROQ_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
    sys.exit(1)

# -------------------------------------------------------------------
# 2. DOWNLOAD FULL AUDIO USING YT-DLP CLI
# -------------------------------------------------------------------
def download_audio_cli(url):
    print("🎵 Extracting full audio stream via yt-dlp CLI...")
    audio_path = os.path.join(MEDIA_DIR, "full_audio.mp3")

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "64K",
        "-o", audio_path,
        "--force-overwrites",
        url
    ]
    subprocess.run(cmd, check=True)
    return audio_path

# -------------------------------------------------------------------
# 3. TRANSCRIPTION & CHUNKING
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
    print(f"📦 Audio file is {file_size_mb:.2f} MB. Splitting into {num_chunks} chunks for API limits...")

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

    audio_providers = [p for p in PROVIDERS if p.get("whisper_model")]
    if not audio_providers:
        raise RuntimeError("❌ No provider available supporting audio transcription (GROQ_API_KEY or OPENAI_API_KEY required).")

    for chunk_path, time_offset in audio_chunks:
        print(f"  └─ Uploading {os.path.basename(chunk_path)}...")
        transcription_success = False

        for provider in audio_providers:
            try:
                print(f"     Attempting Whisper with {provider['name']}...")
                with open(chunk_path, "rb") as f:
                    transcription = provider["client"].audio.transcriptions.create(
                        file=(os.path.basename(chunk_path), f.read()),
                        model=provider["whisper_model"],
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

                transcription_success = True
                break

            except Exception as e:
                print(f"⚠️ Whisper failed on {provider['name']}: {e}. Trying next provider...")

        if not transcription_success:
            raise RuntimeError(f"❌ Failed to transcribe chunk {chunk_path}.")

    return combined_segments

# -------------------------------------------------------------------
# 4. LLM EXTRACTION FOR VIRAL MOMENTS
# -------------------------------------------------------------------
def seconds_to_hhmmss(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"

def get_viral_timestamps(transcript_segments):
    print("🤖 Analyzing transcript for top viral clips...")

    formatted_transcript = "\n".join([f"[{s['start']}s - {s['end']}s] {s['text']}" for s in transcript_segments])
    max_chars = 48000
    if len(formatted_transcript) > max_chars:
        formatted_transcript = formatted_transcript[:max_chars]

    prompt = f"""
    Analyze the podcast transcript below and pick 3 to 5 engaging, viral-worthy clip segments (30-90 seconds each).
    Return a JSON object with a key 'clips' containing an array of objects.
    Each object must have:
      - 'start_seconds': start time in float seconds (e.g., 2070.0)
      - 'end_seconds': end time in float seconds (e.g., 2150.0)
      - 'title': A short catchy title in the video's language.

    Transcript:
    {formatted_transcript}
    """

    for provider in PROVIDERS:
        for model_name in provider["preferred_models"]:
            try:
                print(f"  └─ Requesting LLM from {provider['name']} ({model_name})...")
                response = provider["client"].chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a professional social media video editor. Return valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )

                raw_content = response.choices[0].message.content
                clean_json = raw_content.replace("```json", "").replace("```", "").strip()
                return json.loads(clean_json)

            except Exception as e:
                print(f"⚠️ [{provider['name']}] Model {model_name} failed: {e}")

    raise RuntimeError("❌ All LLM models failed to extract viral clips.")

# -------------------------------------------------------------------
# 5. SECTION DOWNLOAD & 9:16 CROP RENDER
# -------------------------------------------------------------------
def download_and_crop_clip(url, clip, index):
    start_sec = clip["start_seconds"]
    end_sec = clip["end_seconds"]
    title = clip.get("title", f"Clip {index+1}")

    start_str = seconds_to_hhmmss(start_sec)
    end_str = seconds_to_hhmmss(end_sec)
    time_range = f"*{start_str}-{end_str}"

    raw_clip_path = os.path.join(MEDIA_DIR, f"raw_clip_{index+1}.mp4")
    final_output_path = os.path.join(OUTPUT_DIR, f"Clip {index+1} — {title}.mp4")

    print(f"\n🎬 Downloading clip {index+1} [{start_str} to {end_str}] via yt-dlp section download...")

    # Exact format string verified working on your machine
    yt_cmd = [
        "yt-dlp",
        "--download-sections", time_range,
        "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "--merge-output-format", "mp4",
        "--force-keyframes-at-cuts",
        "-o", raw_clip_path,
        "--force-overwrites",
        url
    ]
    subprocess.run(yt_cmd, check=True)

    print(f"📐 Cropping to 9:16 vertical format (1080x1920)...")
    crop_filter = "crop=ih*(9/16):ih:(iw-ih*(9/16))/2:0,scale=1080:1920"

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", raw_clip_path,
        "-vf", crop_filter,
        "-c:v", "libx264", "-crf", "22", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        final_output_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"✅ Finished: {final_output_path}")

# -------------------------------------------------------------------
# MAIN RUNNER
# -------------------------------------------------------------------
def main():
    audio_file = download_audio_cli(YOUTUBE_URL)
    segments = transcribe_audio(audio_file)
    clip_data = get_viral_timestamps(segments)

    clips = clip_data.get("clips", [])
    print(f"\n🎯 Found {len(clips)} viral moments!")

    for i, clip in enumerate(clips):
        download_and_crop_clip(YOUTUBE_URL, clip, i)

    print("\n🎉 All clips generated successfully in ./output directory!")

if __name__ == "__main__":
    main()