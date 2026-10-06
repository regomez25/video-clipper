import os
import json
import sys
import math
import subprocess
import yt_dlp
from openai import OpenAI

# -------------------------------------------------------------------
# 1. READ ENVIRONMENT VARIABLES & SETUP MULTI-PROVIDER CONFIGS
# -------------------------------------------------------------------
YOUTUBE_URL = os.getenv("TARGET_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not YOUTUBE_URL:
    print("❌ Error: TARGET_URL environment variable is missing.")
    sys.exit(1)

MEDIA_DIR = "./media"
OUTPUT_DIR = "./output"
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define Provider Cascade Hierarchy
PROVIDERS = []

if GROQ_API_KEY:
    PROVIDERS.append({
        "name": "Groq",
        "client": OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1"),
        "preferred_models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "qwen-2.5-coder-32b",
            "deepseek-r1-distill-llama-70b"
        ],
        "whisper_model": "whisper-large-v3-turbo"
    })

if OPENAI_API_KEY:
    PROVIDERS.append({
        "name": "OpenAI",
        "client": OpenAI(api_key=OPENAI_API_KEY),
        "preferred_models": ["gpt-4o-mini", "gpt-4o"],
        "whisper_model": "whisper-1"
    })

if OPENROUTER_API_KEY:
    PROVIDERS.append({
        "name": "OpenRouter",
        "client": OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1"),
        "preferred_models": [
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-r1",
            "anthropic/claude-3.5-haiku"
        ],
        "whisper_model": None  # OpenRouter is LLM only
    })

if not PROVIDERS:
    print("❌ Error: No valid API keys found (GROQ_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY).")
    sys.exit(1)

print(f"🔗 Loaded Provider Cascade: {[p['name'] for p in PROVIDERS]}")

EXCLUDED_KEYWORDS = ["guard", "prompt-guard", "whisper", "embedding", "safetensors", "moderation", "vision"]

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
        'extractor_args': {'youtube': {'player_client': ['android', 'ios']}}
    }

    ydl_opts_video = {
        **ydl_opts_base,
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best',
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
# 3. AUDIO SPLITTING HELPER
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
# 4. TRANSCRIPTION WITH FALLBACK
# -------------------------------------------------------------------
def transcribe_audio(audio_path):
    print("🎙 Transcribing audio via API...")
    audio_chunks = split_audio_into_chunks(audio_path)
    combined_segments = []

    # Find first provider that supports Whisper transcription
    audio_providers = [p for p in PROVIDERS if p.get("whisper_model")]
    if not audio_providers:
        raise RuntimeError("❌ No provider available supporting Whisper transcription.")

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
                print(f"⚠️️ Whisper failed on {provider['name']}: {e}. Trying next provider...")

        if not transcription_success:
            raise RuntimeError(f"❌ Failed to transcribe chunk {chunk_path} across all providers.")

    return combined_segments

# -------------------------------------------------------------------
# 5. LLM MULTI-PROVIDER FALLBACK EXTRACTION
# -------------------------------------------------------------------
def is_valid_chat_model(model_id):
    model_id_lower = model_id.lower()
    return not any(excluded in model_id_lower for excluded in EXCLUDED_KEYWORDS)

def format_transcript_for_llm(segments):
    return "\n".join([f"[{s['start']}s - {s['end']}s] {s['text']}" for s in segments])

def get_viral_timestamps(transcript_segments):
    print("🤖 Asking LLM to pick top viral moments...")

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

    # Cascade across providers in defined order (Groq -> OpenAI -> OpenRouter)
    for provider in PROVIDERS:
        provider_name = provider["name"]
        client = provider["client"]
        preferred = provider["preferred_models"]

        print(f"🔄 Evaluating Provider: {provider_name}...")

        # Discover active models on current provider
        candidate_models = preferred
        try:
            models_response = client.models.list()
            active_ids = {m.id for m in models_response.data if is_valid_chat_model(m.id)}
            valid_candidates = [m for m in preferred if m in active_ids]
            if valid_candidates:
                candidate_models = valid_candidates
        except Exception:
            pass  # Fallback to preferred list if list API call fails

        for model_name in candidate_models:
            try:
                print(f"  └─ [{provider_name}] Requesting model: {model_name}...")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert video editor. Return valid JSON only containing viral clips with exact start and end timestamps."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )

                raw_content = response.choices[0].message.content
                print(f"✅ Success using {provider_name} ({model_name})")
                return json.loads(raw_content)

            except Exception as e:
                print(f"⚠️ [{provider_name}] Model {model_name} failed: {e}")
                last_error = e

    raise RuntimeError(f"❌ All providers and candidate models failed. Last error: {last_error}")

# -------------------------------------------------------------------
# 6. FFMPEG CROP & RENDER (9:16 VERTICAL)
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