import re
import os
import requests
import math

import static_ffmpeg
static_ffmpeg.add_paths()

import yt_dlp
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv("WHISPER_ENDPOINT")
API_KEY = os.getenv("WHISPER_API_KEY")
MAX_SIZE = 25_000_000  # 25 MB limit

def sanitize_filename(s: str, replace: str = "_") -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', replace, s)

def download_audio(url: str, output_dir: str = "downloaded") -> str:
    print("⏳ Downloading audio...")
    os.makedirs(output_dir, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192"
        }],
        "noplaylist": True,
        "quiet": False
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    mp3 = info['requested_downloads'][0]['filepath']
    clean = sanitize_filename(os.path.splitext(os.path.basename(mp3))[0])
    clean_path = os.path.join(output_dir, clean + ".mp3")
    if mp3 != clean_path:
        os.rename(mp3, clean_path)
    print(f"✅ Audio saved: {clean_path}")
    return clean_path

def split_audio(mp3_path: str, part_dir="chunks") -> list[str]:
    size = os.path.getsize(mp3_path)
    if size <= MAX_SIZE:
        return [mp3_path]
    os.makedirs(part_dir, exist_ok=True)
    duration = float(os.popen(f'ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "{mp3_path}"').read())
    parts = math.ceil(size / MAX_SIZE)
    chunk_duration = duration / parts
    filenames = []
    base = os.path.splitext(os.path.basename(mp3_path))[0]
    for i in range(parts):
        out = os.path.join(part_dir, f"{base}_part{i+1}.mp3")
        cmd = f'ffmpeg -y -i "{mp3_path}" -ss {i*chunk_duration} -t {chunk_duration} -c copy "{out}"'
        os.system(cmd)
        filenames.append(out)
        print(f"✅ Created chunk: {out}")
    return filenames

def transcribe_file(path: str) -> str:
    print(f"📡 Transcribing chunk {os.path.basename(path)}")
    headers = {"api-key": API_KEY}
    with open(path, "rb") as f:
        resp = requests.post(ENDPOINT, headers=headers, files={"file": (os.path.basename(path), f, "audio/mpeg")})
    if not resp.ok:
        raise RuntimeError(f"❌ Chunk failed: {resp.status_code} {resp.text}")
    return resp.json().get("text", "")

def main():
    url = input("YouTube URL: ").strip()
    mp3 = download_audio(url)
    chunks = split_audio(mp3)
    full_text = ""
    for chunk in chunks:
        full_text += transcribe_file(chunk) + "\n"
    out = os.path.splitext(mp3)[0] + ".txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"🎉 Transcript saved: {out}")

if __name__ == "__main__":
    main()
