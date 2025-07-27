import os, re, time, uuid, subprocess, requests, json
import static_ffmpeg; static_ffmpeg.add_paths()
import yt_dlp
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

load_dotenv()

def sanitize_filename(s: str, replace="_"):
    return re.sub(r'[<>:"/\\|?*\x00-\x1F]', replace, s)

def download_audio(url: str, output_dir="downloaded") -> str:
    os.makedirs(output_dir, exist_ok=True)
    print("⏳ Downloading audio...")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "postprocessors":[{"key":"FFmpegExtractAudio","preferredcodec":"mp3","preferredquality":"192"}],
        "noplaylist": True, "quiet": False
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    mp3 = info["requested_downloads"][0]["filepath"]
    clean = sanitize_filename(os.path.splitext(os.path.basename(mp3))[0])
    clean_path = os.path.join(output_dir, clean + ".mp3")
    if mp3 != clean_path: os.rename(mp3, clean_path)
    print(f"✅ Audio saved: {clean_path}")
    return clean_path

def convert_to_mono(src: str) -> str:
    base = os.path.splitext(src)[0]
    mono = f"{base}_mono.mp3"
    print("🔧 Converting to mono 16 kHz for diarization...")
    subprocess.run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", mono], check=True)
    print(f"✅ Converted to mono: {mono}")
    return mono

def upload_to_blob(filepath: str) -> str:
    svc = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    cont_cl = svc.get_container_client(os.getenv("AZURE_STORAGE_CONTAINER"))
    try: cont_cl.create_container()
    except ResourceExistsError: pass
    blob = cont_cl.get_blob_client(os.path.basename(filepath))
    with open(filepath, "rb") as f: blob.upload_blob(f, overwrite=True)
    print(f"✅ Uploaded to blob URL: {blob.url}")
    return blob.url

def submit_transcription(blob_url: str) -> str:
    headers = {
        "Ocp-Apim-Subscription-Key": os.getenv("AZURE_SPEECH_KEY"),
        "Content-Type": "application/json"
    }
    body = {
        "contentUrls": [blob_url],
        "locale": "en-US",
        "displayName": f"Job-{uuid.uuid4()}",
        "properties": {
            "diarizationEnabled": True,
            "wordLevelTimestampsEnabled": True,
            "timeToLiveHours": 48
        }
    }
    resp = requests.post(f"{os.getenv('AZURE_SPEECH_ENDPOINT')}/speechtotext/v3.2/transcriptions",
                         headers=headers, json=body)
    resp.raise_for_status()
    loc = resp.headers["Location"]
    print("✅ Job submitted:", loc)
    return loc

def poll_job(loc: str) -> dict:
    headers = {"Ocp-Apim-Subscription-Key": os.getenv("AZURE_SPEECH_KEY")}
    print("⏳ Polling job status…")
    while True:
        resp = requests.get(loc, headers=headers); resp.raise_for_status()
        job = resp.json()
        print("Status:", job["status"])
        if job["status"] in ("Succeeded", "Failed"):
            return job
        time.sleep(15)

def download_results(job: dict):
    headers = {"Ocp-Apim-Subscription-Key": os.getenv("AZURE_SPEECH_KEY")}
    os.makedirs("results", exist_ok=True)
    files_url = job.get("links", {}).get("files")
    if not files_url:
        print("❌ No files link found.")
        return
    resp = requests.get(files_url, headers=headers); resp.raise_for_status()
    for item in resp.json().get("values", []):
        kind = item.get("kind"); url = item.get("links", {}).get("contentUrl")
        if url:
            print(f"⬇️ Downloading {kind}...")
            r = requests.get(url, headers=headers); r.raise_for_status()
            path = os.path.join("results", f"{kind}.json")
            with open(path, "wb") as fh: fh.write(r.content)
            print(f"✅ Saved: {path}")

def save_combined_transcript(json_path="results/Transcription.json", txt_path="results/transcript.txt"):
    data = json.load(open(json_path, encoding="utf-8"))
    segments = data.get("recognizedPhrases", [])
    speaker_map, next_label = {}, 1
    combined = []
    for seg in segments:
        sp = seg.get("speaker", 0)
        if sp not in speaker_map:
            speaker_map[sp] = next_label
            next_label += 1
        label = speaker_map[sp]
        text = seg.get("nBest", [{}])[0].get("display", seg.get("display", "")).strip()
        if combined and combined[-1]['speaker'] == label:
            combined[-1]['text'] += " " + text
        else:
            offset_sec = seg.get("offsetInTicks",0)/10_000_000
            combined.append({"speaker": label, "time": offset_sec, "text": text})

    with open(txt_path, "w", encoding="utf-8") as out:
        for entry in combined:
            out.write(f"Speaker {entry['speaker']}\n{entry['text']}\n\n")
    print(f"✅ Transcript saved to: {txt_path}")

if __name__ == "__main__":
    mp3 = download_audio(os.getenv("VIDEO_URL"))
    mono = convert_to_mono(mp3)
    blob_url = upload_to_blob(mono)
    job = poll_job(submit_transcription(blob_url))
    download_results(job)
    save_combined_transcript()
