# create_db.py

import os
from pathlib import Path
import unicodedata, re
import pandas as pd
import pdfplumber
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
assert ENDPOINT and API_KEY and API_VERSION and DEPLOYMENT, \
    "Please set Azure endpoint, key, version & deployment in .env"

client = AzureOpenAI(azure_endpoint=ENDPOINT, api_key=API_KEY, api_version=API_VERSION)

def normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s.strip().lower())

def safe_collection_name(s: str) -> str:
    name = re.sub(r'[^A-Za-z0-9_-]', '-', s)
    return re.sub(r'-+', '-', name).strip('-')[:63]

def chunk_text(text: str, chunk_size=500, overlap=50) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks

class FunctionEmbedding:
    def __init__(self, fn, name="azure-embed-fn"):
        self.fn = fn; self._name = name
    def __call__(self, input): return self.fn(input)
    def name(self): return self._name

def azure_embed(input: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=DEPLOYMENT, input=input)
    return [item.embedding for item in resp.data]

emb_fn = FunctionEmbedding(azure_embed, name=f"azure-{DEPLOYMENT}")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PDF_FOLDER = PROJECT_ROOT / "product_info_pdfs"
EXCEL_PATH = SCRIPT_DIR / "smpc_aimovig_filename_description.xlsx"
QA_CSV_PATH = PDF_FOLDER / "migraine interview question answer.csv"
PERSIST_DIR = PROJECT_ROOT / "chroma_data"

def get_chroma_client():
    PERSIST_DIR.mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=str(PERSIST_DIR), settings=Settings())



def process_qa_csv(client_chroma):
    if not QA_CSV_PATH.exists():
        print("⚠️ QA CSV missing:", QA_CSV_PATH)
        return
    df = pd.read_csv(QA_CSV_PATH, dtype=str).fillna("")
    coll = client_chroma.get_or_create_collection("migraine_QA", embedding_function=emb_fn)
    added = 0

    for idx, row in df.iterrows():
        parts, meta = [], {"row_index": idx}
        if row.get("question"):
            parts.append("Q: " + row["question"])
            meta["part"] = meta.get("part", "question")
        if row.get("answer"):
            parts.append("A: " + row["answer"])
            meta["part"] = "question_answer" if meta.get("part") == "question" else meta.get("part", "answer")
        text = "\n\n".join(parts) if parts else "[no QA content]"
        for cid, chunk in enumerate(chunk_text(text)):
            coll.add(
                documents=[chunk],
                ids=[f"qa_row{idx}_chunk{cid}"],
                metadatas=[{**meta}]
            )
            added += 1

    # If no rows added at all, insert a fallback chunk
    if added == 0:
        coll.add(
            documents=["[fallback] no question or answer found"],
            ids=["qa_fallback"],
            metadatas=[{"row_index": -1, "part": "fallback"}]
        )
        added = 1
        print("ℹ️ Added fallback QA chunk to ensure folder creation.")

    print(f"✅ Processed QA CSV: {len(df)} rows, {added} chunks added")

def ingest_pdfs_and_excel(client_chroma):
    df = pd.read_excel(EXCEL_PATH, dtype=str).fillna("")
    for _, row in df.iterrows():
        fname = row["filename"]
        path = PDF_FOLDER / fname
        if path.suffix.lower() != ".pdf":
            # Skip anything that's not PDF
            continue
        if not path.exists():
            print(f"⚠️ Missing PDF: {fname}")
            continue
        try:
            with pdfplumber.open(path) as pdf:
                text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            print(f"⚠️ Failed to parse {fname}: {e}")
            continue

        chunks = chunk_text(text)
        coll = client_chroma.get_or_create_collection(
            name=safe_collection_name(path.stem),
            embedding_function=emb_fn
        )
        for cid, chunk in enumerate(chunks):
            coll.add(
                documents=[chunk],
                ids=[f"{path.name}_chunk{cid}"],
                metadatas=[{
                    "filename": path.name,
                    "chunk_index": cid,
                    "description": row.get("description", ""),
                    "folder_path": row.get("folder_path", "")
                }]
            )
        print(f"✅ Ingested {len(chunks)} chunks from {fname}")
 
def main():
    print("Excel exists?", EXCEL_PATH.exists(), "| PDF folder exists?", PDF_FOLDER.exists())
    client_chroma = get_chroma_client()
    process_qa_csv(client_chroma)
    ingest_pdfs_and_excel(client_chroma)
    print("Collections:", [c.name for c in client_chroma.list_collections()])
    print("Chunks in migraine_QA:", client_chroma.get_collection("migraine_QA").count())

if __name__ == "__main__":
    main()