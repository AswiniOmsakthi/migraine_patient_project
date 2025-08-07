import os, re
from pathlib import Path
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
assert ENDPOINT and API_KEY and API_VERSION and DEPLOYMENT, "Missing Azure env vars"

client = AzureOpenAI(azure_endpoint=ENDPOINT, api_key=API_KEY, api_version=API_VERSION)

class FunctionEmbedding:
    def __init__(self, fn, name="azure_embed_fn"):
        self.fn = fn
        self._name = name
    def __call__(self, input: list[str]):
        return self.fn(input)
    def name(self):
        return self._name

def azure_embed(input: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(
        model=DEPLOYMENT,
        input=input,
        dimensions=1024  # truncate embeddings to 1024 dims
    )
    return [item.embedding for item in resp.data]

emb_fn = FunctionEmbedding(azure_embed, name=f"azure-{DEPLOYMENT}")

def safe_collection_name(s: str) -> str:
    sanitized = re.sub(r'[^A-Za-z0-9_-]', '-', s)
    return re.sub(r'-+', '-', sanitized).strip('-')[:63]

def chunk_text(text: str, chunk_size=500, overlap=50) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PDF_FOLDER = PROJECT_ROOT / "product_info_pdfs"
EXCEL_PATH = SCRIPT_DIR / "smpc_aimovig_filename_description.xlsx"
QA_CSV_PATH = PDF_FOLDER / "migraine interview question answer.csv"
PERSIST_DIR = PROJECT_ROOT / "chroma_data"

def get_persistent_client():
    PERSIST_DIR.mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=str(PERSIST_DIR), settings=Settings())

def process_qa_csv(cli):
    df = pd.read_csv(QA_CSV_PATH, dtype=str).fillna("")
    df.columns = df.columns.str.strip().str.lower()
    # Remove first or delete existing collection if dimension mismatch
    try:
        cli.delete_collection("migraine_QA")
    except Exception:
        pass
    coll = cli.get_or_create_collection("migraine_QA", embedding_function=emb_fn)
    total = 0
    for idx, row in df.iterrows():
        q, a = row.get("question", "").strip(), row.get("answer", "").strip()
        if not (q or a):
            continue
        text = f"Q: {q}\n\nA: {a}"
        chunks = chunk_text(text)
        total += len(chunks)
        coll.add(
            documents=chunks,
            ids=[f"qa_{idx}_chunk{cid}" for cid in range(len(chunks))],
            metadatas=[{"row_index": idx} for _ in chunks]
        )
    print(f"✅ Processed QA CSV: {len(df)} rows → {total} chunks")

def ingest_pdfs(cli):
    df = pd.read_excel(EXCEL_PATH, dtype=str).fillna("")
    pdf_summary = {}
    for _, row in df.iterrows():
        fname = row["filename"]
        path = PDF_FOLDER / fname
        if not (path.exists() and path.suffix.lower() == ".pdf"):
            print(f"⚠️ Skipped: {fname}")
            continue
        try:
            with pdfplumber.open(path) as pdf:
                text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as e:
            print(f"⚠️ Failed parse: {fname} → {e}")
            continue
        chunks = chunk_text(text)
        pdf_summary[fname] = len(chunks)
        coll_name = safe_collection_name(path.stem)
        try:
            cli.delete_collection(coll_name)
        except Exception:
            pass
        coll = cli.get_or_create_collection(coll_name, embedding_function=emb_fn)
        coll.add(
            documents=chunks,
            ids=[f"{fname}_chunk{cid}" for cid in range(len(chunks))],
            metadatas=[{"filename": fname, "chunk_index": cid, "description": row.get("description","")} for cid in range(len(chunks))]
        )
    print("\n=== PDF Chunk Summary ===")
    for fname, count in pdf_summary.items():
        print(f"{fname} → {count} chunks")
    print("✅ PDF ingestion complete")

def build_in_memory(pers_cli):
    mem = chromadb.Client()
    for info in pers_cli.list_collections():
        src = pers_cli.get_collection(info.name, embedding_function=emb_fn)
        dst = mem.get_or_create_collection(info.name, embedding_function=emb_fn)
        data = src.get(include=["documents","metadatas","embeddings"])
        dst.add(documents=data["documents"], ids=data["ids"], metadatas=data["metadatas"], embeddings=data["embeddings"])
    return mem

def main():
    print("Excel exists?", EXCEL_PATH.exists(), "| PDFs folder exists?", PDF_FOLDER.exists())
    pers = get_persistent_client()

    process_qa_csv(pers)
    ingest_pdfs(pers)

    print("\nPersistent collections:", [c.name for c in pers.list_collections()])
    qa_coll = pers.get_collection("migraine_QA", embedding_function=emb_fn)
    print("QA chunks count:", qa_coll.count())

    mem = build_in_memory(pers)
    qa_mem = mem.get_collection("migraine_QA", embedding_function=emb_fn)
    res = qa_mem.query(query_texts=["What is migraine?"], n_results=3)
    print("\n--- QUERY RESULTS ---")
    for idx, doc in enumerate(res["documents"]):
        print(f"{idx}. {doc[:200]!r} … (id={res['ids'][idx]})")

if __name__ == "__main__":
    main()