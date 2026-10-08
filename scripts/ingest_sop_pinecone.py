import os
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, cast
import pandas as pd
import pypdf  

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Suppress the Windows symlink warning noise completely
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# ==========================================
# 1. PATH RESOLUTION & SETUP
# ==========================================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import EmbeddingsModel, reveal, settings

PINECONE_API_KEY = reveal(settings.pinecone_api_key)

# Set up clean production data cache path
cache_dir = project_root / "data" / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. DYNAMIC ENVIRONMENT ROUTING
# ==========================================
EMBEDDINGS_MODEL_SETTING = settings.embeddings_model

# Isolated index and embedding width per provider
INDEX_NAME = "fde-sop-index-openai" if EMBEDDINGS_MODEL_SETTING is EmbeddingsModel.OPENAI else "fde-sop-index-local"
TARGET_DIMENSION = 1536 if EMBEDDINGS_MODEL_SETTING is EmbeddingsModel.OPENAI else 1024  # 1024 is the standard width for BGE-M3

if EMBEDDINGS_MODEL_SETTING is EmbeddingsModel.OPENAI:
    print("🤖 Mode: Utilizing Cloud OpenAI Embeddings (1536 Dim)...")
    embeddings = OpenAIEmbeddings()
else:
    # Read the explicit model identifier casing string from the .env parameters
    local_model_target = settings.local_embedding_model
    
    print(f"🤗 Mode: Local Fallback Settings Activated. Launching [{local_model_target}] (1024 Dim)...")
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=local_model_target,   # Passes parameter dynamically
        model_kwargs={'device': 'cpu'}
    )

# ==========================================
# 3. PINECONE PROVISIONING
# ==========================================
print(f"Connecting to Pinecone Index Target: [{INDEX_NAME}]...")
pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = pc.list_indexes().names()

# Each index keeps its own hash cache. One shared file would make unchanged files look
# "already ingested" after switching LOCAL/OPENAI or rebuilding an index, leaving it empty.
HASH_CACHE_FILE = cache_dir / f"ingestion_hash_cache_{INDEX_NAME}.json"
index_rebuilt = False

# Self-healing verification in case an index was created with a wrong legacy dimension
if INDEX_NAME in existing_indexes:
    desc = pc.describe_index(INDEX_NAME)
    if desc.dimension != TARGET_DIMENSION:
        print(f"⚠️ Fixing tracking: Purging mismatched {desc.dimension} dim index...")
        pc.delete_index(INDEX_NAME)
        existing_indexes = [name for name in existing_indexes if name != INDEX_NAME]
        index_rebuilt = True

if INDEX_NAME not in existing_indexes:
    print(f"Creating isolated target index: {INDEX_NAME} ({TARGET_DIMENSION} Dim)...")
    cast(Any, pc).create_index(
        name=INDEX_NAME,
        dimension=TARGET_DIMENSION, 
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    index_rebuilt = True
    # Upserting before the index is ready fails; wait (bounded) for it.
    ready_deadline = time.monotonic() + 180
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        if time.monotonic() > ready_deadline:
            raise TimeoutError(f"Pinecone index {INDEX_NAME} was not ready after 180s.")
        print("  ⏳ Waiting for index to become ready...")
        time.sleep(3)

hash_cache: dict[str, str] = {}
if HASH_CACHE_FILE.exists() and not index_rebuilt:
    try:
        with open(HASH_CACHE_FILE, "r", encoding="utf-8") as f:
            hash_cache = json.load(f)
    except Exception:
        hash_cache = {}

index_client = cast(Any, pc).Index(INDEX_NAME)
vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

def purge_file_vectors(file_name: str) -> None:
    """Delete every vector belonging to a source file.

    Serverless indexes don't reliably support delete-by-metadata-filter, so list the
    deterministic chunk ids by prefix first and fall back to the filter.
    """
    prefix = f"{file_name}-chunk-"
    try:
        stale_ids: list[str] = []
        for id_page in index_client.list(prefix=prefix):
            stale_ids.extend(str(vector_id) for vector_id in cast(list[Any], id_page))
        for i in range(0, len(stale_ids), 1000):
            index_client.delete(ids=stale_ids[i : i + 1000])
    except Exception:
        index_client.delete(filter={"source_file": {"$eq": file_name}})

# ==========================================
# 4. ROBUST POLYMORPHIC PARSER
# ==========================================
def parse_and_chunk_document(doc_path: Path) -> list[Document]:
    ext = doc_path.suffix.lower()
    raw_chunks: list[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    
    if ext == ".md":
        headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        raw_text = doc_path.read_text(encoding="utf-8")
        header_docs = md_splitter.split_text(raw_text)
        raw_chunks = text_splitter.split_documents(header_docs)
        
    elif ext == ".txt":
        raw_text = doc_path.read_text(encoding="utf-8")
        raw_docs = [Document(page_content=raw_text)]
        raw_chunks = text_splitter.split_documents(raw_docs)
        
    elif ext == ".pdf":
        pdf_docs: list[Document] = []
        try:
            with open(doc_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pdf_docs.append(Document(page_content=page_text, metadata={"page_number": page_num + 1}))
            raw_chunks = text_splitter.split_documents(pdf_docs)
        except Exception as e:
            print(f"  ❌ Error parsing PDF {doc_path.name}: {e}")
            return []
        
    elif ext in [".csv", ".xlsx"]:
        try:
            df = pd.read_csv(doc_path) if ext == ".csv" else cast(Any, pd).read_excel(doc_path)
        except Exception as e:
            print(f"  ❌ Error reading table: {e}")
            return []
            
        for idx, row in df.iterrows():
            row_dict = cast(dict[Any, Any], row.to_dict())
            row_items = [
                f"{str(col)}: {str(val)}" 
                for col, val in row_dict.items() 
                if pd.notna(val) and str(val).strip() != ""
            ]
            
            if row_items:
                row_text = " | ".join(row_items)
                doc_item = Document(page_content=row_text, metadata={"row_index": int(cast(int, idx))})
                raw_chunks.append(doc_item)

    # sanity check
    valid_chunks: list[Document] = []
    for chunk in raw_chunks:
        clean_text = chunk.page_content.strip()
        if clean_text:
            chunk.page_content = clean_text
            valid_chunks.append(chunk)
            
    return valid_chunks

# ==========================================
# 5. INCREMENTAL PIPELINE WITH BATCHING
# ==========================================
policy_dir = project_root / "data" / "policy"
target_patterns = ["*.md", "*.txt", "*.pdf", "*.csv", "*.xlsx"]
current_files: dict[str, Path] = {}
for pattern in target_patterns:
    for file_path in policy_dir.glob(pattern):
        current_files[file_path.name] = file_path

print(f"Found {len(current_files)} policy file(s) in {policy_dir}...")

updated_cache: dict[str, str] = {}
cache_modified = False

cached_filenames = set(hash_cache.keys())
current_filenames = set(current_files.keys())
deleted_files = cached_filenames - current_filenames

for deleted_file in deleted_files:
    print(f"🗑️ Detected deleted file: {deleted_file}. Purging from Pinecone...")
    try:
        purge_file_vectors(deleted_file)
    except Exception as e:
        print(f"  ❌ Failed to purge {deleted_file}: {e}")
        updated_cache[deleted_file] = hash_cache[deleted_file]  # keep it so the purge is retried next run
    cache_modified = True

for file_name, file_path in current_files.items():
    file_bytes = file_path.read_bytes()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    updated_cache[file_name] = file_hash
    
    if hash_cache.get(file_name) == file_hash:
        print(f"✨ Skipped (Unchanged): {file_name}")
        continue
        
    print(f"🔄 Processing updates: {file_name}...")
    cache_modified = True
    
    try:
        # Must succeed: a silent failure would leave stale chunks from the previous version.
        purge_file_vectors(file_name)

        chunks = parse_and_chunk_document(file_path)
        if not chunks:
            print(f"  ⚠️ No valid text chunks extracted from {file_name}.")
            continue
            
        explicit_ids: list[str] = []
        for idx, chunk in enumerate(chunks):
            chunk.metadata["source_file"] = file_name
            chunk.metadata["file_format"] = file_path.suffix.replace(".", "").upper()
            chunk.metadata["document_type"] = "Compliance Asset"
            explicit_ids.append(f"{file_name}-chunk-{idx}")
            
        batch_size = 100
        total_chunks = len(chunks)
        print(f"  📤 Upserting {total_chunks} chunk(s) in batches of {batch_size}...")
        
        for i in range(0, total_chunks, batch_size):
            batch_docs = chunks[i : i + batch_size]
            batch_ids = explicit_ids[i : i + batch_size]
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            
    except Exception as e:
        print(f"❌ Error during ingestion of {file_name}: {e}")
        updated_cache.pop(file_name, None)

# ==========================================
# 6. SYNC HASH CACHE
# ==========================================
if cache_modified:
    with open(HASH_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_cache, f, indent=4)
    print("✅ Ingestion & cache update complete.")
else:
    print("🌴 Index is already up-to-date.")