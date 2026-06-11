import os
import re
import uuid
import json
import asyncio
import tempfile
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from pinecone import Pinecone
from neo4j import GraphDatabase
import fitz
import pytesseract
from PIL import Image

from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.errors import DuplicateKeyError

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# =========================
# APP
# =========================

app = FastAPI(title="GraphRAG QA API", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CONFIG
# =========================

MIN_TEXT_PER_PAGE    = int(os.getenv("MIN_TEXT_PER_PAGE", 40))
MIN_PARAGRAPH_CHARS  = int(os.getenv("MIN_PARAGRAPH_CHARS", 60))
CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE", 1200))
CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP", 180))

MONGO_URI           = os.getenv("MONGO_URI",      "mongodb://localhost:27017")
MONGO_DB_NAME       = os.getenv("MONGO_DB_NAME",  "data_edtech")
CHUNKS_COLLECTION   = "chunks"
BOOKS_COLLECTION    = "books"
CHAT_COLLECTION     = "chat_history"   # NEW: chat history
USERS_COLLECTION    = "users"          # NEW: user profiles

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL        = os.getenv("OPENAI_MODEL",       "gpt-4.1-mini")
OPENAI_EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = 1536

PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "pcs....")
PINECONE_INDEX_NAME = "graph-rag"
PINECONE_NAMESPACE  = "all_documents"

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "secretgraph")

BATCH_SIZE           = int(os.getenv("BATCH_SIZE",           50))
VECTOR_TOP_K         = int(os.getenv("VECTOR_TOP_K",         30))
KEYWORD_TOP_K        = int(os.getenv("KEYWORD_TOP_K",        30))
MIN_FINAL_CONFIDENCE = float(os.getenv("MIN_FINAL_CONFIDENCE", 0.45))

# =========================
# DB INIT
# =========================

mongo_client = MongoClient(MONGO_URI)
mongo_db     = mongo_client[MONGO_DB_NAME]
chunks_col   = mongo_db[CHUNKS_COLLECTION]
books_col    = mongo_db[BOOKS_COLLECTION]
chat_col     = mongo_db[CHAT_COLLECTION]
users_col    = mongo_db[USERS_COLLECTION]

# Chunks indexes
chunks_col.create_index([("chunk_id", ASCENDING)], unique=True)
chunks_col.create_index([("book_id",  ASCENDING)])
chunks_col.create_index([("page_no",  ASCENDING)])
chunks_col.create_index([("chapter",  ASCENDING)])
chunks_col.create_index([("heading_or_subtopic", ASCENDING)])

try:
    chunks_col.create_index(
        [
            ("text",               "text"),
            ("chapter",            "text"),
            ("heading_or_subtopic","text"),
        ],
        name="chunks_fulltext",
        default_language="english",
    )
except Exception:
    pass

books_col.create_index([("book_id", ASCENDING)], unique=True)

# Chat indexes
chat_col.create_index([("thread_id", ASCENDING)])
chat_col.create_index([("user_id",   ASCENDING)])
chat_col.create_index([("created_at", ASCENDING)])

# Users indexes
users_col.create_index([("user_id", ASCENDING)], unique=True)

# =========================
# CLIENTS
# =========================

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var is required")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY env var is required")

pc             = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

neo4j_driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

# =========================
# UTILS
# =========================

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()

def clean_paragraph(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def remove_noise_lines(text: str) -> str:
    cleaned = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            cleaned.append("")
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if re.match(r"^[A-Z\s]+ \d+$", line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

def split_into_blocks(text: str) -> List[str]:
    text = normalize_text(text)
    text = remove_noise_lines(text)
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]

# =========================
# OCR  (pytesseract fallback)
# =========================

def ocr_page(page: fitz.Page) -> str:
    """Full OCR fallback using pytesseract with 300 DPI rendering."""
    try:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Use LSTM engine for better accuracy
        custom_config = r"--oem 3 --psm 6"
        text = pytesseract.image_to_string(img, config=custom_config)
        return normalize_text(text)
    except Exception as e:
        print(f"[OCR] pytesseract failed: {e}")
        return ""

def extract_page_text(page: fitz.Page) -> Dict[str, Any]:
    try:
        normal_text = normalize_text(page.get_text("text"))
    except Exception:
        normal_text = ""

    if len(normal_text) >= MIN_TEXT_PER_PAGE:
        return {"text": normal_text, "extraction_method": "pymupdf"}

    # Fallback: try layout-preserving extraction first
    try:
        layout_text = normalize_text(page.get_text("blocks"))
        if len(layout_text) >= MIN_TEXT_PER_PAGE:
            return {"text": layout_text, "extraction_method": "pymupdf_blocks"}
    except Exception:
        pass

    # Final fallback: OCR
    ocr_text = ocr_page(page)
    method = "ocr_tesseract" if ocr_text else "failed"
    return {"text": ocr_text, "extraction_method": method}

# =========================
# HEADING DETECTION
# =========================

def looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    words = line.split()
    if len(words) > 14:
        return False
    if re.fullmatch(r"\d+", line):
        return False
    if re.match(r"^(chapter|unit|lesson|section|topic|part)\s+[\dA-Za-z]+", line, re.I):
        return True
    if re.match(r"^\d+(\.\d+)*[\).]?\s+[A-Za-z]", line):
        return True
    if line.isupper() and 2 <= len(words) <= 10:
        return True
    title_words = sum(1 for w in words if w[:1].isupper())
    if 2 <= len(words) <= 10:
        if title_words >= len(words) - 1:
            if not re.search(r"[.!?]$", line):
                return True
    return False

def clean_topic_name(topic: str) -> str:
    topic = clean_paragraph(topic)
    topic = re.sub(r"^(chapter|unit|lesson|section)\s*\d*[:.-]?\s*", "", topic, flags=re.I)
    topic = topic.strip(" :-–—.")
    return topic[:90] if topic else "General Content"

# =========================
# AI TOPIC GENERATOR
# =========================

def fallback_topic_from_text(text: str) -> str:
    words = clean_paragraph(text).split()
    return clean_topic_name(" ".join(words[:8])) if words else "General Content"

def ai_generate_topic_title(text: str, previous_topic: Optional[str] = None) -> Dict[str, Any]:
    try:
        sample = clean_paragraph(text[:1800])
        prompt = f"""You are an intelligent document understanding engine.

Task: Extract or generate the BEST section/topic title.

Rules:
- If heading exists → return heading (max 8 words).
- If no heading → generate a short meaningful topic.
- If paragraph continues previous topic, keep same topic.

Previous Topic: {previous_topic or "None"}

Return ONLY valid JSON:
{{"chapter":"string","heading_or_subtopic":"string","is_new_section":true/false}}

Content:
{sample}"""

        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You generate clean document section names."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return {
            "chapter":             clean_topic_name(data.get("chapter") or "General Document"),
            "heading_or_subtopic": clean_topic_name(data.get("heading_or_subtopic") or fallback_topic_from_text(text)),
            "is_new_section":      bool(data.get("is_new_section", False)),
            "detection_method":    "ai_generated_topic",
        }
    except Exception:
        return {
            "chapter":             "General Document",
            "heading_or_subtopic": fallback_topic_from_text(text),
            "is_new_section":      False,
            "detection_method":    "fallback_summary_title",
        }

def detect_document_section(block: str, current_chapter: Optional[str], current_heading: Optional[str]) -> Dict[str, Any]:
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    if lines and looks_like_heading(lines[0]):
        heading = clean_topic_name(lines[0])
        chapter = current_chapter or "General Document"
        if re.match(r"^(chapter|unit|lesson|section|part)", lines[0], re.I):
            chapter = heading
        return {
            "chapter":             chapter,
            "heading_or_subtopic": heading,
            "is_new_section":      True,
            "detection_method":    "real_heading_detected",
        }
    generated = ai_generate_topic_title(block, current_heading)
    if current_heading and not generated.get("is_new_section"):
        return {
            "chapter":             current_chapter or generated["chapter"],
            "heading_or_subtopic": current_heading,
            "is_new_section":      False,
            "detection_method":    generated["detection_method"],
        }
    return generated

# =========================
# SENTENCE-AWARE CHUNKING
# =========================

def sentence_aware_chunking(text: str, max_chars: int = CHUNK_SIZE) -> List[str]:
    text = clean_paragraph(text)
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]

# =========================
# MONGO HELPERS
# =========================

def safe_insert_chunk(chunk: dict) -> bool:
    try:
        chunks_col.insert_one(chunk)
        return True
    except DuplicateKeyError:
        return False
    except Exception:
        return False

def save_book_meta(book_doc: dict) -> None:
    try:
        books_col.update_one(
            {"book_id": book_doc["book_id"]},
            {"$set": book_doc},
            upsert=True,
        )
    except Exception:
        pass

# =========================
# CHUNK BUILDER
# =========================

def build_chunks(
    book_id: str,
    book_name: str,
    filename: str,
    pages: List[Dict[str, Any]],
    save_to_mongo: bool = True,
) -> List[dict]:
    chunks = []
    current_chapter = None
    current_heading = None
    global_paragraph_no = 0

    for page_data in pages:
        page_no           = page_data["page_no"]
        page_text         = page_data["text"]
        extraction_method = page_data["extraction_method"]
        blocks            = split_into_blocks(page_text)
        page_paragraph_no = 0

        for block in blocks:
            lines = [l.strip() for l in block.split("\n") if l.strip()]
            if not lines:
                continue
            paragraph_text = clean_paragraph(" ".join(lines))
            if len(paragraph_text) < MIN_PARAGRAPH_CHARS:
                continue

            detected = detect_document_section(block, current_chapter, current_heading)
            current_chapter = detected["chapter"] or current_chapter or "General Document"
            current_heading = detected["heading_or_subtopic"] or current_heading or "General Content"

            page_paragraph_no   += 1
            global_paragraph_no += 1

            split_texts = sentence_aware_chunking(paragraph_text)
            for idx, chunk_text in enumerate(split_texts, start=1):
                chunk = {
                    "chunk_id":                     str(uuid.uuid4()),
                    "book_id":                      book_id,
                    "book_name":                    book_name,
                    "source_file":                  filename,
                    "page_no":                      page_no,
                    "page_paragraph_no":            page_paragraph_no,
                    "global_paragraph_no":          global_paragraph_no,
                    "chapter":                      current_chapter,
                    "heading_or_subtopic":          current_heading,
                    "topic_detection_method":       detected["detection_method"],
                    "is_new_section":               detected.get("is_new_section", False),
                    "chunk_index_inside_paragraph": idx,
                    "total_chunks_from_paragraph":  len(split_texts),
                    "extraction_method":            extraction_method,
                    "char_count":                   len(chunk_text),
                    "word_count":                   len(chunk_text.split()),
                    "text_preview":                 chunk_text[:250],
                    "text":                         chunk_text,
                    "status":                       "chunked",
                    "pinecone_indexed":              False,
                    "neo4j_indexed":                False,
                    "created_at":                   now_utc(),
                    "updated_at":                   now_utc(),
                }
                if save_to_mongo:
                    safe_insert_chunk(chunk)
                chunks.append(chunk)

    return chunks

# =========================
# EMBEDDINGS
# =========================

def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    response = openai_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]

def build_embedding_text(chunk: dict) -> str:
    """
    Rich embedding text — combines structural metadata + content.
    This improves semantic retrieval significantly for QA systems.
    """
    return (
        f"Book: {chunk.get('book_name', '')}\n"
        f"Chapter: {chunk.get('chapter', 'General Document')}\n"
        f"Topic: {chunk.get('heading_or_subtopic', 'General Content')}\n"
        f"Page: {chunk.get('page_no', '')}\n\n"
        f"{chunk.get('text', '')}"
    ).strip()

# =========================
# PINECONE UPSERT
# =========================

def save_chunks_to_pinecone_batch(chunks: List[dict]) -> None:
    texts   = [build_embedding_text(c) for c in chunks]
    vectors = create_embeddings_batch(texts)

    pinecone_vectors = []
    for chunk, vector in zip(chunks, vectors):
        pinecone_vectors.append({
            "id":     chunk["chunk_id"],
            "values": vector,
            "metadata": {
                "chunk_id":    chunk["chunk_id"],
                "book_id":     chunk["book_id"],
                "book_name":   chunk.get("book_name", ""),
                "source_file": chunk.get("source_file", ""),
                "page_no":     chunk.get("page_no", 0),
                "chapter":     chunk.get("chapter", "General Document"),
                "heading":     chunk.get("heading_or_subtopic", "General Content"),
                "text_preview":chunk.get("text_preview", ""),
            },
        })

    pinecone_index.upsert(
        vectors=pinecone_vectors,
        namespace=PINECONE_NAMESPACE,
    )

# =========================
# NEO4J UPSERT
# =========================

def save_chunks_to_neo4j_batch(chunks: List[dict]) -> None:
    query = """
    UNWIND $chunks AS chunk

    MERGE (b:Book {book_id: chunk.book_id})
    SET b.name = chunk.book_name, b.source_file = chunk.source_file

    MERGE (c:Chapter {book_id: chunk.book_id, name: chunk.chapter})
    MERGE (h:Heading {book_id: chunk.book_id, chapter: chunk.chapter, name: chunk.heading})
    MERGE (p:Page    {book_id: chunk.book_id, page_no: chunk.page_no})

    MERGE (ch:Chunk {chunk_id: chunk.chunk_id})
    SET ch.text_preview        = chunk.text_preview,
        ch.page_no             = chunk.page_no,
        ch.word_count          = chunk.word_count,
        ch.global_paragraph_no = chunk.global_paragraph_no

    MERGE (b)-[:HAS_CHAPTER]->(c)
    MERGE (c)-[:HAS_HEADING]->(h)
    MERGE (h)-[:HAS_CHUNK]  ->(ch)
    MERGE (b)-[:HAS_PAGE]   ->(p)
    MERGE (p)-[:HAS_CHUNK]  ->(ch)
    """
    rows = [
        {
            "chunk_id":            c["chunk_id"],
            "book_id":             c["book_id"],
            "book_name":           c.get("book_name", ""),
            "source_file":         c.get("source_file", ""),
            "chapter":             c.get("chapter", "General Document"),
            "heading":             c.get("heading_or_subtopic", "General Content"),
            "page_no":             c.get("page_no", 0),
            "text_preview":        c.get("text_preview", ""),
            "word_count":          c.get("word_count", 0),
            "global_paragraph_no": c.get("global_paragraph_no", 0),
        }
        for c in chunks
    ]
    with neo4j_driver.session() as session:
        session.run(query, chunks=rows)

def update_mongo_index_status_batch(chunks: List[dict]) -> None:
    ops = [
        UpdateOne(
            {"chunk_id": c["chunk_id"]},
            {"$set": {"pinecone_indexed": True, "neo4j_indexed": True, "indexed_at": now_utc()}},
        )
        for c in chunks
    ]
    if ops:
        chunks_col.bulk_write(ops, ordered=False)

# =========================
# OPENAI JSON HELPER
# =========================

def call_openai_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return json.loads(resp.choices[0].message.content)

# =========================
# CHAT HISTORY HELPERS
# =========================

def get_chat_history(thread_id: str, limit: int = 20) -> List[dict]:
    """Get recent messages for a thread."""
    msgs = list(
        chat_col.find({"thread_id": thread_id}, {"_id": 0})
        .sort("created_at", ASCENDING)
        .limit(limit)
    )
    return msgs

def save_chat_message(thread_id: str, user_id: str, role: str, content: str, metadata: dict = None) -> None:
    """Save a chat message."""
    chat_col.insert_one({
        "message_id": str(uuid.uuid4()),
        "thread_id":  thread_id,
        "user_id":    user_id,
        "role":       role,        # "user" | "assistant"
        "content":    content,
        "metadata":   metadata or {},
        "created_at": now_utc(),
    })

def get_or_create_user(user_id: str) -> dict:
    """Get user profile or create empty one."""
    user = users_col.find_one({"user_id": user_id}, {"_id": 0})
    if not user:
        user = {
            "user_id":    user_id,
            "name":       None,
            "facts":      [],        # important facts about the user
            "created_at": now_utc(),
            "updated_at": now_utc(),
        }
        try:
            users_col.insert_one(user.copy())
        except Exception:
            pass
    return user

def update_user_facts(user_id: str, new_facts: List[str], name: Optional[str] = None) -> None:
    """Update user profile with new facts."""
    update = {"$set": {"updated_at": now_utc()}}
    if new_facts:
        update["$addToSet"] = {"facts": {"$each": new_facts}}
    if name:
        update["$set"]["name"] = name
    users_col.update_one({"user_id": user_id}, update, upsert=True)

def extract_user_facts(message: str, user_id: str) -> None:
    """
    Extract and save user facts if they share personal info.
    Runs async-style: fire and (softly) forget.
    """
    try:
        resp = call_openai_json(
            system_prompt="""Extract personal facts the user just shared about themselves.
Return JSON: {"name": null_or_string, "facts": ["list of facts as short strings"]}
Only include EXPLICIT personal info (name, job, location, hobby, etc).
If nothing personal → return {"name": null, "facts": []}""",
            user_prompt=f"Message: {message}"
        )
        name  = resp.get("name")
        facts = resp.get("facts", [])
        if name or facts:
            update_user_facts(user_id, facts, name)
    except Exception:
        pass

def build_history_messages(thread_id: str, user_profile: dict) -> List[dict]:
    """Build OpenAI messages array from thread history."""
    history = get_chat_history(thread_id, limit=20)
    messages = []

    # Add user context if available
    user_name  = user_profile.get("name")
    user_facts = user_profile.get("facts", [])

    if user_name or user_facts:
        context_parts = []
        if user_name:
            context_parts.append(f"User's name: {user_name}")
        if user_facts:
            context_parts.append("Known facts about user: " + "; ".join(user_facts[:10]))
        messages.append({
            "role":    "system",
            "content": "User context — " + " | ".join(context_parts),
        })

    for msg in history:
        messages.append({
            "role":    msg["role"],
            "content": msg["content"],
        })
    return messages

# =========================
# QUERY CLASSIFIER
# =========================

def classify_user_query(user_query: str, chat_history: List[dict] = None) -> Dict[str, Any]:
    history_hint = ""
    if chat_history:
        last_few = chat_history[-4:]
        history_hint = "\nRecent conversation:\n" + "\n".join(
            f"{m['role']}: {m['content'][:200]}" for m in last_few
        )

    system = """You are a query router for an educational AI app.

Classify the user query strictly:
- greetings, thanks, casual chat, personal talk, general knowledge = "chitchat"
- anything about uploaded books, PDFs, topics, lessons, chapters, MCQs, summaries of content, questions needing document context = "rag_search"

Return ONLY JSON:
{
  "route": "chitchat|rag_search",
  "reason": "",
  "should_search_db": true/false
}"""

    return call_openai_json(system, f"User query: {user_query}{history_hint}")

def generate_chitchat_reply(user_query: str, history_messages: List[dict], user_profile: dict) -> str:
    """Answer chitchat using conversation history + user profile."""
    user_name = user_profile.get("name")
    
    system_content = (
        "You are a helpful, friendly educational assistant. "
        "Answer naturally and concisely. "
        "Remember personal details the user shares. "
        "Do NOT mention database, documents, or uploaded files in casual chat."
    )
    if user_name:
        system_content += f" Address the user as {user_name}."

    messages = [{"role": "system", "content": system_content}]
    messages.extend(history_messages[-10:])
    messages.append({"role": "user", "content": user_query})

    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.5,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()

# =========================
# SEARCH HELPERS
# =========================

def openai_query_planner(user_query: str, book_name: str = "") -> Dict[str, Any]:
    system = """You are a production-level educational search query planner for a QA system.
1. Understand the user's learning intent deeply.
2. Fix spelling mistakes naturally.
3. Expand query into diverse semantic queries covering different angles.
4. Generate keyword queries for full-text search.
5. Detect intent type.
Return JSON only."""

    user = f"""User query: {user_query}
Book context: {book_name}

Return JSON:
{{
  "corrected_query": "",
  "intent": "definition|concept|formula|example|exercise|explanation|comparison|unknown",
  "main_topic": "",
  "semantic_queries": ["query1", "query2", "query3", "query4"],
  "keyword_queries": ["kw1", "kw2", "kw3"],
  "must_have_terms": [],
  "avoid_terms": []
}}"""
    return call_openai_json(system, user)

def vector_search(
    queries: List[str],
    book_id: Optional[str] = None,
    top_k: int = VECTOR_TOP_K,
) -> List[Dict[str, Any]]:
    all_matches: Dict[str, dict] = {}
    pinecone_filter: Dict[str, Any] = {}
    if book_id:
        pinecone_filter["book_id"] = {"$eq": book_id}

    for q in queries:
        if not q or not q.strip():
            continue
        query_vector = create_embeddings_batch([q])[0]
        result = pinecone_index.query(
            namespace=PINECONE_NAMESPACE,
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter if pinecone_filter else None,
        )
        for match in result.get("matches", []):
            cid   = match["id"]
            score = float(match.get("score", 0))
            if cid not in all_matches or score > all_matches[cid]["vector_score"]:
                all_matches[cid] = {
                    "chunk_id":     cid,
                    "vector_score": score,
                    "metadata":     match.get("metadata", {}),
                }

    return list(all_matches.values())

def keyword_search(
    book_id: Optional[str],
    keyword_queries: List[str],
    top_k: int = KEYWORD_TOP_K,
) -> List[Dict[str, Any]]:
    results: Dict[str, dict] = {}
    for q in keyword_queries:
        if not q or not q.strip():
            continue
        mongo_filter: Dict[str, Any] = {"$text": {"$search": q}}
        if book_id:
            mongo_filter["book_id"] = book_id
        cursor = (
            chunks_col.find(
                mongo_filter,
                {"_id": 0, "chunk_id": 1, "keyword_score": {"$meta": "textScore"}},
            )
            .sort([("keyword_score", {"$meta": "textScore"})])
            .limit(top_k)
        )
        for doc in cursor:
            cid   = doc["chunk_id"]
            score = float(doc.get("keyword_score", 0))
            if cid not in results or score > results[cid]["keyword_score"]:
                results[cid] = {"chunk_id": cid, "keyword_score": score}
    return list(results.values())

def normalize_scores(items: List[Dict], score_key: str) -> Dict[str, float]:
    if not items:
        return {}
    scores       = [x.get(score_key, 0) for x in items]
    min_s, max_s = min(scores), max(scores)
    out = {}
    for item in items:
        s    = item.get(score_key, 0)
        norm = 1.0 if max_s == min_s else (s - min_s) / (max_s - min_s)
        out[item["chunk_id"]] = round(norm, 4)
    return out

def merge_hybrid_results(
    vector_results: List[Dict], keyword_results: List[Dict]
) -> List[Dict]:
    v_norm  = normalize_scores(vector_results,  "vector_score")
    k_norm  = normalize_scores(keyword_results, "keyword_score")
    all_ids = set(v_norm) | set(k_norm)
    merged = [
        {
            "chunk_id":      cid,
            "vector_score":  v_norm.get(cid, 0),
            "keyword_score": k_norm.get(cid, 0),
            "hybrid_score":  round(0.70 * v_norm.get(cid, 0) + 0.30 * k_norm.get(cid, 0), 4),
        }
        for cid in all_ids
    ]
    merged.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return merged

def get_related_chunks_from_neo4j(chunk_ids: List[str], limit: int = 20) -> List[str]:
    if not chunk_ids:
        return []
    query = """
    MATCH (ch:Chunk) WHERE ch.chunk_id IN $chunk_ids
    OPTIONAL MATCH (h:Heading)-[:HAS_CHUNK]->(ch)
    OPTIONAL MATCH (h)-[:HAS_CHUNK]->(related:Chunk)
    WITH DISTINCT related.chunk_id AS chunk_id
    WHERE chunk_id IS NOT NULL
    RETURN chunk_id LIMIT $limit
    """
    related: set = set()
    try:
        with neo4j_driver.session() as session:
            result = session.run(query, chunk_ids=chunk_ids, limit=limit)
            for record in result:
                related.add(record["chunk_id"])
    except Exception as e:
        print(f"[neo4j] get_related_chunks error: {e}")
    return list(related)

def openai_rerank(query_plan: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    compact = [
        {
            "rank_id":      i,
            "chunk_id":     c["chunk_id"],
            "chapter":      c.get("chapter"),
            "heading":      c.get("heading_or_subtopic"),
            "page_no":      c.get("page_no"),
            "text":         c.get("text", "")[:1200],
            "hybrid_score": c.get("hybrid_score", 0),
            "source":       c.get("source"),
        }
        for i, c in enumerate(chunks)
    ]

    system = """You are a strict educational RAG reranker for a QA system.
Select chunks that are truly relevant to the user query.
Be generous with confidence when clear matches exist.
Assign confidence 0–1.
Return JSON only."""

    user = f"""Query plan:
{json.dumps(query_plan, indent=2)}

Candidate chunks:
{json.dumps(compact, indent=2)}

Return JSON:
{{
  "confidence": 0.0,
  "reason": "",
  "selected_chunk_ids": [],
  "rejected_chunk_ids": []
}}"""

    return call_openai_json(system, user)

def generate_long_answer(query: str, chunks: List[dict], history_messages: List[dict], user_profile: dict) -> str:
    """
    Generate a comprehensive long-form answer using retrieved chunks + chat history.
    """
    user_name = user_profile.get("name")

    context_parts = []
    for c in chunks[:12]:
        context_parts.append(
            f"[Book: {c.get('book_name','')} | Chapter: {c.get('chapter','')} | "
            f"Topic: {c.get('heading_or_subtopic','')} | Page: {c.get('page_no','')}]\n"
            f"{c.get('text','')}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    system_content = f"""You are an expert educational AI assistant for a QA system.
You have access to relevant document excerpts. Use them to give a comprehensive, well-structured answer.

Rules:
- Give a DETAILED, LONG answer (minimum 300 words unless the topic is very simple).
- Structure with headings, bullet points, and examples where helpful.
- Cite which book/chapter/page the information comes from.
- If context has partial info, supplement with your knowledge but mark it clearly.
- Be educational, clear, and thorough.
{"- Address the user as " + user_name + "." if user_name else ""}

Document Context:
{context_text[:7000]}"""

    messages = [{"role": "system", "content": system_content}]
    # Include last 6 history messages for continuity
    messages.extend(history_messages[-6:])
    messages.append({"role": "user", "content": query})

    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.3,
        max_tokens=2000,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()

# =========================
# PYDANTIC MODELS
# =========================

class IndexBookRequest(BaseModel):
    book_id: str

class ChatRequest(BaseModel):
    query:     str
    user_id:   str
    thread_id: str

class MCQRequest(BaseModel):
    topic:         str
    book_id:       Optional[str] = None
    num_questions: int = 10
    difficulty:    str = "medium"

class LessonSummaryRequest(BaseModel):
    topic:   str
    book_id: Optional[str] = None

# =========================
# TOPIC CHUNK FETCHER
# =========================

def fetch_topic_chunks(topic: str, book_id: Optional[str], top_k: int = 20) -> List[dict]:
    query_plan = openai_query_planner(user_query=topic)
    search_queries = list(set(
        [query_plan.get("corrected_query", topic)]
        + query_plan.get("semantic_queries", [])
    ))
    keyword_queries = list(set(
        [query_plan.get("main_topic", topic)]
        + query_plan.get("keyword_queries", [])
        + query_plan.get("must_have_terms", [])
    ))

    vector_results  = vector_search(queries=search_queries, book_id=book_id, top_k=top_k)
    keyword_results = keyword_search(book_id=book_id, keyword_queries=keyword_queries, top_k=top_k)
    hybrid_results  = merge_hybrid_results(vector_results, keyword_results)

    if not hybrid_results:
        return []

    top_ids     = [x["chunk_id"] for x in hybrid_results[:20]]
    related_ids = get_related_chunks_from_neo4j(top_ids[:10])
    final_ids   = list(set(top_ids + related_ids))

    mongo_filter: Dict[str, Any] = {"chunk_id": {"$in": final_ids}}
    if book_id:
        mongo_filter["book_id"] = book_id

    chunks = list(
        chunks_col.find(mongo_filter, {"_id": 0})
        .sort([("page_no", ASCENDING), ("global_paragraph_no", ASCENDING)])
        .limit(top_k)
    )

    score_map = {x["chunk_id"]: x for x in hybrid_results}
    return sorted(chunks, key=lambda c: score_map.get(c["chunk_id"], {}).get("hybrid_score", 0), reverse=True)

# =========================
# API — ADMIN: CHUNK BOOK
# =========================

@app.post("/v1/admin/books/chunk")
async def upload_and_chunk_book(
    book_name: str        = Form(...),
    file:      UploadFile = File(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    book_id   = str(uuid.uuid4())
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        pdf = fitz.open(temp_path)
        save_book_meta({
            "book_id":     book_id,
            "book_name":   book_name,
            "source_file": file.filename,
            "status":      "processing",
            "created_at":  now_utc(),
            "updated_at":  now_utc(),
        })

        pages = []
        for i in range(len(pdf)):
            page      = pdf.load_page(i)
            extracted = extract_page_text(page)
            pages.append({
                "page_no":           i + 1,
                "text":              extracted["text"],
                "extraction_method": extracted["extraction_method"],
                "char_count":        len(extracted["text"]),
            })

        chunks = build_chunks(
            book_id=book_id, book_name=book_name,
            filename=file.filename, pages=pages, save_to_mongo=True,
        )

        save_book_meta({
            "book_id":      book_id,
            "book_name":    book_name,
            "source_file":  file.filename,
            "status":       "chunked",
            "total_pages":  len(pdf),
            "total_chunks": len(chunks),
            "updated_at":   now_utc(),
        })

        return JSONResponse({
            "success":      True,
            "book_id":      book_id,
            "book_name":    book_name,
            "total_pages":  len(pdf),
            "total_chunks": len(chunks),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# =========================
# API — ADMIN: CHUNK BOOK (SSE STREAM)
# =========================

@app.post("/v1/admin/books/chunk/stream")
async def upload_and_chunk_book_stream(
    book_name: str        = Form(...),
    file:      UploadFile = File(...),
):
    """
    Stream chunking progress via SSE.
    Events: start | pdf_info | page_extracted | heading_detected | chunk | page_done | done | error
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    book_id   = str(uuid.uuid4())
    temp_path = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        temp_path = tmp.name

    async def stream_generator():
        total_chunks = 0
        try:
            yield sse_event("start", {
                "success":   True,
                "book_id":   book_id,
                "book_name": book_name,
                "source_file": file.filename,
            })

            pdf         = fitz.open(temp_path)
            total_pages = len(pdf)
            yield sse_event("pdf_info", {"book_id": book_id, "total_pages": total_pages})

            save_book_meta({
                "book_id":     book_id,
                "book_name":   book_name,
                "source_file": file.filename,
                "status":      "processing",
                "total_pages": total_pages,
                "created_at":  now_utc(),
                "updated_at":  now_utc(),
            })

            current_chapter     = None
            current_heading     = None
            global_paragraph_no = 0

            for i in range(total_pages):
                page_no   = i + 1
                page      = pdf.load_page(i)
                extracted = extract_page_text(page)
                page_text = extracted["text"]
                exm       = extracted["extraction_method"]

                yield sse_event("page_extracted", {
                    "page_no":          page_no,
                    "total_pages":      total_pages,
                    "char_count":       len(page_text),
                    "extraction_method": exm,
                    "progress_pct":     round((page_no / total_pages) * 100, 1),
                })

                blocks            = split_into_blocks(page_text)
                page_paragraph_no = 0

                for block in blocks:
                    lines = [l.strip() for l in block.split("\n") if l.strip()]
                    if not lines:
                        continue
                    paragraph_text = clean_paragraph(" ".join(lines))
                    if len(paragraph_text) < MIN_PARAGRAPH_CHARS:
                        continue

                    detected = detect_document_section(block, current_chapter, current_heading)
                    current_chapter = detected["chapter"] or current_chapter or "General Document"
                    current_heading = detected["heading_or_subtopic"] or current_heading or "General Content"

                    if detected.get("is_new_section"):
                        yield sse_event("heading_detected", {
                            "page_no":             page_no,
                            "chapter":             current_chapter,
                            "heading_or_subtopic": current_heading,
                            "detection_method":    detected["detection_method"],
                        })

                    page_paragraph_no   += 1
                    global_paragraph_no += 1
                    split_texts = sentence_aware_chunking(paragraph_text)

                    for idx, chunk_text in enumerate(split_texts, start=1):
                        total_chunks += 1
                        chunk = {
                            "chunk_id":                     str(uuid.uuid4()),
                            "book_id":                      book_id,
                            "book_name":                    book_name,
                            "source_file":                  file.filename,
                            "page_no":                      page_no,
                            "page_paragraph_no":            page_paragraph_no,
                            "global_paragraph_no":          global_paragraph_no,
                            "chapter":                      current_chapter,
                            "heading_or_subtopic":          current_heading,
                            "topic_detection_method":       detected["detection_method"],
                            "is_new_section":               detected.get("is_new_section", False),
                            "chunk_index_inside_paragraph": idx,
                            "total_chunks_from_paragraph":  len(split_texts),
                            "extraction_method":            exm,
                            "char_count":                   len(chunk_text),
                            "word_count":                   len(chunk_text.split()),
                            "text_preview":                 chunk_text[:250],
                            "text":                         chunk_text,
                            "status":                       "chunked",
                            "pinecone_indexed":              False,
                            "neo4j_indexed":                False,
                            "created_at":                   now_utc(),
                            "updated_at":                   now_utc(),
                        }
                        safe_insert_chunk(chunk)
                        yield sse_event("chunk", {
                            "chunk_id":    chunk["chunk_id"],
                            "page_no":     page_no,
                            "chapter":     current_chapter,
                            "heading":     current_heading,
                            "word_count":  chunk["word_count"],
                            "total_so_far":total_chunks,
                        })
                        await asyncio.sleep(0)

                yield sse_event("page_done", {
                    "page_no":           page_no,
                    "total_pages":       total_pages,
                    "total_chunks_so_far": total_chunks,
                    "progress_pct":      round((page_no / total_pages) * 100, 1),
                })

            save_book_meta({
                "book_id":      book_id,
                "book_name":    book_name,
                "source_file":  file.filename,
                "status":       "chunked",
                "total_pages":  total_pages,
                "total_chunks": total_chunks,
                "updated_at":   now_utc(),
            })

            yield sse_event("done", {
                "success":      True,
                "book_id":      book_id,
                "book_name":    book_name,
                "total_pages":  total_pages,
                "total_chunks": total_chunks,
            })

        except Exception as e:
            yield sse_event("error", {"success": False, "message": str(e)})
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":              "no-cache",
            "Connection":                 "keep-alive",
            "X-Accel-Buffering":          "no",
            "Access-Control-Allow-Origin":"*",
        },
    )


# =========================
# API — ADMIN: INDEX BOOK (SSE STREAM)
# =========================

@app.post("/v1/admin/books/index/stream")
async def index_book_stream(payload: IndexBookRequest):
    """
    Stream indexing progress via SSE.
    Embeds chunks → Pinecone + Neo4j in background with real-time updates.
    Events: start | batch_progress | batch_done | done | error
    """
    async def stream_generator():
        try:
            chunks = list(
                chunks_col.find(
                    {
                        "book_id": payload.book_id,
                        "$or": [
                            {"pinecone_indexed": {"$ne": True}},
                            {"neo4j_indexed":    {"$ne": True}},
                        ],
                    },
                    {"_id": 0},
                ).sort([("page_no", ASCENDING), ("global_paragraph_no", ASCENDING)])
            )

            total = len(chunks)

            if not chunks:
                yield sse_event("done", {
                    "success": True,
                    "message": "Already indexed or no chunks found",
                    "book_id": payload.book_id,
                    "indexed": 0,
                })
                return

            yield sse_event("start", {
                "book_id":      payload.book_id,
                "total_chunks": total,
                "batch_size":   BATCH_SIZE,
            })

            indexed        = 0
            failed_batches = 0

            for i in range(0, total, BATCH_SIZE):
                batch      = chunks[i: i + BATCH_SIZE]
                batch_num  = (i // BATCH_SIZE) + 1
                total_bats = (total + BATCH_SIZE - 1) // BATCH_SIZE

                yield sse_event("batch_progress", {
                    "batch_num":    batch_num,
                    "total_batches":total_bats,
                    "batch_size":   len(batch),
                    "indexed_so_far": indexed,
                    "progress_pct": round((indexed / total) * 100, 1),
                    "status":       "embedding",
                })

                try:
                    # Step 1: Pinecone embeddings
                    save_chunks_to_pinecone_batch(batch)
                    yield sse_event("batch_progress", {
                        "batch_num":    batch_num,
                        "total_batches":total_bats,
                        "indexed_so_far": indexed,
                        "progress_pct": round((indexed / total) * 100, 1),
                        "status":       "graph",
                    })

                    # Step 2: Neo4j graph
                    save_chunks_to_neo4j_batch(batch)

                    # Step 3: Update Mongo status
                    update_mongo_index_status_batch(batch)

                    indexed += len(batch)
                    yield sse_event("batch_done", {
                        "batch_num":    batch_num,
                        "total_batches":total_bats,
                        "batch_size":   len(batch),
                        "indexed_so_far": indexed,
                        "progress_pct": round((indexed / total) * 100, 1),
                    })

                except Exception as e:
                    failed_batches += 1
                    ops = [
                        UpdateOne(
                            {"chunk_id": c["chunk_id"]},
                            {"$set": {"indexing_error": str(e), "indexed_at": now_utc()}},
                        )
                        for c in batch
                    ]
                    chunks_col.bulk_write(ops, ordered=False)
                    yield sse_event("batch_error", {
                        "batch_num": batch_num,
                        "error":     str(e),
                    })

                await asyncio.sleep(0)

            # Update book status
            books_col.update_one(
                {"book_id": payload.book_id},
                {"$set": {"status": "indexed", "indexed_at": now_utc(), "updated_at": now_utc()}},
            )

            yield sse_event("done", {
                "success":        True,
                "book_id":        payload.book_id,
                "total_chunks":   total,
                "indexed":        indexed,
                "failed_batches": failed_batches,
            })

        except Exception as e:
            yield sse_event("error", {"success": False, "message": str(e)})

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":              "no-cache",
            "Connection":                 "keep-alive",
            "X-Accel-Buffering":          "no",
            "Access-Control-Allow-Origin":"*",
        },
    )


# =========================
# API — ADMIN: INDEX (fast, non-streaming)
# =========================

@app.post("/v1/admin/books/index")
async def index_book_fast(payload: IndexBookRequest):
    chunks = list(
        chunks_col.find(
            {
                "book_id": payload.book_id,
                "$or": [
                    {"pinecone_indexed": {"$ne": True}},
                    {"neo4j_indexed":    {"$ne": True}},
                ],
            },
            {"_id": 0},
        ).sort([("page_no", ASCENDING), ("global_paragraph_no", ASCENDING)])
    )

    if not chunks:
        return {"success": True, "message": "Already indexed or no chunks found", "book_id": payload.book_id}

    total = len(chunks)
    indexed = 0
    failed_batches = 0

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        try:
            save_chunks_to_pinecone_batch(batch)
            save_chunks_to_neo4j_batch(batch)
            update_mongo_index_status_batch(batch)
            indexed += len(batch)
        except Exception as e:
            failed_batches += 1

    books_col.update_one(
        {"book_id": payload.book_id},
        {"$set": {"status": "indexed", "indexed_at": now_utc(), "updated_at": now_utc()}},
    )

    return {
        "success":        True,
        "book_id":        payload.book_id,
        "total_chunks":   total,
        "indexed":        indexed,
        "failed_batches": failed_batches,
    }


# =========================
# API — ADMIN: LIST BOOKS
# =========================

@app.get("/v1/admin/books")
async def list_books(skip: int = 0, limit: int = 50):
    books = list(
        books_col.find({}, {"_id": 0})
        .sort("created_at", -1)
        .skip(skip)
        .limit(limit)
    )
    total = books_col.count_documents({})
    return {"success": True, "total": total, "books": books}


@app.get("/v1/admin/books/{book_id}")
async def get_book(book_id: str):
    book = books_col.find_one({"book_id": book_id}, {"_id": 0})
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")
    chunk_count = chunks_col.count_documents({"book_id": book_id})
    indexed_count = chunks_col.count_documents({"book_id": book_id, "pinecone_indexed": True})
    return {"success": True, "book": book, "chunk_count": chunk_count, "indexed_count": indexed_count}


@app.delete("/v1/admin/books/{book_id}")
async def delete_book(book_id: str):
    book = books_col.find_one({"book_id": book_id})
    if not book:
        raise HTTPException(status_code=404, detail="Book not found")

    # Remove from Mongo
    chunk_ids = [c["chunk_id"] for c in chunks_col.find({"book_id": book_id}, {"chunk_id": 1})]
    chunks_col.delete_many({"book_id": book_id})
    books_col.delete_one({"book_id": book_id})

    # Remove from Pinecone
    if chunk_ids:
        try:
            pinecone_index.delete(ids=chunk_ids, namespace=PINECONE_NAMESPACE)
        except Exception:
            pass

    # Remove from Neo4j
    try:
        with neo4j_driver.session() as session:
            session.run(
                "MATCH (b:Book {book_id: $book_id}) DETACH DELETE b",
                book_id=book_id,
            )
    except Exception:
        pass

    return {"success": True, "book_id": book_id, "chunks_deleted": len(chunk_ids)}


# =========================
# API — USER: CHAT
# =========================

@app.post("/v1/chat")
async def user_chat(payload: ChatRequest):
    """
    Main user chat endpoint.
    - Classifies query (chitchat vs rag_search)
    - For chitchat: replies from history + model knowledge, saves to history
    - For rag_search: full hybrid retrieval → long-form answer with references
    - Extracts + saves user facts automatically
    - Returns references for document-based answers
    """
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    user_profile  = get_or_create_user(payload.user_id)
    chat_history  = get_chat_history(payload.thread_id, limit=20)
    history_msgs  = build_history_messages(payload.thread_id, user_profile)

    # Async: extract facts from user message (non-blocking)
    asyncio.create_task(asyncio.to_thread(extract_user_facts, payload.query, payload.user_id))

    # Save user message
    save_chat_message(payload.thread_id, payload.user_id, "user", payload.query)

    # Classify
    route = classify_user_query(payload.query, chat_history)
    print("ROUTE:", route)

    # if route.get("route") == "chitchat" or not route.get("should_search_db"):

    force_rag_keywords = [
    "book", "pdf", "chapter", "topic", "lesson", "mcq",
    "summary", "explain from document", "uploaded"
    ]

    force_rag = any(k in payload.query.lower() for k in force_rag_keywords)

    if not force_rag and (route.get("route") == "chitchat" or not route.get("should_search_db")):
        # ── CHITCHAT ─────────────────────────────────────────────────────────
        reply = generate_chitchat_reply(payload.query, history_msgs, user_profile)
        save_chat_message(payload.thread_id, payload.user_id, "assistant", reply, {"route": "chitchat"})

        return {
            "success":     True,
            "route":       "chitchat",
            "answer":      reply,
            "references":  [],
            "thread_id":   payload.thread_id,
            "user_name":   user_profile.get("name"),
        }

    # ── RAG SEARCH ───────────────────────────────────────────────────────────
    query_plan = openai_query_planner(user_query=payload.query)

    search_queries = list(set(
        [query_plan.get("corrected_query", payload.query)]
        + query_plan.get("semantic_queries", [])
    ))
    keyword_queries = list(set(
        [query_plan.get("main_topic", payload.query)]
        + query_plan.get("keyword_queries", [])
        + query_plan.get("must_have_terms", [])
    ))

    # Hybrid retrieval across ALL books (no book_id filter)
    vector_results  = vector_search(queries=search_queries,  book_id=None, top_k=VECTOR_TOP_K)
    keyword_results = keyword_search(book_id=None, keyword_queries=keyword_queries)
    hybrid_results  = merge_hybrid_results(vector_results, keyword_results)

    if not hybrid_results:
        no_answer = "Sorry, I couldn't find relevant information in the knowledge base for your question."
        save_chat_message(payload.thread_id, payload.user_id, "assistant", no_answer, {"route": "rag_search", "found": False})
        return {
            "success":    True,
            "route":      "rag_search",
            "answer":     no_answer,
            "references": [],
            "confidence": 0,
            "thread_id":  payload.thread_id,
        }

    # Graph context expansion
    top_ids     = [x["chunk_id"] for x in hybrid_results[:30]]
    related_ids = get_related_chunks_from_neo4j(top_ids[:10])
    final_ids   = list(set(top_ids + related_ids))

    mongo_chunks = list(
        chunks_col.find({"chunk_id": {"$in": final_ids}}, {"_id": 0})
        .sort([("page_no", ASCENDING), ("global_paragraph_no", ASCENDING)])
    )

    if not mongo_chunks:
        no_answer = "Sorry, I couldn't find relevant information in the knowledge base."
        save_chat_message(payload.thread_id, payload.user_id, "assistant", no_answer, {"route": "rag_search", "found": False})
        return {
            "success":    True,
            "route":      "rag_search",
            "answer":     no_answer,
            "references": [],
            "confidence": 0,
            "thread_id":  payload.thread_id,
        }

    score_map = {x["chunk_id"]: x for x in hybrid_results}
    enriched = [
        {
            **c,
            "vector_score":  score_map.get(c["chunk_id"], {}).get("vector_score",  0),
            "keyword_score": score_map.get(c["chunk_id"], {}).get("keyword_score", 0),
            "hybrid_score":  score_map.get(c["chunk_id"], {}).get("hybrid_score",  0),
            "source":        "direct" if c["chunk_id"] in score_map else "neo4j_related",
        }
        for c in mongo_chunks
    ]

    diverse_chunks = sorted(enriched, key=lambda x: x.get("hybrid_score", 0), reverse=True)[:20]
    rerank_result  = openai_rerank(query_plan, diverse_chunks)
    confidence     = float(rerank_result.get("confidence", 0))
    selected_ids   = set(rerank_result.get("selected_chunk_ids", []))

    if confidence < MIN_FINAL_CONFIDENCE or not selected_ids:
        no_answer = "I found some related content but it doesn't seem directly relevant to your question. Could you rephrase or be more specific?"
        save_chat_message(payload.thread_id, payload.user_id, "assistant", no_answer, {"route": "rag_search", "confidence": confidence})
        return {
            "success":    True,
            "route":      "rag_search",
            "answer":     no_answer,
            "references": [],
            "confidence": confidence,
            "thread_id":  payload.thread_id,
        }

    final_chunks = [c for c in diverse_chunks if c["chunk_id"] in selected_ids]

    # Generate long-form answer
    answer = generate_long_answer(payload.query, final_chunks, history_msgs, user_profile)

    # Build references
    references = [
        {
            "chunk_id":     c["chunk_id"],
            "book_id":      c.get("book_id"),
            "book_name":    c.get("book_name"),
            "page_no":      c.get("page_no"),
            "chapter":      c.get("chapter"),
            "heading":      c.get("heading_or_subtopic"),
            "text_preview": c.get("text_preview"),
            "hybrid_score": c.get("hybrid_score"),
            "source":       c.get("source"),
        }
        for c in final_chunks
    ]

    save_chat_message(
        payload.thread_id, payload.user_id, "assistant", answer,
        {"route": "rag_search", "confidence": confidence, "chunks_used": len(final_chunks)},
    )

    return {
        "success":      True,
        "route":        "rag_search",
        "answer":       answer,
        "references":   references,
        "confidence":   confidence,
        "query_plan":   query_plan,
        "thread_id":    payload.thread_id,
        "user_name":    user_profile.get("name"),
        "chunks_used":  len(final_chunks),
    }


# =========================
# API — USER: GET THREADS
# =========================

@app.get("/v1/chat/threads/{user_id}")
async def get_user_threads(user_id: str):
    """Get all threads for a user with last message preview."""
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$sort": {"created_at": -1}},
        {"$group": {
            "_id":         "$thread_id",
            "last_msg":    {"$first": "$content"},
            "last_role":   {"$first": "$role"},
            "last_time":   {"$first": "$created_at"},
            "msg_count":   {"$sum": 1},
        }},
        {"$sort": {"last_time": -1}},
        {"$limit": 50},
    ]
    threads = list(chat_col.aggregate(pipeline))
    return {
        "success": True,
        "user_id": user_id,
        "threads": [
            {
                "thread_id": t["_id"],
                "last_msg":  t["last_msg"][:100] if t.get("last_msg") else "",
                "last_time": t.get("last_time"),
                "msg_count": t.get("msg_count", 0),
            }
            for t in threads
        ],
    }


@app.get("/v1/chat/history/{thread_id}")
async def get_thread_history(thread_id: str, limit: int = 50):
    """Get full message history for a thread."""
    messages = get_chat_history(thread_id, limit=limit)
    return {"success": True, "thread_id": thread_id, "messages": messages}


@app.get("/v1/user/{user_id}/profile")
async def get_user_profile(user_id: str):
    user = get_or_create_user(user_id)
    return {"success": True, "user": user}


# =========================
# API — MCQ GENERATOR
# =========================

@app.post("/v1/books/generate-mcq")
async def generate_mcq(payload: MCQRequest):
    if not payload.topic or not payload.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")

    chunks = fetch_topic_chunks(payload.topic, payload.book_id, top_k=20)

    if chunks:
        context_text = "\n\n---\n\n".join(
            f"[Chapter: {c.get('chapter','')} | Heading: {c.get('heading_or_subtopic','')} | Page: {c.get('page_no','')}]\n{c.get('text','')}"
            for c in chunks[:12]
        )
        source = "document_context"
    else:
        context_text = f"General knowledge about: {payload.topic}"
        source = "model_knowledge"

    difficulty_desc = {
        "easy":   "simple recall, definitions, and basic understanding",
        "medium": "application and conceptual understanding",
        "hard":   "analysis, evaluation, and deep understanding",
    }.get(payload.difficulty, "application and conceptual understanding")

    result = call_openai_json(
        system_prompt="""You are an expert educational MCQ generator.
Generate high-quality multiple choice questions with exactly 4 options (A, B, C, D), one correct answer, and a clear explanation.
Return ONLY valid JSON.""",
        user_prompt=f"""Topic: {payload.topic}
Difficulty: {payload.difficulty} ({difficulty_desc})
Number of questions: {payload.num_questions}

Context:
{context_text[:6000]}

Return JSON:
{{
  "topic": "{payload.topic}",
  "difficulty": "{payload.difficulty}",
  "questions": [
    {{
      "question_no": 1,
      "question": "question text here",
      "options": {{"A": "", "B": "", "C": "", "D": ""}},
      "correct_answer": "A",
      "explanation": "why this is correct",
      "chapter": "chapter name if known",
      "page_no": 0
    }}
  ]
}}"""
    )

    return {
        "success":       True,
        "topic":         payload.topic,
        "difficulty":    payload.difficulty,
        "source":        source,
        "chunks_used":   len(chunks),
        "mcq":           result,
    }


# =========================
# API — LESSON SUMMARY
# =========================

@app.post("/v1/books/generate-lesson-summary")
async def generate_lesson_summary(payload: LessonSummaryRequest):
    if not payload.topic or not payload.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty")

    chunks    = fetch_topic_chunks(payload.topic, payload.book_id, top_k=25)
    book_name = chunks[0].get("book_name", "") if chunks else ""

    if chunks:
        context_text = "\n\n---\n\n".join(
            f"[Chapter: {c.get('chapter','')} | Heading: {c.get('heading_or_subtopic','')} | Page: {c.get('page_no','')}]\n{c.get('text','')}"
            for c in chunks[:15]
        )
        source = "document_context"
    else:
        context_text = f"General knowledge about: {payload.topic}"
        source = "model_knowledge"

    result = call_openai_json(
        system_prompt="""You are an expert educational content writer.
Write a comprehensive, well-structured 2-page lesson summary for students.
Return ONLY valid JSON.""",
        user_prompt=f"""Topic: {payload.topic}
Book: {book_name}

Context:
{context_text[:8000]}

Return JSON:
{{
  "topic": "{payload.topic}",
  "book_name": "{book_name}",
  "summary": {{
    "overview": "2-3 sentence overview",
    "learning_objectives": ["obj1", "obj2", "obj3"],
    "page_1": {{
      "title": "Core Concepts and Fundamentals",
      "sections": [{{"heading": "", "content": "", "key_points": []}}]
    }},
    "page_2": {{
      "title": "Deep Dive and Applications",
      "sections": [{{"heading": "", "content": "", "key_points": []}}]
    }},
    "important_terms": [{{"term": "", "definition": ""}}],
    "summary_paragraph": "4-5 sentence recap",
    "exam_tips": ["tip1", "tip2", "tip3"]
  }}
}}"""
    )

    return {
        "success":     True,
        "topic":       payload.topic,
        "source":      source,
        "chunks_used": len(chunks),
        "lesson":      result,
    }


# =========================
# API — GRAPH DATA
# =========================

@app.get("/v1/books/{book_id}/graph")
async def get_book_graph(book_id: str, chunk_ids: str = ""):
    highlight_ids = set(chunk_ids.split(",")) if chunk_ids else set()

    query = """
    MATCH (b:Book {book_id: $book_id})
    OPTIONAL MATCH (b)-[:HAS_CHAPTER]->(c:Chapter)
    OPTIONAL MATCH (c)-[:HAS_HEADING]->(h:Heading)
    OPTIONAL MATCH (h)-[:HAS_CHUNK]  ->(ch:Chunk)
    OPTIONAL MATCH (b)-[:HAS_PAGE]   ->(p:Page)
    OPTIONAL MATCH (p)-[:HAS_CHUNK]  ->(ch2:Chunk)
    RETURN b, c, h, ch, p, ch2
    LIMIT 500
    """

    nodes: Dict[str, dict] = {}
    edges: List[dict]      = []

    try:
        with neo4j_driver.session() as session:
            result = session.run(query, book_id=book_id)

            def add_node(nid, label, props, ntype):
                if nid not in nodes:
                    nodes[nid] = {
                        "id":        nid,
                        "label":     label,
                        "type":      ntype,
                        "props":     props,
                        "highlight": nid in highlight_ids,
                    }

            def add_edge(src, tgt, rel):
                edges.append({"source": src, "target": tgt, "relation": rel})

            for record in result:
                b   = record.get("b")
                c   = record.get("c")
                h   = record.get("h")
                ch  = record.get("ch")
                p   = record.get("p")
                ch2 = record.get("ch2")

                if b:
                    bid = f"book_{b['book_id']}"
                    add_node(bid, b.get("name", "Book"), dict(b), "book")
                if c and b:
                    bid = f"book_{b['book_id']}"
                    cid = f"chapter_{b['book_id']}_{c['name']}"
                    add_node(cid, c.get("name", "Chapter"), dict(c), "chapter")
                    add_edge(bid, cid, "HAS_CHAPTER")
                if h and c and b:
                    cid = f"chapter_{b['book_id']}_{c['name']}"
                    hid = f"heading_{b['book_id']}_{c['name']}_{h['name']}"
                    add_node(hid, h.get("name", "Heading"), dict(h), "heading")
                    add_edge(cid, hid, "HAS_HEADING")
                if ch and h and c and b:
                    hid   = f"heading_{b['book_id']}_{c['name']}_{h['name']}"
                    ch_id = ch.get("chunk_id", "")
                    add_node(ch_id, f"Chunk p{ch.get('page_no','')}", dict(ch), "chunk")
                    add_edge(hid, ch_id, "HAS_CHUNK")
                if p and b:
                    bid = f"book_{b['book_id']}"
                    pid = f"page_{b['book_id']}_{p['page_no']}"
                    add_node(pid, f"Page {p.get('page_no','')}", dict(p), "page")
                    add_edge(bid, pid, "HAS_PAGE")
                if ch2 and p and b:
                    pid   = f"page_{b['book_id']}_{p['page_no']}"
                    ch_id = ch2.get("chunk_id", "")
                    if ch_id not in nodes:
                        add_node(ch_id, f"Chunk p{ch2.get('page_no','')}", dict(ch2), "chunk")
                    add_edge(pid, ch_id, "HAS_CHUNK")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j error: {e}")

    return {
        "success": True,
        "book_id": book_id,
        "nodes":   list(nodes.values()),
        "edges":   edges,
    }


# =========================
# HEALTH
# =========================

@app.get("/health")
async def health():
    return {
        "status":    "ok",
        "version":   "5.0.0",
        "timestamp": now_utc(),
        "pinecone_index":    PINECONE_INDEX_NAME,
        "pinecone_namespace": PINECONE_NAMESPACE,
    }




<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>GraphRAG — Knowledge Intelligence Platform</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
<style>
/* ─── DESIGN TOKENS ─────────────────────────────────────── */
:root {
  --bg:        #0a0d13;
  --surface:   #111520;
  --surface2:  #161c2a;
  --border:    #1e2a3f;
  --border2:   #253348;
  --accent:    #3b82f6;
  --accent2:   #6366f1;
  --accent3:   #10b981;
  --warn:      #f59e0b;
  --danger:    #ef4444;
  --text:      #e2e8f0;
  --text2:     #94a3b8;
  --text3:     #4a5568;
  --radius:    10px;
  --r-sm:      6px;
  --shadow:    0 4px 24px rgba(0,0,0,.45);
  --shadow-lg: 0 8px 48px rgba(0,0,0,.6);
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  --glow:      0 0 20px rgba(59,130,246,.2);
  --glow-g:    0 0 20px rgba(16,185,129,.2);
}

/* ─── RESET ─────────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:'Inter','Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;line-height:1.6}
a{color:inherit;text-decoration:none}
button{cursor:pointer;border:none;outline:none}
input,textarea,select{outline:none;border:none;background:none;font:inherit;color:inherit}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}

/* ─── LAYOUT ─────────────────────────────────────────────── */
#app{display:flex;height:100vh}

/* SIDEBAR */
.sidebar{
  width:220px;min-width:220px;
  background:var(--surface);
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;
  padding:0;
  transition:width .2s;
  z-index:100;
}
.sidebar-logo{
  padding:20px 18px 16px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:10px;
}
.logo-icon{
  width:32px;height:32px;border-radius:8px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;
  font-size:14px;font-weight:900;color:#fff;
  flex-shrink:0;
}
.logo-text{font-size:13px;font-weight:700;letter-spacing:.3px}
.logo-sub{font-size:10px;color:var(--text2)}

.sidebar-nav{flex:1;padding:12px 8px;display:flex;flex-direction:column;gap:2px;overflow-y:auto}
.nav-section-label{font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:var(--text3);padding:12px 10px 6px;font-weight:600}
.nav-btn{
  display:flex;align-items:center;gap:10px;
  padding:9px 10px;border-radius:var(--r-sm);
  font-size:12.5px;font-weight:500;color:var(--text2);
  background:none;transition:.15s;
  border:1px solid transparent;
  text-align:left;width:100%;
}
.nav-btn:hover{background:var(--surface2);color:var(--text);border-color:var(--border)}
.nav-btn.active{background:rgba(59,130,246,.12);color:var(--accent);border-color:rgba(59,130,246,.25)}
.nav-btn svg{width:15px;height:15px;flex-shrink:0;opacity:.8}
.nav-btn.active svg{opacity:1}

.sidebar-footer{padding:12px 8px;border-top:1px solid var(--border)}
.api-badge{
  display:flex;align-items:center;gap:7px;
  padding:8px 10px;border-radius:var(--r-sm);
  background:var(--surface2);font-size:11px;color:var(--text2);
}
.dot{width:7px;height:7px;border-radius:50%;background:var(--accent3);box-shadow:0 0 6px var(--accent3);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* MAIN */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.topbar{
  height:52px;min-height:52px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;padding:0 24px;gap:12px;
  background:var(--surface);
}
.topbar-title{font-size:14px;font-weight:600;flex:1}
.topbar-actions{display:flex;align-items:center;gap:8px}

.content{flex:1;overflow-y:auto;padding:24px}

/* ─── PANELS ─────────────────────────────────────────────── */
.panel{display:none;height:100%;flex-direction:column;gap:20px}
.panel.active{display:flex}

/* ─── CARDS ──────────────────────────────────────────────── */
.card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:20px;
}
.card-title{font-size:13px;font-weight:600;color:var(--text);margin-bottom:14px;display:flex;align-items:center;gap:8px}
.card-grid{display:grid;gap:16px}
.card-grid-2{grid-template-columns:1fr 1fr}
.card-grid-3{grid-template-columns:1fr 1fr 1fr}

/* ─── STATS ──────────────────────────────────────────────── */
.stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:4px}
.stat{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--radius);padding:16px 18px;
  transition:.2s;
}
.stat:hover{border-color:var(--border2);box-shadow:var(--glow)}
.stat-label{font-size:11px;color:var(--text2);margin-bottom:6px;font-weight:500}
.stat-value{font-size:24px;font-weight:700;color:var(--text);font-family:var(--font-mono)}
.stat-sub{font-size:10px;color:var(--text3);margin-top:3px}

/* ─── FORMS ──────────────────────────────────────────────── */
.form-group{display:flex;flex-direction:column;gap:6px}
.form-label{font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.6px}
.form-input{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--r-sm);padding:9px 12px;font-size:13px;color:var(--text);
  transition:.15s;width:100%;
}
.form-input:focus{border-color:var(--accent);box-shadow:0 0 0 2px rgba(59,130,246,.15)}
.form-input::placeholder{color:var(--text3)}

/* ─── BUTTONS ────────────────────────────────────────────── */
.btn{
  display:inline-flex;align-items:center;gap:7px;
  padding:9px 16px;border-radius:var(--r-sm);
  font-size:12.5px;font-weight:600;transition:.15s;
  border:1px solid transparent;
}
.btn-primary{background:var(--accent);color:#fff;box-shadow:0 2px 8px rgba(59,130,246,.3)}
.btn-primary:hover{background:#2563eb;box-shadow:0 4px 16px rgba(59,130,246,.4)}
.btn-secondary{background:var(--surface2);color:var(--text);border-color:var(--border)}
.btn-secondary:hover{border-color:var(--border2);color:var(--text)}
.btn-success{background:var(--accent3);color:#fff;box-shadow:0 2px 8px rgba(16,185,129,.3)}
.btn-success:hover{background:#059669}
.btn-danger{background:transparent;color:var(--danger);border-color:rgba(239,68,68,.3)}
.btn-danger:hover{background:rgba(239,68,68,.1)}
.btn-sm{padding:6px 12px;font-size:11.5px}
.btn:disabled{opacity:.4;cursor:not-allowed}

/* ─── UPLOAD ZONE ────────────────────────────────────────── */
.upload-zone{
  border:2px dashed var(--border2);border-radius:var(--radius);
  padding:40px;text-align:center;cursor:pointer;
  transition:.2s;background:var(--surface2);
  position:relative;
}
.upload-zone:hover,.upload-zone.dragover{border-color:var(--accent);background:rgba(59,130,246,.05)}
.upload-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.upload-icon{font-size:32px;margin-bottom:10px}
.upload-text{font-size:13px;color:var(--text2);margin-bottom:4px}
.upload-sub{font-size:11px;color:var(--text3)}

/* ─── PROGRESS / SSE ─────────────────────────────────────── */
.progress-container{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--radius);padding:18px;
  display:none;
}
.progress-container.active{display:block}
.progress-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.progress-title{font-size:12px;font-weight:600;color:var(--text)}
.progress-pct{font-size:12px;font-weight:700;color:var(--accent);font-family:var(--font-mono)}
.progress-bar-wrap{background:var(--border);border-radius:4px;height:6px;overflow:hidden;margin-bottom:10px}
.progress-bar{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width .3s;width:0%}
.progress-log{
  max-height:200px;overflow-y:auto;
  font-size:11px;font-family:var(--font-mono);color:var(--text2);
  display:flex;flex-direction:column;gap:3px;
}
.log-line{display:flex;gap:8px;align-items:flex-start}
.log-time{color:var(--text3);flex-shrink:0}
.log-event{color:var(--accent3)}
.log-msg{color:var(--text2);word-break:break-all}
.log-line.error .log-event{color:var(--danger)}
.log-line.warn .log-event{color:var(--warn)}

/* ─── BOOK LIST ──────────────────────────────────────────── */
.book-list{display:flex;flex-direction:column;gap:10px}
.book-item{
  background:var(--surface2);border:1px solid var(--border);
  border-radius:var(--radius);padding:16px 18px;
  display:flex;align-items:center;gap:14px;
  transition:.15s;
}
.book-item:hover{border-color:var(--border2)}
.book-icon{
  width:40px;height:40px;border-radius:8px;
  background:linear-gradient(135deg,rgba(59,130,246,.2),rgba(99,102,241,.2));
  display:flex;align-items:center;justify-content:center;
  font-size:18px;flex-shrink:0;
  border:1px solid rgba(59,130,246,.2);
}
.book-info{flex:1;min-width:0}
.book-name{font-size:13px;font-weight:600;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.book-meta{font-size:11px;color:var(--text2);margin-top:3px;display:flex;gap:12px;flex-wrap:wrap}
.book-meta span{display:flex;align-items:center;gap:4px}
.book-actions{display:flex;gap:6px;flex-shrink:0}
.status-badge{
  display:inline-flex;align-items:center;gap:5px;
  padding:3px 8px;border-radius:20px;font-size:10px;font-weight:600;
  text-transform:uppercase;letter-spacing:.5px;
}
.status-chunked{background:rgba(245,158,11,.1);color:var(--warn);border:1px solid rgba(245,158,11,.2)}
.status-indexed{background:rgba(16,185,129,.1);color:var(--accent3);border:1px solid rgba(16,185,129,.2)}
.status-processing{background:rgba(59,130,246,.1);color:var(--accent);border:1px solid rgba(59,130,246,.2)}

/* ─── GRAPH ──────────────────────────────────────────────── */
.graph-wrap{position:relative;flex:1;min-height:500px}
#graphCanvas{
  width:100%;height:100%;border-radius:var(--radius);
  background:var(--surface2);border:1px solid var(--border);
  cursor:grab;
}
#graphCanvas:active{cursor:grabbing}
.graph-legend{
  position:absolute;top:14px;left:14px;
  background:rgba(10,13,19,.85);backdrop-filter:blur(8px);
  border:1px solid var(--border);border-radius:var(--r-sm);
  padding:12px;display:flex;flex-direction:column;gap:6px;
}
.legend-item{display:flex;align-items:center;gap:8px;font-size:11px;color:var(--text2)}
.legend-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.graph-controls{
  position:absolute;bottom:14px;right:14px;
  display:flex;flex-direction:column;gap:6px;
}
.graph-btn{
  width:32px;height:32px;border-radius:var(--r-sm);
  background:rgba(10,13,19,.85);backdrop-filter:blur(8px);
  border:1px solid var(--border);color:var(--text);font-size:14px;
  display:flex;align-items:center;justify-content:center;
  cursor:pointer;transition:.15s;
}
.graph-btn:hover{border-color:var(--accent);color:var(--accent)}
.graph-tooltip{
  position:absolute;background:rgba(10,13,19,.95);backdrop-filter:blur(8px);
  border:1px solid var(--border2);border-radius:var(--r-sm);
  padding:10px 12px;font-size:11px;color:var(--text);
  pointer-events:none;display:none;max-width:220px;z-index:10;
  box-shadow:var(--shadow);
}

/* ─── CHAT ───────────────────────────────────────────────── */
.chat-layout{display:flex;height:100%;gap:0;overflow:hidden}
.chat-sidebar{
  width:220px;min-width:220px;
  background:var(--surface);border-right:1px solid var(--border);
  display:flex;flex-direction:column;
}
.chat-sidebar-header{padding:14px 14px 10px;border-bottom:1px solid var(--border)}
.chat-sidebar-title{font-size:12px;font-weight:600;color:var(--text);margin-bottom:8px}
.thread-list{flex:1;overflow-y:auto;padding:8px}
.thread-item{
  padding:10px;border-radius:var(--r-sm);cursor:pointer;
  transition:.15s;border:1px solid transparent;margin-bottom:4px;
}
.thread-item:hover{background:var(--surface2)}
.thread-item.active{background:rgba(59,130,246,.1);border-color:rgba(59,130,246,.2)}
.thread-preview{font-size:11.5px;color:var(--text2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.thread-time{font-size:10px;color:var(--text3);margin-top:3px}

.chat-main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.chat-topbar{
  padding:12px 20px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:12px;background:var(--surface);
}
.user-avatar{
  width:30px;height:30px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;
  font-size:12px;font-weight:700;color:#fff;flex-shrink:0;
}
.chat-user-info{flex:1}
.chat-user-name{font-size:12.5px;font-weight:600;color:var(--text)}
.chat-user-sub{font-size:10.5px;color:var(--text2)}
.chat-route-badge{
  padding:3px 8px;border-radius:20px;font-size:10px;font-weight:600;
  text-transform:uppercase;letter-spacing:.5px;
}
.route-rag{background:rgba(59,130,246,.1);color:var(--accent);border:1px solid rgba(59,130,246,.2)}
.route-chat{background:rgba(16,185,129,.1);color:var(--accent3);border:1px solid rgba(16,185,129,.2)}

.messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:16px}
.msg{display:flex;gap:12px;align-items:flex-start;max-width:100%}
.msg.user{flex-direction:row-reverse}
.msg-avatar{
  width:28px;height:28px;border-radius:50%;flex-shrink:0;
  display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;
}
.msg.user .msg-avatar{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff}
.msg.assistant .msg-avatar{background:linear-gradient(135deg,rgba(16,185,129,.3),rgba(59,130,246,.3));border:1px solid var(--border2);color:var(--text2)}

.msg-body{max-width:78%;min-width:40px}
.msg-bubble{
  padding:12px 15px;border-radius:12px;font-size:13px;line-height:1.65;
  word-break:break-word;
}
.msg.user .msg-bubble{
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  color:#fff;border-radius:12px 2px 12px 12px;
}
.msg.assistant .msg-bubble{
  background:var(--surface2);border:1px solid var(--border);
  color:var(--text);border-radius:2px 12px 12px 12px;
}
.msg.assistant .msg-bubble h1,.msg.assistant .msg-bubble h2,.msg.assistant .msg-bubble h3{
  font-size:14px;font-weight:700;margin:10px 0 6px;color:var(--text);
}
.msg.assistant .msg-bubble p{margin-bottom:8px}
.msg.assistant .msg-bubble ul,.msg.assistant .msg-bubble ol{padding-left:18px;margin-bottom:8px}
.msg.assistant .msg-bubble li{margin-bottom:3px;font-size:12.5px}
.msg.assistant .msg-bubble code{
  background:rgba(59,130,246,.1);padding:1px 5px;border-radius:3px;
  font-family:var(--font-mono);font-size:11.5px;color:var(--accent);
}
.msg.assistant .msg-bubble strong{color:var(--text);font-weight:600}
.msg-time{font-size:10px;color:var(--text3);margin-top:4px;text-align:right}
.msg.user .msg-time{text-align:right}
.msg.assistant .msg-time{text-align:left}

.references{margin-top:10px;display:flex;flex-direction:column;gap:6px}
.ref-header{font-size:10px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px}
.ref-chip{
  display:flex;align-items:center;gap:8px;
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r-sm);padding:7px 10px;cursor:pointer;
  transition:.15s;font-size:11px;
}
.ref-chip:hover{border-color:var(--border2);background:var(--surface2)}
.ref-book{font-weight:600;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}
.ref-page{color:var(--text2);flex-shrink:0}
.ref-score{
  padding:2px 6px;border-radius:10px;font-size:9px;font-weight:700;
  background:rgba(16,185,129,.1);color:var(--accent3);flex-shrink:0;
}

.typing-indicator{display:flex;gap:4px;padding:4px 0}
.typing-dot{width:7px;height:7px;border-radius:50%;background:var(--text3);animation:typing 1.2s infinite}
.typing-dot:nth-child(2){animation-delay:.2s}
.typing-dot:nth-child(3){animation-delay:.4s}
@keyframes typing{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px)}}

.chat-input-area{
  padding:14px 20px;border-top:1px solid var(--border);
  background:var(--surface);
}
.chat-input-row{display:flex;gap:8px;align-items:flex-end}
.chat-textarea{
  flex:1;background:var(--surface2);border:1px solid var(--border);
  border-radius:10px;padding:11px 14px;font-size:13px;color:var(--text);
  resize:none;min-height:44px;max-height:140px;line-height:1.5;
  transition:.15s;font-family:inherit;
}
.chat-textarea:focus{border-color:var(--accent);box-shadow:0 0 0 2px rgba(59,130,246,.12)}
.chat-textarea::placeholder{color:var(--text3)}
.send-btn{
  width:44px;height:44px;border-radius:10px;flex-shrink:0;
  background:var(--accent);color:#fff;font-size:16px;
  display:flex;align-items:center;justify-content:center;
  transition:.15s;box-shadow:0 2px 8px rgba(59,130,246,.3);
}
.send-btn:hover{background:#2563eb;box-shadow:0 4px 16px rgba(59,130,246,.4)}
.send-btn:disabled{opacity:.4;cursor:not-allowed}

/* ─── USER SETUP ─────────────────────────────────────────── */
.user-setup{
  position:fixed;inset:0;background:rgba(10,13,19,.95);backdrop-filter:blur(10px);
  z-index:200;display:flex;align-items:center;justify-content:center;
}
.user-setup-card{
  background:var(--surface);border:1px solid var(--border2);border-radius:16px;
  padding:36px;max-width:420px;width:100%;box-shadow:var(--shadow-lg);
  text-align:center;
}
.setup-logo{
  width:56px;height:56px;border-radius:14px;margin:0 auto 20px;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:900;color:#fff;
}
.setup-title{font-size:20px;font-weight:700;margin-bottom:8px}
.setup-sub{font-size:13px;color:var(--text2);margin-bottom:24px;line-height:1.5}

/* ─── EMPTY STATE ────────────────────────────────────────── */
.empty{text-align:center;padding:60px 20px;color:var(--text2)}
.empty-icon{font-size:40px;margin-bottom:12px;opacity:.4}
.empty-title{font-size:14px;font-weight:600;color:var(--text);margin-bottom:6px}
.empty-sub{font-size:12px;color:var(--text2)}

/* ─── TOAST ──────────────────────────────────────────────── */
.toast-container{position:fixed;bottom:20px;right:20px;z-index:500;display:flex;flex-direction:column;gap:8px}
.toast{
  background:var(--surface2);border:1px solid var(--border2);
  border-radius:var(--r-sm);padding:12px 16px;
  font-size:12.5px;color:var(--text);box-shadow:var(--shadow);
  display:flex;align-items:center;gap:10px;min-width:240px;
  animation:slideIn .2s ease;
}
@keyframes slideIn{from{transform:translateX(100%);opacity:0}to{transform:translateX(0);opacity:1}}
.toast.success{border-left:3px solid var(--accent3)}
.toast.error{border-left:3px solid var(--danger)}
.toast.info{border-left:3px solid var(--accent)}

/* ─── MODAL ──────────────────────────────────────────────── */
.modal-overlay{
  position:fixed;inset:0;background:rgba(10,13,19,.8);backdrop-filter:blur(6px);
  z-index:300;display:flex;align-items:center;justify-content:center;
  opacity:0;pointer-events:none;transition:.2s;
}
.modal-overlay.open{opacity:1;pointer-events:all}
.modal{
  background:var(--surface);border:1px solid var(--border2);
  border-radius:16px;padding:28px;max-width:560px;width:90%;
  box-shadow:var(--shadow-lg);transform:scale(.95);transition:.2s;
}
.modal-overlay.open .modal{transform:scale(1)}
.modal-title{font-size:15px;font-weight:700;margin-bottom:16px;display:flex;justify-content:space-between;align-items:center}
.modal-close{background:none;color:var(--text2);font-size:18px;cursor:pointer;padding:2px 6px;border-radius:4px}
.modal-close:hover{background:var(--surface2);color:var(--text)}

/* ─── MISC ───────────────────────────────────────────────── */
.sep{border:none;border-top:1px solid var(--border);margin:4px 0}
.flex{display:flex}.gap-2{gap:8px}.gap-3{gap:12px}.items-center{align-items:center}
.flex-1{flex:1}.justify-between{justify-content:space-between}
.mt-2{margin-top:8px}.mt-3{margin-top:12px}.mt-4{margin-top:16px}
.text-xs{font-size:11px}.text-sm{font-size:12.5px}
.text-muted{color:var(--text2)}.text-accent{color:var(--accent)}.text-green{color:var(--accent3)}
.font-mono{font-family:var(--font-mono)}
.truncate{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.w-full{width:100%}
.spinner{
  width:16px;height:16px;border:2px solid rgba(255,255,255,.2);
  border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
.confidence-bar{height:4px;border-radius:2px;background:var(--border);overflow:hidden;margin-top:4px}
.confidence-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--accent3),var(--accent));transition:width .4s}
</style>
</head>
<body>

<!-- USER SETUP MODAL -->
<div class="user-setup" id="userSetup">
  <div class="user-setup-card">
    <div class="setup-logo">G</div>
    <div class="setup-title">Welcome to GraphRAG</div>
    <div class="setup-sub">Knowledge Intelligence Platform.<br>Enter your name to personalize your experience.</div>
    <div class="form-group" style="text-align:left;margin-bottom:16px">
      <label class="form-label">Your Name</label>
      <input class="form-input" id="setupName" placeholder="e.g. Arjun Sharma" style="text-align:center;font-size:14px"/>
    </div>
    <div class="form-group" style="text-align:left;margin-bottom:20px">
      <label class="form-label">API Base URL</label>
      <input class="form-input" id="setupApi" value="http://localhost:8000" placeholder="http://localhost:8000"/>
    </div>
    <button class="btn btn-primary w-full" onclick="setupUser()" style="justify-content:center;padding:12px">
      Get Started →
    </button>
  </div>
</div>

<!-- APP -->
<div id="app">
  <!-- SIDEBAR -->
  <div class="sidebar">
    <div class="sidebar-logo">
      <div class="logo-icon">G</div>
      <div>
        <div class="logo-text">GraphRAG</div>
        <div class="logo-sub">Knowledge Platform</div>
      </div>
    </div>
    <div class="sidebar-nav">
      <div class="nav-section-label">User</div>
      <button class="nav-btn active" onclick="showPanel('chat')" id="nav-chat">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        Chat
      </button>

      <div class="nav-section-label">Admin</div>
      <button class="nav-btn" onclick="showPanel('upload')" id="nav-upload">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        Upload Book
      </button>
      <button class="nav-btn" onclick="showPanel('books')" id="nav-books">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
        Books
      </button>
      <button class="nav-btn" onclick="showPanel('graph')" id="nav-graph">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
        Graph
      </button>
      <button class="nav-btn" onclick="showPanel('tools')" id="nav-tools">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>
        AI Tools
      </button>
    </div>
    <div class="sidebar-footer">
      <div class="api-badge">
        <div class="dot" id="apiDot"></div>
        <span id="apiStatus">Connecting…</span>
      </div>
    </div>
  </div>

  <!-- MAIN -->
  <div class="main">
    <!-- TOPBAR -->
    <div class="topbar">
      <div class="topbar-title" id="topbarTitle">Chat</div>
      <div class="topbar-actions">
        <div id="userBadge" class="api-badge" style="gap:8px;cursor:pointer" onclick="showPanel('chat')">
          <div class="user-avatar" id="userAvatarTop" style="width:24px;height:24px;font-size:10px">?</div>
          <span id="userNameTop">Guest</span>
        </div>
      </div>
    </div>

    <div class="content">

      <!-- ── CHAT PANEL ───────────────────────────────────────── -->
      <div class="panel active" id="panel-chat" style="padding:0;height:100%">
        <div class="chat-layout">
          <!-- Chat Sidebar -->
          <div class="chat-sidebar">
            <div class="chat-sidebar-header">
              <div class="chat-sidebar-title">Conversations</div>
              <button class="btn btn-primary btn-sm w-full" onclick="newThread()">+ New Chat</button>
            </div>
            <div class="thread-list" id="threadList">
              <div class="empty" style="padding:30px 10px">
                <div style="font-size:20px;margin-bottom:6px">💬</div>
                <div style="font-size:11px;color:var(--text3)">No conversations yet</div>
              </div>
            </div>
          </div>
          <!-- Chat Main -->
          <div class="chat-main">
            <div class="chat-topbar">
              <div class="user-avatar" id="chatUserAvatar">?</div>
              <div class="chat-user-info">
                <div class="chat-user-name" id="chatUserName">Guest User</div>
                <div class="chat-user-sub" id="chatUserSub">Ask anything from your knowledge base</div>
              </div>
              <div id="routeBadge" style="display:none"></div>
            </div>
            <div class="messages" id="messages">
              <div style="text-align:center;padding:40px 20px;color:var(--text2)">
                <div style="font-size:32px;margin-bottom:12px">📚</div>
                <div style="font-size:14px;font-weight:600;color:var(--text);margin-bottom:6px">Ask your knowledge base</div>
                <div style="font-size:12px">Search across all uploaded books instantly</div>
              </div>
            </div>
            <div class="chat-input-area">
              <div class="chat-input-row">
                <textarea class="chat-textarea" id="chatInput" rows="1"
                  placeholder="Ask anything… (Enter to send, Shift+Enter for new line)"
                  onkeydown="handleChatKey(event)" oninput="autoResize(this)"></textarea>
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ── UPLOAD PANEL ─────────────────────────────────────── -->
      <div class="panel" id="panel-upload">
        <div class="stat-row">
          <div class="stat">
            <div class="stat-label">Total Books</div>
            <div class="stat-value font-mono" id="statBooks">—</div>
            <div class="stat-sub">in knowledge base</div>
          </div>
          <div class="stat">
            <div class="stat-label">Indexed Books</div>
            <div class="stat-value font-mono" id="statIndexed">—</div>
            <div class="stat-sub">ready to search</div>
          </div>
          <div class="stat">
            <div class="stat-label">Total Chunks</div>
            <div class="stat-value font-mono" id="statChunks">—</div>
            <div class="stat-sub">across all books</div>
          </div>
          <div class="stat">
            <div class="stat-label">Namespace</div>
            <div class="stat-value" style="font-size:13px" id="statNS">all_documents</div>
            <div class="stat-sub">Pinecone namespace</div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
            Upload Book
          </div>
          <div class="card-grid card-grid-2">
            <div class="form-group">
              <label class="form-label">Book Name</label>
              <input class="form-input" id="bookName" placeholder="e.g. Physics Class 11"/>
            </div>
            <div class="form-group">
              <label class="form-label">Chunk Mode</label>
              <select class="form-input" id="chunkMode">
                <option value="stream">Streaming (real-time progress)</option>
                <option value="sync">Sync (wait for completion)</option>
              </select>
            </div>
          </div>
          <div style="margin-top:14px">
            <div class="upload-zone" id="uploadZone">
              <input type="file" id="pdfInput" accept=".pdf" onchange="onFileSelect(event)"/>
              <div class="upload-icon">📄</div>
              <div class="upload-text">Drop PDF here or click to browse</div>
              <div class="upload-sub" id="fileNameDisplay">Only PDF files supported</div>
            </div>
          </div>
          <div style="margin-top:14px;display:flex;gap:8px">
            <button class="btn btn-primary" id="uploadBtn" onclick="uploadBook()">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
              Upload & Chunk
            </button>
            <button class="btn btn-secondary" id="clearUploadBtn" onclick="clearUpload()" style="display:none">Clear</button>
          </div>
        </div>

        <!-- SSE Progress -->
        <div class="progress-container" id="uploadProgress">
          <div class="progress-header">
            <span class="progress-title" id="progressTitle">Processing…</span>
            <span class="progress-pct" id="progressPct">0%</span>
          </div>
          <div class="progress-bar-wrap">
            <div class="progress-bar" id="progressBar"></div>
          </div>
          <div class="progress-log" id="progressLog"></div>
        </div>

        <!-- After chunk: index button -->
        <div id="indexSection" style="display:none">
          <div class="card" style="border-color:rgba(245,158,11,.2);background:rgba(245,158,11,.03)">
            <div class="card-title">
              ⚡ Ready to Index
            </div>
            <p style="font-size:12.5px;color:var(--text2);margin-bottom:14px">
              Chunking complete! Now embed into Pinecone vector DB + Neo4j graph.
              This runs in background with real-time progress.
            </p>
            <div style="display:flex;gap:8px">
              <button class="btn btn-success" id="indexBtn" onclick="startIndexing()">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                Start Indexing (Streaming)
              </button>
              <button class="btn btn-secondary" id="indexFastBtn" onclick="startIndexingFast()">Fast Index</button>
            </div>
          </div>
          <!-- Index SSE Progress -->
          <div class="progress-container mt-3" id="indexProgress">
            <div class="progress-header">
              <span class="progress-title" id="indexProgressTitle">Indexing…</span>
              <span class="progress-pct" id="indexProgressPct">0%</span>
            </div>
            <div class="progress-bar-wrap">
              <div class="progress-bar" id="indexProgressBar"></div>
            </div>
            <div class="progress-log" id="indexProgressLog"></div>
          </div>
        </div>
      </div>

      <!-- ── BOOKS PANEL ──────────────────────────────────────── -->
      <div class="panel" id="panel-books">
        <div class="flex items-center justify-between">
          <div style="font-size:14px;font-weight:600">Knowledge Base</div>
          <button class="btn btn-secondary btn-sm" onclick="loadBooks()">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-.49-7"/></svg>
            Refresh
          </button>
        </div>
        <div class="book-list" id="bookList">
          <div class="empty"><div class="empty-icon">📚</div><div class="empty-title">No books yet</div><div class="empty-sub">Upload a PDF to get started</div></div>
        </div>
      </div>

      <!-- ── GRAPH PANEL ──────────────────────────────────────── -->
      <div class="panel" id="panel-graph">
        <div class="flex items-center gap-3">
          <div class="form-group" style="flex:1">
            <select class="form-input" id="graphBookSelect" onchange="loadGraph()">
              <option value="">Select a book to visualize…</option>
            </select>
          </div>
          <button class="btn btn-secondary" onclick="loadGraph()">Load Graph</button>
          <button class="btn btn-secondary" onclick="resetGraphView()">Reset View</button>
        </div>
        <div class="graph-wrap">
          <canvas id="graphCanvas"></canvas>
          <div class="graph-legend">
            <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>Book</div>
            <div class="legend-item"><div class="legend-dot" style="background:#3b82f6"></div>Chapter</div>
            <div class="legend-item"><div class="legend-dot" style="background:#8b5cf6"></div>Heading</div>
            <div class="legend-item"><div class="legend-dot" style="background:#10b981"></div>Chunk</div>
            <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>Page</div>
          </div>
          <div class="graph-controls">
            <div class="graph-btn" onclick="graphZoom(1.2)" title="Zoom In">+</div>
            <div class="graph-btn" onclick="graphZoom(0.8)" title="Zoom Out">−</div>
            <div class="graph-btn" onclick="resetGraphView()" title="Reset">⌂</div>
          </div>
          <div class="graph-tooltip" id="graphTooltip"></div>
        </div>
        <div id="graphStats" style="font-size:11px;color:var(--text2)"></div>
      </div>

      <!-- ── TOOLS PANEL ──────────────────────────────────────── -->
      <div class="panel" id="panel-tools">
        <div class="card-grid card-grid-2">
          <!-- MCQ Generator -->
          <div class="card">
            <div class="card-title">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
              MCQ Generator
            </div>
            <div class="form-group">
              <label class="form-label">Topic</label>
              <input class="form-input" id="mcqTopic" placeholder="e.g. Newton's Laws of Motion"/>
            </div>
            <div class="card-grid card-grid-2 mt-2">
              <div class="form-group">
                <label class="form-label">Questions</label>
                <input class="form-input" id="mcqCount" type="number" value="5" min="1" max="20"/>
              </div>
              <div class="form-group">
                <label class="form-label">Difficulty</label>
                <select class="form-input" id="mcqDiff">
                  <option value="easy">Easy</option>
                  <option value="medium" selected>Medium</option>
                  <option value="hard">Hard</option>
                </select>
              </div>
            </div>
            <button class="btn btn-primary mt-3 w-full" onclick="generateMCQ()" style="justify-content:center" id="mcqBtn">Generate MCQ</button>
            <div id="mcqResult" style="margin-top:14px"></div>
          </div>
          <!-- Lesson Summary -->
          <div class="card">
            <div class="card-title">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
              Lesson Summary
            </div>
            <div class="form-group">
              <label class="form-label">Topic</label>
              <input class="form-input" id="summaryTopic" placeholder="e.g. Photosynthesis"/>
            </div>
            <button class="btn btn-primary mt-3 w-full" onclick="generateSummary()" style="justify-content:center" id="summaryBtn">Generate Summary</button>
            <div id="summaryResult" style="margin-top:14px"></div>
          </div>
        </div>
      </div>

    </div><!-- /content -->
  </div><!-- /main -->
</div><!-- /app -->

<!-- TOAST CONTAINER -->
<div class="toast-container" id="toastContainer"></div>

<!-- REF MODAL -->
<div class="modal-overlay" id="refModal">
  <div class="modal">
    <div class="modal-title">
      <span>📄 Reference Chunk</span>
      <button class="modal-close" onclick="closeModal('refModal')">✕</button>
    </div>
    <div id="refModalContent"></div>
  </div>
</div>

<script>
/* ═══════════════════════════════════════════════════════════
   STATE
═══════════════════════════════════════════════════════════ */
let API = 'http://localhost:8000';
let currentUser = { user_id: null, name: null };
let currentThread = null;
let chatSending = false;
let pendingBookId = null;   // after chunk, before index

// Graph state
let graphData = { nodes: [], edges: [] };
let graphOffset = { x: 0, y: 0 };
let graphScale  = 1;
let graphDragging = false;
let graphDragStart = { x: 0, y: 0 };
let graphNodePositions = {};
let graphAnimFrame = null;
let graphHover = null;

/* ═══════════════════════════════════════════════════════════
   INIT
═══════════════════════════════════════════════════════════ */
window.onload = () => {
  const saved = localStorage.getItem('gr_user');
  if (saved) {
    currentUser = JSON.parse(saved);
    API = localStorage.getItem('gr_api') || API;
    document.getElementById('userSetup').style.display = 'none';
    initApp();
  }
  // Setup drag-and-drop
  const zone = document.getElementById('uploadZone');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    const f = e.dataTransfer.files[0];
    if (f) { document.getElementById('pdfInput').files = e.dataTransfer.files; onFileSelect({ target: { files: [f] } }); }
  });
};

function setupUser() {
  const name = document.getElementById('setupName').value.trim();
  const api  = document.getElementById('setupApi').value.trim();
  if (!name) { toast('Please enter your name', 'error'); return; }
  currentUser = { user_id: 'user_' + Date.now(), name };
  API = api || API;
  localStorage.setItem('gr_user', JSON.stringify(currentUser));
  localStorage.setItem('gr_api', API);
  document.getElementById('userSetup').style.display = 'none';
  initApp();
}

function initApp() {
  updateUserUI();
  checkAPI();
  loadStats();
  loadBooks();
  loadThreads();
  newThread();
}

function updateUserUI() {
  const n = currentUser.name || 'User';
  const ini = n.charAt(0).toUpperCase();
  document.getElementById('userAvatarTop').textContent = ini;
  document.getElementById('userNameTop').textContent = n;
  document.getElementById('chatUserAvatar').textContent = ini;
  document.getElementById('chatUserName').textContent = n;
}

async function checkAPI() {
  try {
    const r = await fetch(API + '/health');
    if (r.ok) {
      document.getElementById('apiDot').style.background = 'var(--accent3)';
      document.getElementById('apiStatus').textContent = 'Connected';
    } else throw new Error();
  } catch {
    document.getElementById('apiDot').style.background = 'var(--danger)';
    document.getElementById('apiStatus').textContent = 'Offline';
  }
}

/* ═══════════════════════════════════════════════════════════
   PANELS
═══════════════════════════════════════════════════════════ */
const panelTitles = { chat:'Chat', upload:'Upload Book', books:'Knowledge Base', graph:'Graph Explorer', tools:'AI Tools' };

function showPanel(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
  document.getElementById('nav-' + name).classList.add('active');
  document.getElementById('topbarTitle').textContent = panelTitles[name];
  if (name === 'books') loadBooks();
  if (name === 'graph') populateGraphSelect();
}

/* ═══════════════════════════════════════════════════════════
   STATS
═══════════════════════════════════════════════════════════ */
async function loadStats() {
  try {
    const r = await fetch(API + '/v1/admin/books?limit=500');
    const d = await r.json();
    const books = d.books || [];
    const indexed = books.filter(b => b.status === 'indexed').length;
    const chunks = books.reduce((s, b) => s + (b.total_chunks || 0), 0);
    document.getElementById('statBooks').textContent = books.length;
    document.getElementById('statIndexed').textContent = indexed;
    document.getElementById('statChunks').textContent = chunks.toLocaleString();
  } catch {}
}

/* ═══════════════════════════════════════════════════════════
   UPLOAD + SSE CHUNKING
═══════════════════════════════════════════════════════════ */
function onFileSelect(e) {
  const f = e.target.files[0];
  if (f) {
    document.getElementById('fileNameDisplay').textContent = '📄 ' + f.name;
    if (!document.getElementById('bookName').value)
      document.getElementById('bookName').value = f.name.replace('.pdf', '');
  }
}

function clearUpload() {
  document.getElementById('pdfInput').value = '';
  document.getElementById('fileNameDisplay').textContent = 'Only PDF files supported';
  document.getElementById('bookName').value = '';
  document.getElementById('uploadProgress').classList.remove('active');
  document.getElementById('indexSection').style.display = 'none';
  document.getElementById('clearUploadBtn').style.display = 'none';
  pendingBookId = null;
}

async function uploadBook() {
  const bookName = document.getElementById('bookName').value.trim();
  const file     = document.getElementById('pdfInput').files[0];
  const mode     = document.getElementById('chunkMode').value;

  if (!bookName) { toast('Enter book name', 'error'); return; }
  if (!file)     { toast('Select a PDF file', 'error'); return; }

  const btn = document.getElementById('uploadBtn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div> Processing…';

  if (mode === 'stream') {
    await uploadBookStream(bookName, file);
  } else {
    await uploadBookSync(bookName, file);
  }
}

async function uploadBookSync(bookName, file) {
  try {
    const fd = new FormData();
    fd.append('book_name', bookName);
    fd.append('file', file);
    const r = await fetch(API + '/v1/admin/books/chunk', { method: 'POST', body: fd });
    const d = await r.json();
    if (d.success) {
      pendingBookId = d.book_id;
      toast(`Chunked: ${d.total_chunks} chunks from ${d.total_pages} pages`, 'success');
      document.getElementById('indexSection').style.display = 'block';
      document.getElementById('clearUploadBtn').style.display = 'inline-flex';
      loadStats();
      loadBooks();
    } else {
      toast('Upload failed: ' + (d.detail || 'Unknown error'), 'error');
    }
  } catch(e) {
    toast('Upload error: ' + e.message, 'error');
  } finally {
    const btn = document.getElementById('uploadBtn');
    btn.disabled = false;
    btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg> Upload & Chunk`;
  }
}

async function uploadBookStream(bookName, file) {
  const progress = document.getElementById('uploadProgress');
  progress.classList.add('active');
  const log = document.getElementById('progressLog');
  const bar = document.getElementById('progressBar');
  const pct = document.getElementById('progressPct');
  const title = document.getElementById('progressTitle');
  log.innerHTML = '';

  try {
    const fd = new FormData();
    fd.append('book_name', bookName);
    fd.append('file', file);

    const resp = await fetch(API + '/v1/admin/books/chunk/stream', { method: 'POST', body: fd });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        try {
          const evt = line.replace(/^data:\s*/, '');
          const data = JSON.parse(evt);
          handleChunkSSE(data, log, bar, pct, title);
        } catch {}
      }
    }
  } catch(e) {
    addLog(log, 'error', 'error', e.message);
    toast('Stream error: ' + e.message, 'error');
  } finally {
    const btn = document.getElementById('uploadBtn');
    btn.disabled = false;
    btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg> Upload & Chunk`;
  }
}

function handleChunkSSE(data, log, bar, pct, title) {
  if (data.book_id && !pendingBookId) pendingBookId = data.book_id;

  if (data.total_pages && data.page_no) {
    const p = data.progress_pct || Math.round((data.page_no / data.total_pages) * 100);
    bar.style.width = p + '%';
    pct.textContent = p + '%';
  }

  if (data.total_chunks !== undefined && data.total_pages) {
    title.textContent = `Chunking: ${data.total_chunks} chunks`;
  }

  // Log events
  if (data.extraction_method)
    addLog(log, 'page', `Page ${data.page_no}/${data.total_pages}`, `${data.char_count} chars · ${data.extraction_method}`);
  else if (data.is_new_section !== undefined && data.chapter)
    addLog(log, 'heading', '§ ' + data.heading_or_subtopic, data.chapter);
  else if (data.chunk_id)
    addLog(log, 'chunk', `Chunk`, `Page ${data.page_no} · ${data.word_count} words`);
  else if (data.total_chunks && data.success) {
    addLog(log, 'done', '✓ Done', `${data.total_chunks} chunks · ${data.total_pages} pages`);
    document.getElementById('indexSection').style.display = 'block';
    document.getElementById('clearUploadBtn').style.display = 'inline-flex';
    bar.style.width = '100%';
    pct.textContent = '100%';
    loadStats(); loadBooks();
    toast('Chunking complete!', 'success');
  } else if (data.message && !data.success) {
    addLog(log, 'error', 'Error', data.message);
  }
}

function addLog(container, type, event, msg) {
  const now = new Date().toLocaleTimeString('en',{hour:'2-digit',minute:'2-digit',second:'2-digit'});
  const line = document.createElement('div');
  const isError = type === 'error';
  const isDone  = type === 'done';
  line.className = 'log-line' + (isError ? ' error' : isDone ? '' : '');
  line.innerHTML = `
    <span class="log-time">${now}</span>
    <span class="log-event">${event}</span>
    <span class="log-msg">${msg}</span>
  `;
  container.appendChild(line);
  container.scrollTop = container.scrollHeight;
}

/* ═══════════════════════════════════════════════════════════
   INDEXING (SSE)
═══════════════════════════════════════════════════════════ */
async function startIndexing() {
  if (!pendingBookId) { toast('No book to index', 'error'); return; }
  const btn = document.getElementById('indexBtn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div> Indexing…';

  const progress = document.getElementById('indexProgress');
  progress.classList.add('active');
  const log   = document.getElementById('indexProgressLog');
  const bar   = document.getElementById('indexProgressBar');
  const pct   = document.getElementById('indexProgressPct');
  const title = document.getElementById('indexProgressTitle');
  log.innerHTML = '';

  try {
    const resp = await fetch(API + '/v1/admin/books/index/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ book_id: pendingBookId }),
    });
    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        try {
          const data = JSON.parse(line.replace(/^data:\s*/, ''));
          if (data.total_chunks) title.textContent = `Indexing ${data.indexed || 0}/${data.total_chunks} chunks`;
          if (data.progress_pct !== undefined) {
            bar.style.width = data.progress_pct + '%';
            pct.textContent = data.progress_pct + '%';
          }
          if (data.batch_num)
            addLog(log, data.status || 'index', `Batch ${data.batch_num}/${data.total_batches}`,
              `${data.indexed_so_far || 0} indexed · ${data.progress_pct || 0}% · ${data.status || ''}`);
          if (data.success && data.indexed !== undefined) {
            bar.style.width = '100%';
            pct.textContent = '100%';
            addLog(log, 'done', '✓ Indexed', `${data.indexed}/${data.total_chunks} chunks`);
            toast('Indexing complete! Book is searchable.', 'success');
            loadStats(); loadBooks();
          }
          if (!data.success && data.message)
            addLog(log, 'error', 'Error', data.message);
        } catch {}
      }
    }
  } catch(e) {
    addLog(log, 'error', 'Error', e.message);
    toast('Indexing error: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg> Start Indexing (Streaming)`;
  }
}

async function startIndexingFast() {
  if (!pendingBookId) { toast('No book to index', 'error'); return; }
  const btn = document.getElementById('indexFastBtn');
  btn.disabled = true; btn.textContent = 'Indexing…';
  try {
    const r = await fetch(API + '/v1/admin/books/index', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({book_id: pendingBookId})
    });
    const d = await r.json();
    toast(`Indexed ${d.indexed}/${d.total_chunks} chunks`, 'success');
    loadStats(); loadBooks();
  } catch(e) {
    toast('Fast index error: ' + e.message, 'error');
  } finally {
    btn.disabled = false; btn.textContent = 'Fast Index';
  }
}

/* ═══════════════════════════════════════════════════════════
   BOOKS LIST
═══════════════════════════════════════════════════════════ */
async function loadBooks() {
  try {
    const r = await fetch(API + '/v1/admin/books?limit=100');
    const d = await r.json();
    renderBooks(d.books || []);
    populateGraphSelect(d.books || []);
  } catch(e) {
    document.getElementById('bookList').innerHTML =
      `<div class="empty"><div class="empty-icon">⚠️</div><div class="empty-title">Failed to load</div><div class="empty-sub">${e.message}</div></div>`;
  }
}

function renderBooks(books) {
  const el = document.getElementById('bookList');
  if (!books.length) {
    el.innerHTML = `<div class="empty"><div class="empty-icon">📚</div><div class="empty-title">No books yet</div><div class="empty-sub">Upload a PDF to get started</div></div>`;
    return;
  }
  el.innerHTML = books.map(b => `
    <div class="book-item">
      <div class="book-icon">📗</div>
      <div class="book-info">
        <div class="book-name">${b.book_name}</div>
        <div class="book-meta">
          <span>📄 ${b.total_pages || 0} pages</span>
          <span>🧩 ${(b.total_chunks || 0).toLocaleString()} chunks</span>
          <span>📁 ${b.source_file || ''}</span>
          <span class="status-badge status-${b.status || 'processing'}">${b.status || 'processing'}</span>
        </div>
      </div>
      <div class="book-actions">
        ${b.status === 'chunked' ? `<button class="btn btn-success btn-sm" onclick="indexBookFromList('${b.book_id}')">Index</button>` : ''}
        <button class="btn btn-secondary btn-sm" onclick="viewBookGraph('${b.book_id}')">Graph</button>
        <button class="btn btn-danger btn-sm" onclick="deleteBook('${b.book_id}','${b.book_name}')">Delete</button>
      </div>
    </div>
  `).join('');
}

async function indexBookFromList(bookId) {
  pendingBookId = bookId;
  showPanel('upload');
  document.getElementById('indexSection').style.display = 'block';
}

function viewBookGraph(bookId) {
  document.getElementById('graphBookSelect').value = bookId;
  showPanel('graph');
  loadGraph();
}

async function deleteBook(bookId, bookName) {
  if (!confirm(`Delete "${bookName}"? This removes all chunks, vectors, and graph nodes.`)) return;
  try {
    const r = await fetch(API + '/v1/admin/books/' + bookId, { method: 'DELETE' });
    const d = await r.json();
    toast(`Deleted "${bookName}" (${d.chunks_deleted} chunks)`, 'success');
    loadBooks(); loadStats();
  } catch(e) {
    toast('Delete failed: ' + e.message, 'error');
  }
}

/* ═══════════════════════════════════════════════════════════
   CHAT
═══════════════════════════════════════════════════════════ */
function newThread() {
  currentThread = 'thread_' + Date.now();
  document.getElementById('messages').innerHTML = `
    <div style="text-align:center;padding:40px 20px;color:var(--text2)">
      <div style="font-size:32px;margin-bottom:12px">📚</div>
      <div style="font-size:14px;font-weight:600;color:var(--text);margin-bottom:6px">New conversation</div>
      <div style="font-size:12px">Ask anything from your knowledge base</div>
    </div>`;
  document.getElementById('routeBadge').style.display = 'none';
  loadThreads();
}

async function loadThreads() {
  if (!currentUser.user_id) return;
  try {
    const r = await fetch(API + '/v1/chat/threads/' + currentUser.user_id);
    const d = await r.json();
    renderThreads(d.threads || []);
  } catch {}
}

function renderThreads(threads) {
  const el = document.getElementById('threadList');
  if (!threads.length) {
    el.innerHTML = `<div class="empty" style="padding:30px 10px"><div style="font-size:20px;margin-bottom:6px">💬</div><div style="font-size:11px;color:var(--text3)">No conversations yet</div></div>`;
    return;
  }
  el.innerHTML = threads.map(t => `
    <div class="thread-item ${t.thread_id === currentThread ? 'active' : ''}" onclick="loadThread('${t.thread_id}')">
      <div class="thread-preview">${escHtml(t.last_msg || 'New conversation')}</div>
      <div class="thread-time">${formatTime(t.last_time)}</div>
    </div>
  `).join('');
}

async function loadThread(threadId) {
  currentThread = threadId;
  loadThreads();
  try {
    const r = await fetch(API + '/v1/chat/history/' + threadId + '?limit=50');
    const d = await r.json();
    const msgs = document.getElementById('messages');
    msgs.innerHTML = '';
    for (const m of (d.messages || [])) {
      if (m.role === 'user')
        appendUserMsg(m.content, m.created_at);
      else
        appendAssistantMsg(m.content, [], m.metadata?.route, m.created_at);
    }
    msgs.scrollTop = msgs.scrollHeight;
  } catch(e) {
    toast('Failed to load thread: ' + e.message, 'error');
  }
}

function handleChatKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

async function sendMessage() {
  const input = document.getElementById('chatInput');
  const query = input.value.trim();
  if (!query || chatSending) return;

  chatSending = true;
  input.value = '';
  input.style.height = 'auto';
  document.getElementById('sendBtn').disabled = true;

  appendUserMsg(query);
  const typingId = appendTyping();

  try {
    const r = await fetch(API + '/v1/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        user_id:   currentUser.user_id,
        thread_id: currentThread,
      }),
    });
    const d = await r.json();
    removeTyping(typingId);

    // Update route badge
    const badge = document.getElementById('routeBadge');
    badge.style.display = 'block';
    badge.className = 'chat-route-badge ' + (d.route === 'rag_search' ? 'route-rag' : 'route-chat');
    badge.textContent = d.route === 'rag_search' ? '🔍 RAG Search' : '💬 Chat';

    appendAssistantMsg(d.answer || 'No response', d.references || [], d.route);
    loadThreads();

    // Refresh user name if returned
    if (d.user_name && d.user_name !== currentUser.name) {
      currentUser.name = d.user_name;
      localStorage.setItem('gr_user', JSON.stringify(currentUser));
      updateUserUI();
    }
  } catch(e) {
    removeTyping(typingId);
    appendAssistantMsg('⚠️ Error: ' + e.message, [], 'error');
  } finally {
    chatSending = false;
    document.getElementById('sendBtn').disabled = false;
    input.focus();
  }
}

function appendUserMsg(text, time) {
  const msgs = document.getElementById('messages');
  const el = document.createElement('div');
  el.className = 'msg user';
  el.innerHTML = `
    <div class="msg-body">
      <div class="msg-bubble">${escHtml(text)}</div>
      <div class="msg-time">${formatTime(time)}</div>
    </div>
    <div class="msg-avatar">${(currentUser.name || 'U').charAt(0).toUpperCase()}</div>
  `;
  msgs.appendChild(el);
  msgs.scrollTop = msgs.scrollHeight;
}

function appendAssistantMsg(text, refs, route, time) {
  const msgs = document.getElementById('messages');
  const el = document.createElement('div');
  el.className = 'msg assistant';

  const refsHtml = refs && refs.length ? `
    <div class="references">
      <div class="ref-header">📎 Sources (${refs.length})</div>
      ${refs.map((r,i) => `
        <div class="ref-chip" onclick="showRef(${JSON.stringify(r).replace(/"/g,'&quot;')})">
          <span style="color:var(--text3);font-weight:700;flex-shrink:0">${i+1}</span>
          <span class="ref-book">${escHtml(r.book_name || '')} · ${escHtml(r.chapter || '')}</span>
          <span class="ref-page">p.${r.page_no}</span>
          <span class="ref-score">${Math.round((r.hybrid_score||0)*100)}%</span>
        </div>
      `).join('')}
    </div>
  ` : '';

  const parsed = typeof marked !== 'undefined' ? marked.parse(text || '') : escHtml(text);

  el.innerHTML = `
    <div class="msg-avatar">G</div>
    <div class="msg-body">
      <div class="msg-bubble">${parsed}${refsHtml}</div>
      <div class="msg-time">${formatTime(time)}</div>
    </div>
  `;
  msgs.appendChild(el);
  msgs.scrollTop = msgs.scrollHeight;
}

function appendTyping() {
  const msgs = document.getElementById('messages');
  const el = document.createElement('div');
  const id = 'typing_' + Date.now();
  el.id = id;
  el.className = 'msg assistant';
  el.innerHTML = `
    <div class="msg-avatar">G</div>
    <div class="msg-body">
      <div class="msg-bubble">
        <div class="typing-indicator">
          <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
        </div>
      </div>
    </div>`;
  msgs.appendChild(el);
  msgs.scrollTop = msgs.scrollHeight;
  return id;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function showRef(ref) {
  document.getElementById('refModalContent').innerHTML = `
    <div style="display:flex;flex-direction:column;gap:10px">
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <span class="status-badge status-indexed">${escHtml(ref.book_name || '')}</span>
        <span class="status-badge" style="background:rgba(99,102,241,.1);color:#8b5cf6;border-color:rgba(99,102,241,.2)">${escHtml(ref.chapter || '')}</span>
        <span class="status-badge status-processing">Page ${ref.page_no}</span>
      </div>
      <div style="font-size:12px;color:var(--text2);font-weight:600">${escHtml(ref.heading || '')}</div>
      <div style="background:var(--surface2);border:1px solid var(--border);border-radius:var(--r-sm);padding:14px;font-size:12.5px;line-height:1.7;color:var(--text)">${escHtml(ref.text_preview || '')}</div>
      <div style="display:flex;gap:12px;font-size:11px;color:var(--text2)">
        <span>Vector: ${Math.round((ref.vector_score||0)*100)}%</span>
        <span>Keyword: ${Math.round((ref.keyword_score||0)*100)}%</span>
        <span>Hybrid: ${Math.round((ref.hybrid_score||0)*100)}%</span>
        <span>Source: ${ref.source || ''}</span>
      </div>
    </div>
  `;
  document.getElementById('refModal').classList.add('open');
}

function closeModal(id) {
  document.getElementById(id).classList.remove('open');
}

/* ═══════════════════════════════════════════════════════════
   GRAPH EXPLORER  (canvas-based force simulation)
═══════════════════════════════════════════════════════════ */
function populateGraphSelect(books) {
  const sel = document.getElementById('graphBookSelect');
  const cur = sel.value;
  if (!books) {
    fetch(API + '/v1/admin/books?limit=100').then(r=>r.json()).then(d=>populateGraphSelect(d.books||[]));
    return;
  }
  sel.innerHTML = '<option value="">Select a book to visualize…</option>' +
    books.map(b => `<option value="${b.book_id}">${escHtml(b.book_name)}</option>`).join('');
  if (cur) sel.value = cur;
}

async function loadGraph() {
  const bookId = document.getElementById('graphBookSelect').value;
  if (!bookId) return;
  try {
    const r = await fetch(API + '/v1/books/' + bookId + '/graph');
    const d = await r.json();
    if (!d.success) throw new Error(d.detail || 'Failed');
    graphData = d;
    document.getElementById('graphStats').textContent =
      `${d.nodes.length} nodes · ${d.edges.length} edges · Book: ${bookId}`;
    initGraphLayout();
    drawGraph();
  } catch(e) {
    toast('Graph load error: ' + e.message, 'error');
  }
}

const NODE_COLORS = {
  book:    '#f59e0b',
  chapter: '#3b82f6',
  heading: '#8b5cf6',
  chunk:   '#10b981',
  page:    '#ef4444',
};
const NODE_RADIUS = { book: 22, chapter: 16, heading: 12, chunk: 7, page: 10 };

function initGraphLayout() {
  const canvas = document.getElementById('graphCanvas');
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  graphNodePositions = {};

  graphData.nodes.forEach((n, i) => {
    const angle  = (i / graphData.nodes.length) * Math.PI * 2;
    const radius = { book: 0, chapter: 140, heading: 260, chunk: 370, page: 200 }[n.type] || 300;
    graphNodePositions[n.id] = {
      x: W/2 + Math.cos(angle) * radius + (Math.random()-0.5)*40,
      y: H/2 + Math.sin(angle) * radius + (Math.random()-0.5)*40,
      vx: 0, vy: 0,
    };
  });

  graphOffset = { x: 0, y: 0 };
  graphScale  = 1;
  runForceSimulation();
}

function runForceSimulation() {
  let iter = 0;
  const MAX = 120;
  const step = () => {
    if (iter++ >= MAX) { drawGraph(); return; }
    forceStep();
    drawGraph();
    graphAnimFrame = requestAnimationFrame(step);
  };
  if (graphAnimFrame) cancelAnimationFrame(graphAnimFrame);
  graphAnimFrame = requestAnimationFrame(step);
}

function forceStep() {
  const nodes = graphData.nodes;
  const edges = graphData.edges;
  const pos   = graphNodePositions;
  const canvas = document.getElementById('graphCanvas');
  const CX = canvas.offsetWidth/2, CY = canvas.offsetHeight/2;

  // Repulsion
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i+1; j < nodes.length; j++) {
      const a = pos[nodes[i].id], b = pos[nodes[j].id];
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.sqrt(dx*dx+dy*dy) || 1;
      const force = 4000 / (dist*dist);
      const fx = (dx/dist)*force, fy = (dy/dist)*force;
      a.vx -= fx; a.vy -= fy;
      b.vx += fx; b.vy += fy;
    }
  }
  // Attraction (edges)
  const idealLen = { book:120, chapter:90, heading:70, chunk:50, page:100 };
  for (const e of edges) {
    const a = pos[e.source], b = pos[e.target];
    if (!a || !b) continue;
    const dx = b.x - a.x, dy = b.y - a.y;
    const dist = Math.sqrt(dx*dx+dy*dy) || 1;
    const ideal = idealLen[e.relation?.toLowerCase().split('_')[1]] || 80;
    const force = (dist - ideal) * 0.04;
    const fx = (dx/dist)*force, fy = (dy/dist)*force;
    a.vx += fx; a.vy += fy;
    b.vx -= fx; b.vy -= fy;
  }
  // Center gravity
  for (const n of nodes) {
    const p = pos[n.id]; if (!p) continue;
    p.vx += (CX - p.x) * 0.003;
    p.vy += (CY - p.y) * 0.003;
  }
  // Dampen + apply
  for (const n of nodes) {
    const p = pos[n.id]; if (!p) continue;
    p.vx *= 0.85; p.vy *= 0.85;
    const maxV = 8;
    p.vx = Math.max(-maxV, Math.min(maxV, p.vx));
    p.vy = Math.max(-maxV, Math.min(maxV, p.vy));
    p.x += p.vx; p.y += p.vy;
  }
}

function drawGraph() {
  const canvas = document.getElementById('graphCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  canvas.width = W; canvas.height = H;
  if (!W || !H) return;

  ctx.clearRect(0, 0, W, H);
  ctx.save();
  ctx.translate(graphOffset.x, graphOffset.y);
  ctx.scale(graphScale, graphScale);

  // Edges
  for (const e of graphData.edges) {
    const a = graphNodePositions[e.source], b = graphNodePositions[e.target];
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = 'rgba(30,42,63,0.9)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Nodes
  for (const n of graphData.nodes) {
    const p = graphNodePositions[n.id]; if (!p) continue;
    const r = NODE_RADIUS[n.type] || 8;
    const color = NODE_COLORS[n.type] || '#64748b';
    const isHover = graphHover === n.id;

    // Glow for highlighted / hover
    if (n.highlight || isHover) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, r + 6, 0, Math.PI*2);
      ctx.fillStyle = color + '30';
      ctx.fill();
    }

    // Circle
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI*2);
    ctx.fillStyle = isHover ? color : color + 'cc';
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = isHover ? 2.5 : 1.5;
    ctx.stroke();

    // Label
    if (n.type !== 'chunk' || isHover) {
      ctx.font = `${n.type === 'book' ? 600 : 400} ${n.type === 'book' ? 11 : 9}px Inter,sans-serif`;
      ctx.fillStyle = '#e2e8f0';
      ctx.textAlign = 'center';
      const label = n.label.length > 20 ? n.label.slice(0,18)+'…' : n.label;
      ctx.fillText(label, p.x, p.y + r + 13);
    }
  }
  ctx.restore();
}

// Graph interaction
const canvas = document.getElementById('graphCanvas');
canvas.addEventListener('mousedown', e => {
  graphDragging = true;
  graphDragStart = { x: e.clientX - graphOffset.x, y: e.clientY - graphOffset.y };
});
canvas.addEventListener('mousemove', e => {
  if (graphDragging) {
    graphOffset = { x: e.clientX - graphDragStart.x, y: e.clientY - graphDragStart.y };
    drawGraph();
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left - graphOffset.x) / graphScale;
  const my = (e.clientY - rect.top  - graphOffset.y) / graphScale;

  let hit = null;
  for (const n of graphData.nodes) {
    const p = graphNodePositions[n.id]; if (!p) continue;
    const r = NODE_RADIUS[n.type] || 8;
    if (Math.hypot(mx-p.x, my-p.y) <= r+2) { hit = n; break; }
  }
  if (hit !== graphHover) {
    graphHover = hit ? hit.id : null;
    drawGraph();
  }
  const tt = document.getElementById('graphTooltip');
  if (hit) {
    tt.style.display = 'block';
    tt.style.left = (e.clientX - rect.left + 12) + 'px';
    tt.style.top  = (e.clientY - rect.top  + 12) + 'px';
    tt.innerHTML  = `<strong>${escHtml(hit.label)}</strong><br/><span style="color:var(--text2);font-size:10px">${hit.type}</span>${hit.props?.text_preview ? `<br/><span style="color:var(--text3);font-size:10px">${escHtml(hit.props.text_preview.slice(0,80))}</span>` : ''}`;
  } else {
    tt.style.display = 'none';
  }
});
canvas.addEventListener('mouseup', () => graphDragging = false);
canvas.addEventListener('mouseleave', () => { graphDragging = false; document.getElementById('graphTooltip').style.display='none'; });
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const delta = e.deltaY > 0 ? 0.85 : 1.15;
  graphScale = Math.max(0.15, Math.min(4, graphScale * delta));
  drawGraph();
}, { passive: false });

function graphZoom(factor) { graphScale = Math.max(0.15, Math.min(4, graphScale * factor)); drawGraph(); }
function resetGraphView() { graphOffset={x:0,y:0}; graphScale=1; if(graphData.nodes.length) initGraphLayout(); else drawGraph(); }

/* ═══════════════════════════════════════════════════════════
   AI TOOLS
═══════════════════════════════════════════════════════════ */
async function generateMCQ() {
  const topic = document.getElementById('mcqTopic').value.trim();
  if (!topic) { toast('Enter a topic', 'error'); return; }
  const btn = document.getElementById('mcqBtn');
  btn.disabled = true; btn.textContent = 'Generating…';
  document.getElementById('mcqResult').innerHTML = '<div style="color:var(--text2);font-size:12px">⏳ Generating MCQs from knowledge base…</div>';
  try {
    const r = await fetch(API + '/v1/books/generate-mcq', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        topic,
        num_questions: parseInt(document.getElementById('mcqCount').value)||5,
        difficulty: document.getElementById('mcqDiff').value,
      })
    });
    const d = await r.json();
    renderMCQ(d);
  } catch(e) {
    document.getElementById('mcqResult').innerHTML = `<div style="color:var(--danger);font-size:12px">Error: ${e.message}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = 'Generate MCQ';
  }
}

function renderMCQ(data) {
  const qs = data.mcq?.questions || [];
  if (!qs.length) {
    document.getElementById('mcqResult').innerHTML = '<div style="color:var(--text2);font-size:12px">No questions generated.</div>';
    return;
  }
  document.getElementById('mcqResult').innerHTML = `
    <div style="font-size:11px;color:var(--text2);margin-bottom:10px">
      ${data.source === 'document_context' ? '📚' : '🧠'} Source: ${data.source} · ${data.chunks_used || 0} chunks used
    </div>
    ${qs.map((q,i) => `
      <div style="background:var(--surface2);border:1px solid var(--border);border-radius:var(--r-sm);padding:14px;margin-bottom:10px">
        <div style="font-size:12.5px;font-weight:600;margin-bottom:10px;color:var(--text)">${i+1}. ${escHtml(q.question)}</div>
        ${Object.entries(q.options||{}).map(([k,v]) => `
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;padding:7px 10px;border-radius:5px;border:1px solid var(--border);font-size:12px;cursor:pointer;transition:.15s"
            onclick="selectMCQOption(this,'${k}','${q.correct_answer}')">
            <span style="font-weight:700;color:var(--accent);min-width:18px">${k}</span>
            <span>${escHtml(v)}</span>
          </div>
        `).join('')}
        <div class="mcq-exp-${i}" style="display:none;margin-top:8px;padding:8px 10px;background:rgba(16,185,129,.07);border:1px solid rgba(16,185,129,.2);border-radius:5px;font-size:11.5px;color:var(--accent3)">
          ✓ Correct: <strong>${q.correct_answer}</strong> — ${escHtml(q.explanation||'')}
          ${q.chapter ? `<span style="color:var(--text3)"> · ${escHtml(q.chapter)} p.${q.page_no||'?'}</span>` : ''}
        </div>
      </div>
    `).join('')}
  `;
}

function selectMCQOption(el, chosen, correct) {
  const parent = el.parentElement;
  const opts = parent.querySelectorAll('[onclick^="selectMCQOption"]');
  opts.forEach(o => {
    const k = o.querySelector('span').textContent.trim();
    o.style.borderColor = k === correct ? 'var(--accent3)' : (k === chosen && chosen !== correct ? 'var(--danger)' : 'var(--border)');
    o.style.background  = k === correct ? 'rgba(16,185,129,.08)' : (k === chosen && chosen !== correct ? 'rgba(239,68,68,.08)' : '');
    o.style.pointerEvents = 'none';
  });
  // Show explanation
  const expEls = parent.querySelectorAll('[class^="mcq-exp-"]');
  expEls.forEach(e => e.style.display = 'block');
}

async function generateSummary() {
  const topic = document.getElementById('summaryTopic').value.trim();
  if (!topic) { toast('Enter a topic', 'error'); return; }
  const btn = document.getElementById('summaryBtn');
  btn.disabled = true; btn.textContent = 'Generating…';
  document.getElementById('summaryResult').innerHTML = '<div style="color:var(--text2);font-size:12px">⏳ Generating lesson summary…</div>';
  try {
    const r = await fetch(API + '/v1/books/generate-lesson-summary', {
      method:'POST',headers:{'Content-Type':'application/json'},
      body: JSON.stringify({topic})
    });
    const d = await r.json();
    renderSummary(d);
  } catch(e) {
    document.getElementById('summaryResult').innerHTML = `<div style="color:var(--danger);font-size:12px">Error: ${e.message}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = 'Generate Summary';
  }
}

function renderSummary(data) {
  const s = data.lesson?.summary;
  if (!s) { document.getElementById('summaryResult').innerHTML = '<div style="color:var(--text2)">No summary generated.</div>'; return; }
  const terms = (s.important_terms||[]).map(t=>`<span style="display:inline-flex;flex-direction:column;background:var(--surface);border:1px solid var(--border);border-radius:5px;padding:6px 10px;font-size:11px;gap:2px"><span style="font-weight:700;color:var(--text)">${escHtml(t.term)}</span><span style="color:var(--text2)">${escHtml(t.definition)}</span></span>`).join('');
  document.getElementById('summaryResult').innerHTML = `
    <div style="font-size:11px;color:var(--text2);margin-bottom:10px">${data.source==='document_context'?'📚':'🧠'} Source: ${data.source}</div>
    <div style="background:rgba(59,130,246,.05);border:1px solid rgba(59,130,246,.15);border-radius:var(--r-sm);padding:12px;margin-bottom:10px;font-size:12.5px;color:var(--text);line-height:1.6">${escHtml(s.overview||'')}</div>
    <div style="font-size:11px;font-weight:700;color:var(--text2);margin-bottom:6px;text-transform:uppercase;letter-spacing:.6px">Objectives</div>
    ${(s.learning_objectives||[]).map(o=>`<div style="font-size:12px;color:var(--text);margin-bottom:4px;display:flex;gap:6px"><span style="color:var(--accent3)">✓</span>${escHtml(o)}</div>`).join('')}
    ${s.important_terms?.length ? `<div style="font-size:11px;font-weight:700;color:var(--text2);margin:10px 0 6px;text-transform:uppercase;letter-spacing:.6px">Key Terms</div><div style="display:flex;flex-wrap:wrap;gap:6px">${terms}</div>` : ''}
    ${s.exam_tips?.length ? `<div style="font-size:11px;font-weight:700;color:var(--warn);margin:10px 0 6px;text-transform:uppercase;letter-spacing:.6px">💡 Exam Tips</div>${(s.exam_tips).map(t=>`<div style="font-size:12px;color:var(--text2);margin-bottom:4px;display:flex;gap:6px"><span style="color:var(--warn)">→</span>${escHtml(t)}</div>`).join('')}` : ''}
  `;
}

/* ═══════════════════════════════════════════════════════════
   TOAST
═══════════════════════════════════════════════════════════ */
function toast(msg, type='info') {
  const c = document.getElementById('toastContainer');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  const icons = {success:'✓',error:'✕',info:'ℹ'};
  t.innerHTML = `<span style="font-size:14px">${icons[type]||'ℹ'}</span><span>${escHtml(msg)}</span>`;
  c.appendChild(t);
  setTimeout(() => { t.style.opacity='0'; t.style.transition='opacity .3s'; setTimeout(()=>t.remove(),300); }, 3500);
}

/* ═══════════════════════════════════════════════════════════
   HELPERS
═══════════════════════════════════════════════════════════ */
function escHtml(s) {
  if (typeof s !== 'string') s = String(s||'');
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function formatTime(t) {
  if (!t) return new Date().toLocaleTimeString('en',{hour:'2-digit',minute:'2-digit'});
  const d = new Date(t);
  return d.toLocaleTimeString('en',{hour:'2-digit',minute:'2-digit'});
}

// Close modals on overlay click
document.querySelectorAll('.modal-overlay').forEach(el => {
  el.addEventListener('click', e => { if (e.target === el) el.classList.remove('open'); });
});

// Resize graph canvas on window resize
window.addEventListener('resize', () => { if (graphData.nodes.length) drawGraph(); });
</script>
</body>
</html>














    
