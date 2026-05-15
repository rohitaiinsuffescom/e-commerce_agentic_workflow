# import os
# import re
# import uuid
# import json
# import asyncio
# import tempfile
# from datetime import datetime, timezone
# from typing import List, Dict, Any, Optional
# from pydantic import BaseModel
# from pinecone import Pinecone
# from neo4j import GraphDatabase
# import fitz
# import pytesseract
# from PIL import Image
# import math

# from pymongo import MongoClient, ASCENDING, UpdateOne
# from pymongo.errors import DuplicateKeyError

# from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# from fastapi.responses import JSONResponse, StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# from openai import OpenAI


# # =========================
# # APP
# # =========================

# app = FastAPI(title="GraphRAG Document API", version="3.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # =========================
# # CONFIG  (prefer env vars; fall back to defaults for local dev)
# # =========================

# MIN_TEXT_PER_PAGE    = int(os.getenv("MIN_TEXT_PER_PAGE", 40))
# MIN_PARAGRAPH_CHARS  = int(os.getenv("MIN_PARAGRAPH_CHARS", 60))
# CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE", 1200))
# CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP", 180))

# MONGO_URI      = os.getenv("MONGO_URI",      "mongodb://localhost:27017")
# MONGO_DB_NAME  = os.getenv("MONGO_DB_NAME",  "data_edtech")
# CHUNKS_COLLECTION = "chunks"
# BOOKS_COLLECTION  = "books"

# OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY",       "sk......")
# OPENAI_MODEL         = os.getenv("OPENAI_MODEL",         "gpt-4.1-mini")
# OPENAI_EMBED_MODEL   = os.getenv("OPENAI_EMBED_MODEL",   "text-embedding-3-small")
# EMBEDDING_DIMENSION  = 1536

# PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY",     "pcs....")
# PINECONE_INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME",  "aieducation")

# NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
# NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "secretgraph")

# BATCH_SIZE          = int(os.getenv("BATCH_SIZE",          50))
# VECTOR_TOP_K        = int(os.getenv("VECTOR_TOP_K",        25))
# KEYWORD_TOP_K       = int(os.getenv("KEYWORD_TOP_K",       25))
# MIN_FINAL_CONFIDENCE = float(os.getenv("MIN_FINAL_CONFIDENCE", 0.55))

# # =========================
# # DB
# # =========================

# mongo_client = MongoClient(MONGO_URI)
# mongo_db     = mongo_client[MONGO_DB_NAME]
# chunks_col   = mongo_db[CHUNKS_COLLECTION]
# books_col    = mongo_db[BOOKS_COLLECTION]

# # Standard indexes
# chunks_col.create_index([("chunk_id", ASCENDING)], unique=True)
# chunks_col.create_index([("book_id",  ASCENDING)])
# chunks_col.create_index([("page_no",  ASCENDING)])
# chunks_col.create_index([("chapter",  ASCENDING)])
# chunks_col.create_index([("heading_or_subtopic", ASCENDING)])

# # ── FIX: text index required for $text queries ──────────────────────────────
# # Silently ignore if it already exists (IndexKeySpecsConflict / similar)
# try:
#     chunks_col.create_index(
#         [
#             ("text",               "text"),
#             ("chapter",            "text"),
#             ("heading_or_subtopic","text"),
#         ],
#         name="chunks_fulltext",
#         default_language="english",
#     )
# except Exception:
#     pass  # index already exists or another harmless conflict
# # ─────────────────────────────────────────────────────────────────────────────

# books_col.create_index([("book_id", ASCENDING)], unique=True)

# # =========================
# # OPENAI + PINECONE + NEO4J
# # =========================

# if not OPENAI_API_KEY:
#     raise RuntimeError("OPENAI_API_KEY env var is required")

# openai_client  = OpenAI(api_key=OPENAI_API_KEY)

# if not PINECONE_API_KEY:
#     raise RuntimeError("PINECONE_API_KEY env var is required")

# pc             = Pinecone(api_key=PINECONE_API_KEY)
# pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# neo4j_driver   = GraphDatabase.driver(
#     NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
# )

# # =========================
# # UTILS
# # =========================

# def now_utc() -> datetime:
#     return datetime.now(timezone.utc)


# def sse_event(event: str, data: dict) -> str:
#     return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


# def normalize_text(text: str) -> str:
#     if not text:
#         return ""
#     text = text.replace("\x00", " ")
#     text = re.sub(r"[ \t]+", " ", text)
#     text = re.sub(r"\n{3,}", "\n\n", text)
#     text = re.sub(r" *\n *", "\n", text)
#     return text.strip()


# def clean_paragraph(text: str) -> str:
#     text = normalize_text(text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()


# def remove_noise_lines(text: str) -> str:
#     cleaned = []
#     for line in text.split("\n"):
#         line = line.strip()
#         if not line:
#             cleaned.append("")
#             continue
#         if re.fullmatch(r"\d+", line):
#             continue
#         if re.match(r"^[A-Z\s]+ \d+$", line):
#             continue
#         cleaned.append(line)
#     return "\n".join(cleaned)


# def split_into_blocks(text: str) -> List[str]:
#     text = normalize_text(text)
#     text = remove_noise_lines(text)
#     blocks = re.split(r"\n\s*\n", text)
#     return [b.strip() for b in blocks if b.strip()]


# # =========================
# # OCR
# # =========================

# def ocr_page(page: fitz.Page) -> str:
#     try:
#         pix = page.get_pixmap(dpi=300)
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         return normalize_text(pytesseract.image_to_string(img))
#     except Exception:
#         return ""


# def extract_page_text(page: fitz.Page) -> Dict[str, Any]:
#     try:
#         normal_text = normalize_text(page.get_text("text"))
#     except Exception:
#         normal_text = ""

#     if len(normal_text) >= MIN_TEXT_PER_PAGE:
#         return {"text": normal_text, "extraction_method": "pymupdf"}

#     ocr_text = ocr_page(page)
#     return {
#         "text": ocr_text,
#         "extraction_method": "ocr_fallback" if ocr_text else "failed",
#     }


# # =========================
# # HEADING DETECTION
# # =========================

# def looks_like_heading(line: str) -> bool:
#     line = line.strip()
#     if not line:
#         return False
#     words = line.split()
#     if len(words) > 14:
#         return False
#     if re.fullmatch(r"\d+", line):
#         return False
#     if re.match(r"^(chapter|unit|lesson|section|topic|part)\s+[\dA-Za-z]+", line, re.I):
#         return True
#     if re.match(r"^\d+(\.\d+)*[\).]?\s+[A-Za-z]", line):
#         return True
#     if line.isupper() and 2 <= len(words) <= 10:
#         return True
#     title_words = sum(1 for w in words if w[:1].isupper())
#     if 2 <= len(words) <= 10:
#         if title_words >= len(words) - 1:
#             if not re.search(r"[.!?]$", line):
#                 return True
#     return False


# def clean_topic_name(topic: str) -> str:
#     topic = clean_paragraph(topic)
#     topic = re.sub(r"^(chapter|unit|lesson|section)\s*\d*[:.-]?\s*", "", topic, flags=re.I)
#     topic = topic.strip(" :-–—.")
#     return topic[:90] if topic else "General Content"


# # =========================
# # AI TOPIC GENERATOR
# # =========================

# def fallback_topic_from_text(text: str) -> str:
#     words = clean_paragraph(text).split()
#     return clean_topic_name(" ".join(words[:8])) if words else "General Content"


# def ai_generate_topic_title(text: str, previous_topic: Optional[str] = None) -> Dict[str, Any]:
#     try:
#         sample = clean_paragraph(text[:1800])
#         prompt = f"""You are an intelligent document understanding engine.

# Task: Extract or generate the BEST section/topic title.

# Rules:
# - If heading exists → return heading (max 8 words).
# - If no heading → generate a short meaningful topic.
# - If paragraph continues previous topic, keep same topic.

# Previous Topic: {previous_topic or "None"}

# Return ONLY valid JSON:
# {{"chapter":"string","heading_or_subtopic":"string","is_new_section":true/false}}

# Content:
# {sample}"""

#         resp = openai_client.chat.completions.create(
#             model=OPENAI_MODEL,
#             messages=[
#                 {"role": "system", "content": "You generate clean document section names."},
#                 {"role": "user",   "content": prompt},
#             ],
#             temperature=0,
#             response_format={"type": "json_object"},
#         )
#         data = json.loads(resp.choices[0].message.content)
#         return {
#             "chapter":             clean_topic_name(data.get("chapter") or "General Document"),
#             "heading_or_subtopic": clean_topic_name(data.get("heading_or_subtopic") or fallback_topic_from_text(text)),
#             "is_new_section":      bool(data.get("is_new_section", False)),
#             "detection_method":    "ai_generated_topic",
#         }
#     except Exception:
#         return {
#             "chapter":             "General Document",
#             "heading_or_subtopic": fallback_topic_from_text(text),
#             "is_new_section":      False,
#             "detection_method":    "fallback_summary_title",
#         }


# # =========================
# # SECTION DETECTION
# # =========================

# def detect_document_section(block: str, current_chapter: Optional[str], current_heading: Optional[str]) -> Dict[str, Any]:
#     lines = [l.strip() for l in block.split("\n") if l.strip()]

#     if lines and looks_like_heading(lines[0]):
#         heading = clean_topic_name(lines[0])
#         chapter = current_chapter or "General Document"
#         if re.match(r"^(chapter|unit|lesson|section|part)", lines[0], re.I):
#             chapter = heading
#         return {
#             "chapter":             chapter,
#             "heading_or_subtopic": heading,
#             "is_new_section":      True,
#             "detection_method":    "real_heading_detected",
#         }

#     generated = ai_generate_topic_title(block, current_heading)

#     if current_heading and not generated.get("is_new_section"):
#         return {
#             "chapter":             current_chapter or generated["chapter"],
#             "heading_or_subtopic": current_heading,
#             "is_new_section":      False,
#             "detection_method":    generated["detection_method"],
#         }

#     return generated


# # =========================
# # SENTENCE-AWARE CHUNKING
# # =========================

# def sentence_aware_chunking(text: str, max_chars: int = CHUNK_SIZE) -> List[str]:
#     text = clean_paragraph(text)
#     if len(text) <= max_chars:
#         return [text]

#     sentences = re.split(r"(?<=[.!?])\s+", text)
#     chunks, current = [], ""

#     for sentence in sentences:
#         if len(current) + len(sentence) <= max_chars:
#             current += " " + sentence
#         else:
#             if current.strip():
#                 chunks.append(current.strip())
#             current = sentence

#     if current.strip():
#         chunks.append(current.strip())

#     return chunks or [text]


# # =========================
# # MONGO HELPERS
# # =========================

# def safe_insert_chunk(chunk: dict) -> bool:
#     try:
#         chunks_col.insert_one(chunk)
#         return True
#     except DuplicateKeyError:
#         return False
#     except Exception:
#         return False


# def save_book_meta(book_doc: dict) -> None:
#     try:
#         books_col.update_one(
#             {"book_id": book_doc["book_id"]},
#             {"$set": book_doc},
#             upsert=True,
#         )
#     except Exception:
#         pass


# # =========================
# # CHUNK BUILDER
# # =========================

# def build_chunks(
#     book_id: str,
#     book_name: str,
#     filename: str,
#     pages: List[Dict[str, Any]],
#     save_to_mongo: bool = True,
# ) -> List[dict]:
#     chunks = []
#     current_chapter = None
#     current_heading = None
#     global_paragraph_no = 0

#     for page_data in pages:
#         page_no           = page_data["page_no"]
#         page_text         = page_data["text"]
#         extraction_method = page_data["extraction_method"]
#         blocks            = split_into_blocks(page_text)
#         page_paragraph_no = 0

#         for block in blocks:
#             lines = [l.strip() for l in block.split("\n") if l.strip()]
#             if not lines:
#                 continue
#             paragraph_text = clean_paragraph(" ".join(lines))
#             if len(paragraph_text) < MIN_PARAGRAPH_CHARS:
#                 continue

#             detected = detect_document_section(block, current_chapter, current_heading)
#             current_chapter = detected["chapter"] or current_chapter or "General Document"
#             current_heading = detected["heading_or_subtopic"] or current_heading or "General Content"

#             page_paragraph_no   += 1
#             global_paragraph_no += 1

#             for idx, chunk_text in enumerate(sentence_aware_chunking(paragraph_text), start=1):
#                 chunk = {
#                     "chunk_id":                     str(uuid.uuid4()),
#                     "book_id":                      book_id,
#                     "book_name":                    book_name,
#                     "source_file":                  filename,
#                     "page_no":                      page_no,
#                     "page_paragraph_no":            page_paragraph_no,
#                     "global_paragraph_no":          global_paragraph_no,
#                     "chapter":                      current_chapter,
#                     "heading_or_subtopic":          current_heading,
#                     "topic_detection_method":       detected["detection_method"],
#                     "is_new_section":               detected.get("is_new_section", False),
#                     "chunk_index_inside_paragraph": idx,
#                     "total_chunks_from_paragraph":  len(sentence_aware_chunking(paragraph_text)),
#                     "extraction_method":            extraction_method,
#                     "char_count":                   len(chunk_text),
#                     "word_count":                   len(chunk_text.split()),
#                     "text_preview":                 chunk_text[:250],
#                     "text":                         chunk_text,
#                     "status":                       "chunked",
#                     "created_at":                   now_utc(),
#                     "updated_at":                   now_utc(),
#                 }
#                 if save_to_mongo:
#                     safe_insert_chunk(chunk)
#                 chunks.append(chunk)

#     return chunks


# # =========================
# # EMBEDDINGS
# # =========================

# def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
#     """Always receives a *list* of strings; returns a list of embedding vectors."""
#     if not texts:
#         return []
#     response = openai_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
#     return [item.embedding for item in response.data]


# def build_embedding_text(chunk: dict) -> str:
#     return (
#         f"Book: {chunk.get('book_name', '')}\n"
#         f"Chapter: {chunk.get('chapter', 'General Document')}\n"
#         f"Heading: {chunk.get('heading_or_subtopic', 'General Content')}\n"
#         f"Page: {chunk.get('page_no', '')}\n\n"
#         f"Content:\n{chunk.get('text', '')}"
#     ).strip()


# # =========================
# # PINECONE UPSERT
# # =========================

# def save_chunks_to_pinecone_batch(chunks: List[dict]) -> None:
#     texts   = [build_embedding_text(c) for c in chunks]
#     vectors = create_embeddings_batch(texts)          # ← list of strings, returns list of vectors

#     pinecone_vectors = []
#     for chunk, vector in zip(chunks, vectors):
#         pinecone_vectors.append({
#             "id":     chunk["chunk_id"],
#             "values": vector,
#             "metadata": {
#                 "chunk_id":              chunk["chunk_id"],
#                 "book_id":               chunk["book_id"],
#                 "book_name":             chunk.get("book_name", ""),
#                 "source_file":           chunk.get("source_file", ""),
#                 "page_no":               chunk.get("page_no", 0),
#                 "chapter":               chunk.get("chapter", "General Document"),
#                 "heading":               chunk.get("heading_or_subtopic", "General Content"),
#                 "text":                  chunk.get("text", "")[:3000],
#                 "text_preview":          chunk.get("text_preview", ""),
#                 "topic_detection_method":chunk.get("topic_detection_method", ""),
#             },
#         })

#     pinecone_index.upsert(
#         vectors=pinecone_vectors,
#         namespace=f"book_{chunks[0]['book_id']}",
#     )


# # =========================
# # NEO4J UPSERT
# # =========================

# def save_chunks_to_neo4j_batch(chunks: List[dict]) -> None:
#     query = """
#     UNWIND $chunks AS chunk

#     MERGE (b:Book {book_id: chunk.book_id})
#     SET b.name = chunk.book_name, b.source_file = chunk.source_file

#     MERGE (c:Chapter {book_id: chunk.book_id, name: chunk.chapter})
#     MERGE (h:Heading {book_id: chunk.book_id, chapter: chunk.chapter, name: chunk.heading})
#     MERGE (p:Page    {book_id: chunk.book_id, page_no: chunk.page_no})

#     MERGE (ch:Chunk {chunk_id: chunk.chunk_id})
#     SET ch.text_preview       = chunk.text_preview,
#         ch.page_no            = chunk.page_no,
#         ch.word_count         = chunk.word_count,
#         ch.global_paragraph_no= chunk.global_paragraph_no

#     MERGE (b)-[:HAS_CHAPTER]->(c)
#     MERGE (c)-[:HAS_HEADING]->(h)
#     MERGE (h)-[:HAS_CHUNK]  ->(ch)
#     MERGE (b)-[:HAS_PAGE]   ->(p)
#     MERGE (p)-[:HAS_CHUNK]  ->(ch)
#     """
#     rows = [
#         {
#             "chunk_id":            c["chunk_id"],
#             "book_id":             c["book_id"],
#             "book_name":           c.get("book_name", ""),
#             "source_file":         c.get("source_file", ""),
#             "chapter":             c.get("chapter", "General Document"),
#             "heading":             c.get("heading_or_subtopic", "General Content"),
#             "page_no":             c.get("page_no", 0),
#             "text_preview":        c.get("text_preview", ""),
#             "word_count":          c.get("word_count", 0),
#             "global_paragraph_no": c.get("global_paragraph_no", 0),
#         }
#         for c in chunks
#     ]
#     with neo4j_driver.session() as session:
#         session.run(query, chunks=rows)


# def update_mongo_index_status_batch(chunks: List[dict]) -> None:
#     ops = [
#         UpdateOne(
#             {"chunk_id": c["chunk_id"]},
#             {"$set": {"pinecone_indexed": True, "neo4j_indexed": True, "indexed_at": now_utc()}},
#         )
#         for c in chunks
#     ]
#     if ops:
#         chunks_col.bulk_write(ops, ordered=False)


# # =========================
# # OPENAI JSON HELPER
# # =========================

# def call_openai_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
#     resp = openai_client.chat.completions.create(
#         model=OPENAI_MODEL,
#         temperature=0,
#         response_format={"type": "json_object"},
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user",   "content": user_prompt},
#         ],
#     )
#     return json.loads(resp.choices[0].message.content)


# # =========================
# # SEARCH HELPERS
# # =========================

# def openai_query_planner(user_query: str, book_name: str = "") -> Dict[str, Any]:
#     system = """You are a production-level educational search query planner.
# 1. Understand the user's learning intent.
# 2. Fix spelling mistakes naturally.
# 3. Expand the query into semantic and keyword search queries.
# 4. Detect intent: definition|concept|formula|example|exercise|explanation|unknown.
# Return JSON only."""

#     user = f"""User query: {user_query}
# Book name: {book_name}

# Return JSON:
# {{
#   "corrected_query": "",
#   "intent": "definition|concept|formula|example|exercise|explanation|unknown",
#   "main_topic": "",
#   "semantic_queries": [],
#   "keyword_queries": [],
#   "must_have_terms": [],
#   "avoid_terms": []
# }}"""
#     return call_openai_json(system, user)


# def vector_search(namespace: str, queries: List[str], top_k: int = VECTOR_TOP_K) -> List[Dict[str, Any]]:
#     all_matches: Dict[str, Dict] = {}

#     for q in queries:
#         # ── FIX: create_embeddings_batch expects a *list*; take first element ──
#         query_vector = create_embeddings_batch([q])[0]

#         result = pinecone_index.query(
#             namespace=namespace,
#             vector=query_vector,
#             top_k=top_k,
#             include_metadata=True,
#         )
#         for match in result.get("matches", []):
#             cid   = match["id"]
#             score = float(match.get("score", 0))
#             if cid not in all_matches or score > all_matches[cid]["vector_score"]:
#                 all_matches[cid] = {"chunk_id": cid, "vector_score": score}

#     return list(all_matches.values())


# def keyword_search(book_id: str, keyword_queries: List[str], top_k: int = KEYWORD_TOP_K) -> List[Dict[str, Any]]:
#     """
#     Uses MongoDB $text index.
#     Index must exist on { text, chapter, heading_or_subtopic } — created at startup.
#     """
#     results: Dict[str, Dict] = {}

#     for q in keyword_queries:
#         if not q or not q.strip():
#             continue
#         try:
#             cursor = (
#                 chunks_col.find(
#                     {"book_id": book_id, "$text": {"$search": q}},
#                     {"_id": 0, "chunk_id": 1, "keyword_score": {"$meta": "textScore"}},
#                 )
#                 .sort([("keyword_score", {"$meta": "textScore"})])
#                 .limit(top_k)
#             )
#             for doc in cursor:
#                 cid   = doc["chunk_id"]
#                 score = float(doc.get("keyword_score", 0))
#                 if cid not in results or score > results[cid]["keyword_score"]:
#                     results[cid] = {"chunk_id": cid, "keyword_score": score}
#         except Exception as e:
#             # Log but don't crash — fall back gracefully
#             print(f"[keyword_search] error for query '{q}': {e}")

#     return list(results.values())


# def normalize_scores(items: List[Dict], score_key: str) -> Dict[str, float]:
#     if not items:
#         return {}
#     scores   = [x.get(score_key, 0) for x in items]
#     min_s, max_s = min(scores), max(scores)
#     out = {}
#     for item in items:
#         s = item.get(score_key, 0)
#         norm = 1.0 if max_s == min_s else (s - min_s) / (max_s - min_s)
#         out[item["chunk_id"]] = round(norm, 4)
#     return out


# def merge_hybrid_results(vector_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
#     v_norm = normalize_scores(vector_results,  "vector_score")
#     k_norm = normalize_scores(keyword_results, "keyword_score")
#     all_ids = set(v_norm) | set(k_norm)

#     merged = [
#         {
#             "chunk_id":      cid,
#             "vector_score":  v_norm.get(cid, 0),
#             "keyword_score": k_norm.get(cid, 0),
#             "hybrid_score":  round(0.65 * v_norm.get(cid, 0) + 0.35 * k_norm.get(cid, 0), 4),
#         }
#         for cid in all_ids
#     ]
#     merged.sort(key=lambda x: x["hybrid_score"], reverse=True)
#     return merged


# def get_related_chunks_from_neo4j(chunk_ids: List[str], limit: int = 20) -> List[str]:
#     if not chunk_ids:
#         return []
#     query = """
#     MATCH (ch:Chunk) WHERE ch.chunk_id IN $chunk_ids
#     OPTIONAL MATCH (h:Heading)-[:HAS_CHUNK]->(ch)
#     OPTIONAL MATCH (h)-[:HAS_CHUNK]->(related:Chunk)
#     WITH DISTINCT related.chunk_id AS chunk_id
#     WHERE chunk_id IS NOT NULL
#     RETURN chunk_id LIMIT $limit
#     """
#     related: set = set()
#     try:
#         with neo4j_driver.session() as session:
#             result = session.run(query, chunk_ids=chunk_ids, limit=limit)
#             for record in result:
#                 related.add(record["chunk_id"])
#     except Exception as e:
#         print(f"[neo4j] get_related_chunks error: {e}")
#     return list(related)


# def openai_rerank(query_plan: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
#     compact = [
#         {
#             "rank_id":     i,
#             "chunk_id":    c["chunk_id"],
#             "chapter":     c.get("chapter"),
#             "heading":     c.get("heading_or_subtopic"),
#             "page_no":     c.get("page_no"),
#             "text":        c.get("text", "")[:1200],
#             "hybrid_score":c.get("hybrid_score", 0),
#             "source":      c.get("source"),
#         }
#         for i, c in enumerate(chunks)
#     ]

#     system = """You are a strict educational RAG reranker.
# Select only chunks truly related to the query.
# Return confidence 0–1.
# JSON only."""

#     user = f"""Query plan:
# {json.dumps(query_plan, indent=2)}

# Candidate chunks:
# {json.dumps(compact, indent=2)}

# Return JSON:
# {{
#   "confidence": 0.0,
#   "reason": "",
#   "selected_chunk_ids": [],
#   "rejected_chunk_ids": []
# }}"""

#     return call_openai_json(system, user)


# # =========================
# # PYDANTIC MODELS
# # =========================

# class IndexBookRequest(BaseModel):
#     book_id: str


# class SearchRequest(BaseModel):
#     book_id:   str
#     query:     str
#     top_k:     int = 5
#     book_name: str = ""


# # =========================
# # API — CHUNK (sync)
# # =========================

# @app.post("/v1/books/chunk")
# async def upload_and_chunk_book(
#     book_name: str      = Form(...),
#     file:      UploadFile = File(...),
# ):
#     if not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported")

#     book_id   = str(uuid.uuid4())
#     temp_path = None

#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#             tmp.write(await file.read())
#             temp_path = tmp.name

#         pdf = fitz.open(temp_path)
#         save_book_meta({
#             "book_id": book_id, "book_name": book_name,
#             "source_file": file.filename, "status": "processing",
#             "created_at": now_utc(), "updated_at": now_utc(),
#         })

#         pages = []
#         for i in range(len(pdf)):
#             page      = pdf.load_page(i)
#             extracted = extract_page_text(page)
#             pages.append({
#                 "page_no":          i + 1,
#                 "text":             extracted["text"],
#                 "extraction_method":extracted["extraction_method"],
#                 "char_count":       len(extracted["text"]),
#             })

#         chunks = build_chunks(book_id=book_id, book_name=book_name,
#                               filename=file.filename, pages=pages, save_to_mongo=True)

#         save_book_meta({
#             "book_id": book_id, "book_name": book_name,
#             "source_file": file.filename, "status": "chunked",
#             "total_pages": len(pdf), "total_chunks": len(chunks),
#             "updated_at": now_utc(),
#         })

#         return JSONResponse({
#             "success": True, "book_id": book_id, "book_name": book_name,
#             "total_pages": len(pdf), "total_chunks": len(chunks),
#             "database": MONGO_DB_NAME, "collection": CHUNKS_COLLECTION,
#             "chunks": chunks,
#         })

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         if temp_path and os.path.exists(temp_path):
#             os.remove(temp_path)


# # =========================
# # API — CHUNK (stream)
# # =========================

# @app.post("/v1/books/chunk/stream")
# async def upload_and_chunk_book_stream(
#     book_name: str      = Form(...),
#     file:      UploadFile = File(...),
# ):
#     if not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported")

#     book_id   = str(uuid.uuid4())
#     temp_path = None

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(await file.read())
#         temp_path = tmp.name

#     async def stream_generator():
#         total_chunks = 0
#         try:
#             yield sse_event("start", {
#                 "success": True, "book_id": book_id,
#                 "book_name": book_name, "source_file": file.filename,
#             })

#             pdf         = fitz.open(temp_path)
#             total_pages = len(pdf)
#             yield sse_event("pdf_info", {"book_id": book_id, "total_pages": total_pages})

#             current_chapter     = None
#             current_heading     = None
#             global_paragraph_no = 0

#             for i in range(total_pages):
#                 page_no   = i + 1
#                 page      = pdf.load_page(i)
#                 extracted = extract_page_text(page)
#                 page_text = extracted["text"]
#                 exm       = extracted["extraction_method"]

#                 yield sse_event("page_extracted", {
#                     "page_no": page_no, "char_count": len(page_text),
#                     "extraction_method": exm,
#                 })

#                 blocks            = split_into_blocks(page_text)
#                 page_paragraph_no = 0

#                 for block in blocks:
#                     lines = [l.strip() for l in block.split("\n") if l.strip()]
#                     if not lines:
#                         continue
#                     paragraph_text = clean_paragraph(" ".join(lines))
#                     if len(paragraph_text) < MIN_PARAGRAPH_CHARS:
#                         continue

#                     detected = detect_document_section(block, current_chapter, current_heading)
#                     current_chapter = detected["chapter"] or current_chapter or "General Document"
#                     current_heading = detected["heading_or_subtopic"] or current_heading or "General Content"

#                     yield sse_event("heading_detected", {
#                         "page_no": page_no, "chapter": current_chapter,
#                         "heading_or_subtopic": current_heading,
#                         "detection_method": detected["detection_method"],
#                     })

#                     page_paragraph_no   += 1
#                     global_paragraph_no += 1
#                     split_texts = sentence_aware_chunking(paragraph_text)

#                     for idx, chunk_text in enumerate(split_texts, start=1):
#                         total_chunks += 1
#                         chunk = {
#                             "chunk_id":                     str(uuid.uuid4()),
#                             "book_id":                      book_id,
#                             "book_name":                    book_name,
#                             "source_file":                  file.filename,
#                             "page_no":                      page_no,
#                             "page_paragraph_no":            page_paragraph_no,
#                             "global_paragraph_no":          global_paragraph_no,
#                             "chapter":                      current_chapter,
#                             "heading_or_subtopic":          current_heading,
#                             "topic_detection_method":       detected["detection_method"],
#                             "is_new_section":               detected.get("is_new_section", False),
#                             "chunk_index_inside_paragraph": idx,
#                             "total_chunks_from_paragraph":  len(split_texts),
#                             "extraction_method":            exm,
#                             "char_count":                   len(chunk_text),
#                             "word_count":                   len(chunk_text.split()),
#                             "text_preview":                 chunk_text[:250],
#                             "text":                         chunk_text,
#                             "status":                       "chunked",
#                             "created_at":                   now_utc(),
#                             "updated_at":                   now_utc(),
#                         }
#                         safe_insert_chunk(chunk)
#                         yield sse_event("chunk", chunk)
#                         await asyncio.sleep(0)

#                 yield sse_event("page_done", {
#                     "page_no": page_no, "total_pages": total_pages,
#                     "total_chunks_so_far": total_chunks,
#                 })

#             yield sse_event("done", {
#                 "success": True, "book_id": book_id, "book_name": book_name,
#                 "source_file": file.filename, "total_pages": total_pages,
#                 "total_chunks": total_chunks,
#             })

#         except Exception as e:
#             yield sse_event("error", {"success": False, "message": str(e)})
#         finally:
#             if temp_path and os.path.exists(temp_path):
#                 os.remove(temp_path)

#     return StreamingResponse(
#         stream_generator(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control":       "no-cache",
#             "Connection":          "keep-alive",
#             "X-Accel-Buffering":   "no",
#             "Access-Control-Allow-Origin": "*",
#         },
#     )


# # =========================
# # API — INDEX FAST
# # =========================

# @app.post("/v1/books/index-fast")
# async def index_book_fast(payload: IndexBookRequest):
#     chunks = list(
#         chunks_col.find(
#             {
#                 "book_id": payload.book_id,
#                 "$or": [
#                     {"pinecone_indexed": {"$ne": True}},
#                     {"neo4j_indexed":    {"$ne": True}},
#                 ],
#             },
#             {"_id": 0},
#         ).sort([("page_no", ASCENDING), ("global_paragraph_no", ASCENDING)])
#     )

#     if not chunks:
#         return {
#             "success": True,
#             "message": "Already indexed or no chunks found",
#             "book_id": payload.book_id,
#         }

#     total = len(chunks)
#     indexed = 0
#     failed_batches = 0

#     for i in range(0, total, BATCH_SIZE):
#         batch = chunks[i : i + BATCH_SIZE]
#         try:
#             save_chunks_to_pinecone_batch(batch)
#             save_chunks_to_neo4j_batch(batch)
#             update_mongo_index_status_batch(batch)
#             indexed += len(batch)
#         except Exception as e:
#             failed_batches += 1
#             ops = [
#                 UpdateOne(
#                     {"chunk_id": c["chunk_id"]},
#                     {"$set": {"indexing_error": str(e), "indexed_at": now_utc()}},
#                 )
#                 for c in batch
#             ]
#             chunks_col.bulk_write(ops, ordered=False)

#     return {
#         "success":           True,
#         "book_id":           payload.book_id,
#         "total_chunks":      total,
#         "indexed":           indexed,
#         "failed_batches":    failed_batches,
#         "batch_size":        BATCH_SIZE,
#         "pinecone_namespace":f"book_{payload.book_id}",
#     }


# # =========================
# # API — SEARCH
# # =========================

# @app.post("/v1/books/search")
# async def search_book(payload: SearchRequest):
#     if not payload.query or not payload.query.strip():
#         raise HTTPException(status_code=400, detail="Search query cannot be empty")

#     namespace  = f"book_{payload.book_id}"
#     query_plan = openai_query_planner(user_query=payload.query, book_name=payload.book_name)

#     search_queries  = list(set(
#         [query_plan.get("corrected_query", payload.query)]
#         + query_plan.get("semantic_queries", [])
#     ))
#     keyword_queries = list(set(
#         [query_plan.get("main_topic", payload.query)]
#         + query_plan.get("keyword_queries", [])
#         + query_plan.get("must_have_terms",  [])
#     ))

#     vector_results  = vector_search(namespace, search_queries)
#     keyword_results = keyword_search(payload.book_id, keyword_queries)
#     hybrid_results  = merge_hybrid_results(vector_results, keyword_results)

#     if not hybrid_results:
#         return {
#             "success": True, "query": payload.query, "query_plan": query_plan,
#             "confidence": 0, "results": [],
#             "message": "No relevant content found in this book.",
#         }

#     top_ids     = [x["chunk_id"] for x in hybrid_results[:30]]
#     related_ids = get_related_chunks_from_neo4j(top_ids[:10])
#     final_ids   = list(set(top_ids + related_ids))

#     mongo_chunks = list(
#         chunks_col.find(
#             {"book_id": payload.book_id, "chunk_id": {"$in": final_ids}},
#             {"_id": 0},
#         ).sort([("page_no", ASCENDING), ("global_paragraph_no", ASCENDING)])
#     )

#     score_map = {x["chunk_id"]: x for x in hybrid_results}
#     enriched  = [
#         {
#             **c,
#             "vector_score":  score_map.get(c["chunk_id"], {}).get("vector_score",  0),
#             "keyword_score": score_map.get(c["chunk_id"], {}).get("keyword_score", 0),
#             "hybrid_score":  score_map.get(c["chunk_id"], {}).get("hybrid_score",  0),
#             "source":        "direct_hybrid_match" if c["chunk_id"] in score_map else "neo4j_related_context",
#         }
#         for c in mongo_chunks
#     ]

#     diverse_chunks = sorted(enriched, key=lambda x: x.get("hybrid_score", 0), reverse=True)[:15]
#     rerank_result  = openai_rerank(query_plan, diverse_chunks)

#     confidence   = float(rerank_result.get("confidence", 0))
#     selected_ids = set(rerank_result.get("selected_chunk_ids", []))

#     if confidence < MIN_FINAL_CONFIDENCE or not selected_ids:
#         return {
#             "success": True, "query": payload.query, "query_plan": query_plan,
#             "confidence": confidence, "results": [],
#             "message": "No sufficiently relevant content found.",
#         }

#     final_results = [
#         {
#             "chunk_id":      c["chunk_id"],
#             "book_name":     c.get("book_name"),
#             "page_no":       c.get("page_no"),
#             "chapter":       c.get("chapter"),
#             "heading":       c.get("heading_or_subtopic"),
#             "text":          c.get("text"),
#             "text_preview":  c.get("text_preview"),
#             "vector_score":  c.get("vector_score"),
#             "keyword_score": c.get("keyword_score"),
#             "hybrid_score":  c.get("hybrid_score"),
#             "source":        c.get("source"),
#         }
#         for c in diverse_chunks
#         if c["chunk_id"] in selected_ids
#     ]

#     return {
#         "success":                   True,
#         "query":                     payload.query,
#         "query_plan":                query_plan,
#         "confidence":                confidence,
#         "rerank_reason":             rerank_result.get("reason"),
#         "pinecone_vector_candidates":len(vector_results),
#         "keyword_candidates":        len(keyword_results),
#         "neo4j_related_chunks":      len(related_ids),
#         "total_context_chunks":      len(final_results),
#         "results":                   final_results,
#     }


# # =========================
# # API — GRAPH DATA  (for frontend knowledge graph)
# # =========================

# @app.get("/v1/books/{book_id}/graph")
# async def get_book_graph(book_id: str, chunk_ids: str = ""):
#     """
#     Returns graph nodes + edges for visualisation.
#     Optional: pass comma-separated chunk_ids to highlight matched chunks.
#     """
#     highlight_ids = set(chunk_ids.split(",")) if chunk_ids else set()

#     query = """
#     MATCH (b:Book {book_id: $book_id})
#     OPTIONAL MATCH (b)-[:HAS_CHAPTER]->(c:Chapter)
#     OPTIONAL MATCH (c)-[:HAS_HEADING]->(h:Heading)
#     OPTIONAL MATCH (h)-[:HAS_CHUNK]  ->(ch:Chunk)
#     OPTIONAL MATCH (b)-[:HAS_PAGE]   ->(p:Page)
#     OPTIONAL MATCH (p)-[:HAS_CHUNK]  ->(ch2:Chunk)
#     RETURN b, c, h, ch, p, ch2
#     LIMIT 500
#     """

#     nodes: Dict[str, dict] = {}
#     edges: List[dict]      = []

#     try:
#         with neo4j_driver.session() as session:
#             result = session.run(query, book_id=book_id)

#             def add_node(nid: str, label: str, props: dict, ntype: str):
#                 if nid not in nodes:
#                     nodes[nid] = {
#                         "id":        nid,
#                         "label":     label,
#                         "type":      ntype,
#                         "props":     props,
#                         "highlight": nid in highlight_ids,
#                     }

#             def add_edge(src: str, tgt: str, rel: str):
#                 edges.append({"source": src, "target": tgt, "relation": rel})

#             for record in result:
#                 b   = record.get("b")
#                 c   = record.get("c")
#                 h   = record.get("h")
#                 ch  = record.get("ch")
#                 p   = record.get("p")
#                 ch2 = record.get("ch2")

#                 if b:
#                     bid = f"book_{b['book_id']}"
#                     add_node(bid, b.get("name", "Book"), dict(b), "book")

#                 if c and b:
#                     bid = f"book_{b['book_id']}"
#                     cid = f"chapter_{b['book_id']}_{c['name']}"
#                     add_node(cid, c.get("name", "Chapter"), dict(c), "chapter")
#                     add_edge(bid, cid, "HAS_CHAPTER")

#                 if h and c and b:
#                     cid = f"chapter_{b['book_id']}_{c['name']}"
#                     hid = f"heading_{b['book_id']}_{c['name']}_{h['name']}"
#                     add_node(hid, h.get("name", "Heading"), dict(h), "heading")
#                     add_edge(cid, hid, "HAS_HEADING")

#                 if ch and h and c and b:
#                     hid   = f"heading_{b['book_id']}_{c['name']}_{h['name']}"
#                     ch_id = ch.get("chunk_id", "")
#                     add_node(ch_id, f"Chunk p{ch.get('page_no','')}",
#                              dict(ch), "chunk")
#                     add_edge(hid, ch_id, "HAS_CHUNK")

#                 if p and b:
#                     bid = f"book_{b['book_id']}"
#                     pid = f"page_{b['book_id']}_{p['page_no']}"
#                     add_node(pid, f"Page {p.get('page_no','')}", dict(p), "page")
#                     add_edge(bid, pid, "HAS_PAGE")

#                 if ch2 and p and b:
#                     pid   = f"page_{b['book_id']}_{p['page_no']}"
#                     ch_id = ch2.get("chunk_id", "")
#                     if ch_id not in nodes:
#                         add_node(ch_id, f"Chunk p{ch2.get('page_no','')}",
#                                  dict(ch2), "chunk")
#                     add_edge(pid, ch_id, "HAS_CHUNK")

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Neo4j error: {e}")

#     return {
#         "success": True,
#         "book_id": book_id,
#         "nodes":   list(nodes.values()),
#         "edges":   edges,
#     }
















# # # main.py

# # import os
# # import re
# # import uuid
# # import json
# # import asyncio
# # import tempfile
# # from datetime import datetime, timezone
# # from typing import List, Dict, Any, Optional
# # from pydantic import BaseModel
# # from pinecone import Pinecone
# # from neo4j import GraphDatabase
# # import fitz
# # import pytesseract
# # from PIL import Image

# # from pymongo import MongoClient, ASCENDING
# # from pymongo.errors import DuplicateKeyError

# # from fastapi import FastAPI, UploadFile, File, Form, HTTPException
# # from fastapi.responses import JSONResponse, StreamingResponse
# # from fastapi.middleware.cors import CORSMiddleware
# # from openai import OpenAI


# # # =========================
# # # APP
# # # =========================

# # app = FastAPI(
# #     title="Universal Document Chunking API",
# #     version="2.0.0"
# # )

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=False,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # =========================
# # # CONFIG
# # # =========================

# # MIN_TEXT_PER_PAGE = 40
# # MIN_PARAGRAPH_CHARS = 60

# # CHUNK_SIZE = 1200
# # CHUNK_OVERLAP = 180

# # MONGO_URI = "mongodb://localhost:27017"
# # MONGO_DB_NAME = "data_edtech"

# # CHUNKS_COLLECTION = "chunks"
# # BOOKS_COLLECTION = "books"

# # USE_OPENAI_FOR_TOPIC = 'true'

# # OPENAI_API_KEY = 'sk......'
# # OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


# # # =========================
# # # DB
# # # =========================

# # mongo_client = MongoClient(MONGO_URI)
# # mongo_db = mongo_client[MONGO_DB_NAME]

# # chunks_col = mongo_db[CHUNKS_COLLECTION]
# # books_col = mongo_db[BOOKS_COLLECTION]

# # chunks_col.create_index([("chunk_id", ASCENDING)], unique=True)
# # chunks_col.create_index([("book_id", ASCENDING)])
# # chunks_col.create_index([("page_no", ASCENDING)])
# # chunks_col.create_index([("chapter", ASCENDING)])
# # chunks_col.create_index([("heading_or_subtopic", ASCENDING)])

# # books_col.create_index([("book_id", ASCENDING)], unique=True)


# # # =========================
# # # OPENAI
# # # =========================

# # openai_client = None

# # if USE_OPENAI_FOR_TOPIC and OPENAI_API_KEY and OpenAI:
# #     openai_client = OpenAI(api_key=OPENAI_API_KEY)


# # # =========================
# # # HELPERS
# # # =========================

# # def now_utc():
# #     return datetime.now(timezone.utc)


# # def sse_event(event: str, data: dict):
# #     return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


# # def normalize_text(text: str) -> str:

# #     if not text:
# #         return ""

# #     text = text.replace("\x00", " ")

# #     text = re.sub(r"[ \t]+", " ", text)

# #     text = re.sub(r"\n{3,}", "\n\n", text)

# #     text = re.sub(r" *\n *", "\n", text)

# #     return text.strip()


# # def clean_paragraph(text: str) -> str:

# #     text = normalize_text(text)

# #     text = re.sub(r"\s+", " ", text)

# #     return text.strip()


# # # =========================
# # # NOISE REMOVE
# # # =========================

# # def remove_noise_lines(text: str) -> str:

# #     cleaned = []

# #     for line in text.split("\n"):

# #         line = line.strip()

# #         if not line:
# #             cleaned.append("")
# #             continue

# #         # page number only
# #         if re.fullmatch(r"\d+", line):
# #             continue

# #         # header like MATHS BOOK 17
# #         if re.match(r"^[A-Z\s]+ \d+$", line):
# #             continue

# #         cleaned.append(line)

# #     return "\n".join(cleaned)


# # # =========================
# # # BLOCK SPLIT
# # # =========================

# # def split_into_blocks(text: str) -> List[str]:

# #     text = normalize_text(text)

# #     text = remove_noise_lines(text)

# #     blocks = re.split(r"\n\s*\n", text)

# #     return [b.strip() for b in blocks if b.strip()]


# # # =========================
# # # OCR
# # # =========================

# # def ocr_page(page: fitz.Page) -> str:

# #     try:

# #         pix = page.get_pixmap(dpi=300)

# #         img = Image.frombytes(
# #             "RGB",
# #             [pix.width, pix.height],
# #             pix.samples
# #         )

# #         text = pytesseract.image_to_string(img)

# #         return normalize_text(text)

# #     except Exception:
# #         return ""


# # def extract_page_text(page: fitz.Page):

# #     try:
# #         normal_text = normalize_text(
# #             page.get_text("text")
# #         )
# #     except Exception:
# #         normal_text = ""

# #     if len(normal_text) >= MIN_TEXT_PER_PAGE:

# #         return {
# #             "text": normal_text,
# #             "extraction_method": "pymupdf"
# #         }

# #     ocr_text = ocr_page(page)

# #     return {
# #         "text": ocr_text,
# #         "extraction_method": (
# #             "ocr_fallback"
# #             if ocr_text
# #             else "failed"
# #         )
# #     }


# # # =========================
# # # SMART HEADING DETECTION
# # # =========================

# # def looks_like_heading(line: str) -> bool:

# #     line = line.strip()

# #     if not line:
# #         return False

# #     words = line.split()

# #     if len(words) > 14:
# #         return False

# #     if re.fullmatch(r"\d+", line):
# #         return False

# #     # Chapter 1
# #     if re.match(
# #         r"^(chapter|unit|lesson|section|topic|part)\s+[\dA-Za-z]+",
# #         line,
# #         re.I
# #     ):
# #         return True

# #     # 1.1 Rational Numbers
# #     if re.match(r"^\d+(\.\d+)*[\).]?\s+[A-Za-z]", line):
# #         return True

# #     # ALL CAPS
# #     if line.isupper() and 2 <= len(words) <= 10:
# #         return True

# #     # Title Case
# #     title_words = sum(
# #         1 for w in words if w[:1].isupper()
# #     )

# #     if 2 <= len(words) <= 10:

# #         if title_words >= len(words) - 1:

# #             if not re.search(r"[.!?]$", line):
# #                 return True

# #     return False


# # def clean_topic_name(topic: str) -> str:

# #     topic = clean_paragraph(topic)

# #     topic = re.sub(
# #         r"^(chapter|unit|lesson|section)\s*\d*[:.-]?\s*",
# #         "",
# #         topic,
# #         flags=re.I
# #     )

# #     topic = topic.strip(" :-–—.")

# #     return topic[:90] if topic else "General Content"


# # # =========================
# # # AI TOPIC GENERATOR
# # # =========================

# # def fallback_topic_from_text(text: str) -> str:

# #     text = clean_paragraph(text)

# #     words = text.split()

# #     if not words:
# #         return "General Content"

# #     return clean_topic_name(
# #         " ".join(words[:8])
# #     )


# # def ai_generate_topic_title(
# #     text: str,
# #     previous_topic: Optional[str] = None
# # ):

# #     if not openai_client:

# #         return {
# #             "chapter": "General Document",
# #             "heading_or_subtopic": fallback_topic_from_text(text),
# #             "is_new_section": False,
# #             "detection_method": "fallback_summary_title"
# #         }

# #     try:

# #         sample = clean_paragraph(text[:1800])

# #         prompt = f"""
# # You are an intelligent document understanding engine.

# # Your task:
# # Extract or generate the BEST section/topic title.

# # Rules:
# # - Works for books, reports, notes, scanned PDFs, legal docs.
# # - If heading exists -> return heading.
# # - If no heading exists -> generate small meaningful topic.
# # - Keep heading max 8 words.
# # - Do NOT generate random headings.
# # - If paragraph continues previous topic, keep same topic.

# # Previous Topic:
# # {previous_topic or "None"}

# # Return ONLY JSON:

# # {{
# #   "chapter": "string",
# #   "heading_or_subtopic": "string",
# #   "is_new_section": true/false
# # }}

# # Content:
# # {sample}
# # """

# #         response = openai_client.chat.completions.create(
# #             model=OPENAI_MODEL,
# #             messages=[
# #                 {
# #                     "role": "system",
# #                     "content": (
# #                         "You generate clean "
# #                         "document section names."
# #                     )
# #                 },
# #                 {
# #                     "role": "user",
# #                     "content": prompt
# #                 }
# #             ],
# #             temperature=0,
# #             response_format={
# #                 "type": "json_object"
# #             }
# #         )

# #         data = json.loads(
# #             response.choices[0].message.content
# #         )

# #         return {
# #             "chapter": clean_topic_name(
# #                 data.get("chapter") or "General Document"
# #             ),
# #             "heading_or_subtopic": clean_topic_name(
# #                 data.get("heading_or_subtopic")
# #                 or fallback_topic_from_text(text)
# #             ),
# #             "is_new_section": bool(
# #                 data.get("is_new_section", False)
# #             ),
# #             "detection_method": "ai_generated_topic"
# #         }

# #     except Exception:

# #         return {
# #             "chapter": "General Document",
# #             "heading_or_subtopic": fallback_topic_from_text(text),
# #             "is_new_section": False,
# #             "detection_method": "fallback_summary_title"
# #         }


# # # =========================
# # # SECTION DETECTION
# # # =========================

# # def detect_document_section(
# #     block: str,
# #     current_chapter: Optional[str],
# #     current_heading: Optional[str]
# # ):

# #     lines = [
# #         l.strip()
# #         for l in block.split("\n")
# #         if l.strip()
# #     ]

# #     # REAL HEADING
# #     if lines and looks_like_heading(lines[0]):

# #         heading = clean_topic_name(lines[0])

# #         chapter = current_chapter or "General Document"

# #         if re.match(
# #             r"^(chapter|unit|lesson|section|part)",
# #             lines[0],
# #             re.I
# #         ):
# #             chapter = heading

# #         return {
# #             "chapter": chapter,
# #             "heading_or_subtopic": heading,
# #             "is_new_section": True,
# #             "detection_method": "real_heading_detected"
# #         }

# #     # AI GENERATED
# #     generated = ai_generate_topic_title(
# #         block,
# #         current_heading
# #     )

# #     if (
# #         current_heading
# #         and not generated.get("is_new_section")
# #     ):

# #         return {
# #             "chapter": current_chapter or generated["chapter"],
# #             "heading_or_subtopic": current_heading,
# #             "is_new_section": False,
# #             "detection_method": generated["detection_method"]
# #         }

# #     return generated


# # # =========================
# # # SENTENCE CHUNKING
# # # =========================

# # def sentence_aware_chunking(
# #     text: str,
# #     max_chars: int = CHUNK_SIZE
# # ):

# #     text = clean_paragraph(text)

# #     if len(text) <= max_chars:
# #         return [text]

# #     sentences = re.split(
# #         r'(?<=[.!?])\s+',
# #         text
# #     )

# #     chunks = []

# #     current = ""

# #     for sentence in sentences:

# #         if len(current) + len(sentence) <= max_chars:

# #             current += " " + sentence

# #         else:

# #             if current.strip():
# #                 chunks.append(current.strip())

# #             current = sentence

# #     if current.strip():
# #         chunks.append(current.strip())

# #     return chunks


# # # =========================
# # # MONGO SAVE
# # # =========================

# # def safe_insert_chunk(chunk):

# #     try:
# #         chunks_col.insert_one(chunk)
# #         return True

# #     except DuplicateKeyError:
# #         return False

# #     except Exception:
# #         return False


# # def save_book_meta(book_doc):

# #     try:

# #         books_col.update_one(
# #             {"book_id": book_doc["book_id"]},
# #             {"$set": book_doc},
# #             upsert=True
# #         )

# #     except Exception:
# #         pass


# # # =========================
# # # CHUNK BUILDER
# # # =========================

# # def build_chunks(
# #     book_id: str,
# #     book_name: str,
# #     filename: str,
# #     pages: List[Dict[str, Any]],
# #     save_to_mongo: bool = True
# # ):

# #     chunks = []

# #     current_chapter = None
# #     current_heading = None

# #     global_paragraph_no = 0

# #     for page_data in pages:

# #         page_no = page_data["page_no"]

# #         page_text = page_data["text"]

# #         extraction_method = page_data["extraction_method"]

# #         blocks = split_into_blocks(page_text)

# #         page_paragraph_no = 0

# #         for block in blocks:

# #             lines = [
# #                 l.strip()
# #                 for l in block.split("\n")
# #                 if l.strip()
# #             ]

# #             if not lines:
# #                 continue

# #             paragraph_text = clean_paragraph(
# #                 " ".join(lines)
# #             )

# #             if len(paragraph_text) < MIN_PARAGRAPH_CHARS:
# #                 continue

# #             detected = detect_document_section(
# #                 block,
# #                 current_chapter,
# #                 current_heading
# #             )

# #             current_chapter = (
# #                 detected["chapter"]
# #                 or current_chapter
# #                 or "General Document"
# #             )

# #             current_heading = (
# #                 detected["heading_or_subtopic"]
# #                 or current_heading
# #                 or "General Content"
# #             )

# #             page_paragraph_no += 1
# #             global_paragraph_no += 1

# #             split_texts = sentence_aware_chunking(
# #                 paragraph_text
# #             )

# #             for local_chunk_index, chunk_text in enumerate(
# #                 split_texts,
# #                 start=1
# #             ):

# #                 chunk = {

# #                     "chunk_id": str(uuid.uuid4()),

# #                     "book_id": book_id,

# #                     "book_name": book_name,

# #                     "source_file": filename,

# #                     "page_no": page_no,

# #                     "page_paragraph_no": page_paragraph_no,

# #                     "global_paragraph_no": global_paragraph_no,

# #                     "chapter": current_chapter,

# #                     "heading_or_subtopic": current_heading,

# #                     "topic_detection_method":
# #                         detected["detection_method"],

# #                     "is_new_section":
# #                         detected.get("is_new_section", False),

# #                     "chunk_index_inside_paragraph":
# #                         local_chunk_index,

# #                     "total_chunks_from_paragraph":
# #                         len(split_texts),

# #                     "extraction_method":
# #                         extraction_method,

# #                     "char_count":
# #                         len(chunk_text),

# #                     "word_count":
# #                         len(chunk_text.split()),

# #                     "text_preview":
# #                         chunk_text[:250],

# #                     "text":
# #                         chunk_text,

# #                     "status":
# #                         "chunked",

# #                     "created_at":
# #                         now_utc(),

# #                     "updated_at":
# #                         now_utc()
# #                 }

# #                 if save_to_mongo:
# #                     safe_insert_chunk(chunk)

# #                 chunks.append(chunk)

# #     return chunks


# # # =========================
# # # API
# # # =========================

# # @app.post("/v1/books/chunk")
# # async def upload_and_chunk_book(
# #     book_name: str = Form(...),
# #     file: UploadFile = File(...)
# # ):

# #     if not file.filename.lower().endswith(".pdf"):

# #         raise HTTPException(
# #             status_code=400,
# #             detail="Only PDF supported"
# #         )

# #     temp_path = None

# #     book_id = str(uuid.uuid4())

# #     try:

# #         with tempfile.NamedTemporaryFile(
# #             delete=False,
# #             suffix=".pdf"
# #         ) as tmp:

# #             tmp.write(await file.read())

# #             temp_path = tmp.name

# #         pdf = fitz.open(temp_path)

# #         save_book_meta({
# #             "book_id": book_id,
# #             "book_name": book_name,
# #             "source_file": file.filename,
# #             "status": "processing",
# #             "created_at": now_utc(),
# #             "updated_at": now_utc()
# #         })

# #         pages = []

# #         for i in range(len(pdf)):

# #             page = pdf.load_page(i)

# #             extracted = extract_page_text(page)

# #             pages.append({
# #                 "page_no": i + 1,
# #                 "text": extracted["text"],
# #                 "extraction_method":
# #                     extracted["extraction_method"],
# #                 "char_count":
# #                     len(extracted["text"])
# #             })

# #         chunks = build_chunks(
# #             book_id=book_id,
# #             book_name=book_name,
# #             filename=file.filename,
# #             pages=pages,
# #             save_to_mongo=True
# #         )

# #         save_book_meta({
# #             "book_id": book_id,
# #             "book_name": book_name,
# #             "source_file": file.filename,
# #             "status": "chunked",
# #             "total_pages": len(pdf),
# #             "total_chunks": len(chunks),
# #             "updated_at": now_utc()
# #         })

# #         return JSONResponse({

# #             "success": True,

# #             "book_id": book_id,

# #             "book_name": book_name,

# #             "total_pages": len(pdf),

# #             "total_chunks": len(chunks),

# #             "database": MONGO_DB_NAME,

# #             "collection": CHUNKS_COLLECTION,

# #             "chunks": chunks
# #         })

# #     except Exception as e:

# #         raise HTTPException(
# #             status_code=500,
# #             detail=str(e)
# #         )

# #     finally:

# #         if temp_path and os.path.exists(temp_path):
# #             os.remove(temp_path)


# # # =========================
# # # GET CHUNKS
# # # =========================

# # @app.post("/v1/books/chunk/stream")
# # async def upload_and_chunk_book_stream(
# #     book_name: str = Form(...),
# #     file: UploadFile = File(...)
# # ):
# #     if not file.filename.lower().endswith(".pdf"):
# #         raise HTTPException(status_code=400, detail="Only PDF supported")

# #     temp_path = None
# #     book_id = str(uuid.uuid4())

# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
# #         tmp.write(await file.read())
# #         temp_path = tmp.name

# #     async def stream_generator():
# #         total_chunks = 0

# #         try:
# #             yield sse_event("start", {
# #                 "success": True,
# #                 "book_id": book_id,
# #                 "book_name": book_name,
# #                 "source_file": file.filename
# #             })

# #             pdf = fitz.open(temp_path)
# #             total_pages = len(pdf)

# #             yield sse_event("pdf_info", {
# #                 "book_id": book_id,
# #                 "total_pages": total_pages
# #             })

# #             current_chapter = None
# #             current_heading = None
# #             global_paragraph_no = 0

# #             for i in range(total_pages):
# #                 page_no = i + 1
# #                 page = pdf.load_page(i)
# #                 extracted = extract_page_text(page)

# #                 page_text = extracted["text"]
# #                 extraction_method = extracted["extraction_method"]

# #                 yield sse_event("page_extracted", {
# #                     "page_no": page_no,
# #                     "char_count": len(page_text),
# #                     "extraction_method": extraction_method
# #                 })

# #                 blocks = split_into_blocks(page_text)
# #                 page_paragraph_no = 0

# #                 for block in blocks:
# #                     lines = [l.strip() for l in block.split("\n") if l.strip()]
# #                     if not lines:
# #                         continue

# #                     paragraph_text = clean_paragraph(" ".join(lines))

# #                     if len(paragraph_text) < MIN_PARAGRAPH_CHARS:
# #                         continue

# #                     detected = detect_document_section(
# #                         block,
# #                         current_chapter,
# #                         current_heading
# #                     )

# #                     current_chapter = detected["chapter"] or current_chapter or "General Document"
# #                     current_heading = detected["heading_or_subtopic"] or current_heading or "General Content"

# #                     yield sse_event("heading_detected", {
# #                         "page_no": page_no,
# #                         "chapter": current_chapter,
# #                         "heading_or_subtopic": current_heading,
# #                         "detection_method": detected["detection_method"]
# #                     })

# #                     page_paragraph_no += 1
# #                     global_paragraph_no += 1

# #                     split_texts = sentence_aware_chunking(paragraph_text)

# #                     for local_chunk_index, chunk_text in enumerate(split_texts, start=1):
# #                         total_chunks += 1

# #                         chunk = {
# #                             "chunk_id": str(uuid.uuid4()),
# #                             "book_id": book_id,
# #                             "book_name": book_name,
# #                             "source_file": file.filename,
# #                             "page_no": page_no,
# #                             "page_paragraph_no": page_paragraph_no,
# #                             "global_paragraph_no": global_paragraph_no,
# #                             "chapter": current_chapter,
# #                             "heading_or_subtopic": current_heading,
# #                             "topic_detection_method": detected["detection_method"],
# #                             "is_new_section": detected.get("is_new_section", False),
# #                             "chunk_index_inside_paragraph": local_chunk_index,
# #                             "total_chunks_from_paragraph": len(split_texts),
# #                             "extraction_method": extraction_method,
# #                             "char_count": len(chunk_text),
# #                             "word_count": len(chunk_text.split()),
# #                             "text_preview": chunk_text[:250],
# #                             "text": chunk_text,
# #                             "status": "chunked",
# #                             "created_at": now_utc(),
# #                             "updated_at": now_utc()
# #                         }

# #                         safe_insert_chunk(chunk)

# #                         yield sse_event("chunk", chunk)

# #                         await asyncio.sleep(0)

# #                 yield sse_event("page_done", {
# #                     "page_no": page_no,
# #                     "total_pages": total_pages,
# #                     "total_chunks_so_far": total_chunks
# #                 })

# #             yield sse_event("done", {
# #                 "success": True,
# #                 "book_id": book_id,
# #                 "book_name": book_name,
# #                 "source_file": file.filename,
# #                 "total_pages": total_pages,
# #                 "total_chunks": total_chunks
# #             })

# #         except Exception as e:
# #             yield sse_event("error", {
# #                 "success": False,
# #                 "message": str(e)
# #             })

# #         finally:
# #             if temp_path and os.path.exists(temp_path):
# #                 os.remove(temp_path)

# #     return StreamingResponse(
# #         stream_generator(),
# #         media_type="text/event-stream",
# #         headers={
# #             "Cache-Control": "no-cache",
# #             "Connection": "keep-alive",
# #             "X-Accel-Buffering": "no",
# #             "Access-Control-Allow-Origin": "*"
# #         }
# #     )




# # # =========================
# # # VECTOR + GRAPH CONFIG
# # # =========================

# # OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
# # EMBEDDING_DIMENSION = 1536

# # PINECONE_API_KEY = 'pcs....'
# # PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "aieducation")

# # NEO4J_URI="bolt://localhost:7687"
# # NEO4J_USERNAME="neo4j"
# # NEO4J_PASSWORD="secretgraph"

# # pc = Pinecone(api_key=PINECONE_API_KEY)
# # pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# # neo4j_driver = GraphDatabase.driver(
# #     NEO4J_URI,
# #     auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
# # )


# # # =========================
# # # MODELS
# # # =========================

# # class IndexBookRequest(BaseModel):
# #     book_id: str


# # class SearchRequest(BaseModel):
# #     book_id: str
# #     query: str
# #     top_k: int = 5


# # from pymongo import UpdateOne

# # BATCH_SIZE = 50


# # def create_embeddings_batch(texts: list[str]):
# #     response = openai_client.embeddings.create(
# #         model="text-embedding-3-small",
# #         input=texts
# #     )

# #     return [item.embedding for item in response.data]


# # def build_embedding_text(chunk: dict):
# #     return f"""
# # Book: {chunk.get("book_name", "")}
# # Chapter: {chunk.get("chapter", "General Document")}
# # Heading: {chunk.get("heading_or_subtopic", "General Content")}
# # Page: {chunk.get("page_no", "")}

# # Content:
# # {chunk.get("text", "")}
# # """.strip()


# # def save_chunks_to_pinecone_batch(chunks: list[dict]):
# #     texts = [build_embedding_text(chunk) for chunk in chunks]

# #     vectors = create_embeddings_batch(texts)

# #     pinecone_vectors = []

# #     for chunk, vector in zip(chunks, vectors):
# #         pinecone_vectors.append({
# #             "id": chunk["chunk_id"],
# #             "values": vector,
# #             "metadata": {
# #                 "chunk_id": chunk["chunk_id"],
# #                 "book_id": chunk["book_id"],
# #                 "book_name": chunk.get("book_name", ""),
# #                 "source_file": chunk.get("source_file", ""),
# #                 "page_no": chunk.get("page_no", 0),
# #                 "chapter": chunk.get("chapter", "General Document"),
# #                 "heading": chunk.get("heading_or_subtopic", "General Content"),
# #                 "text": chunk.get("text", "")[:3000],
# #                 "text_preview": chunk.get("text_preview", ""),
# #                 "topic_detection_method": chunk.get("topic_detection_method", "")
# #             }
# #         })

# #     pinecone_index.upsert(
# #         vectors=pinecone_vectors,
# #         namespace=f"book_{chunks[0]['book_id']}"
# #     )


# # def save_chunks_to_neo4j_batch(chunks: list[dict]):
# #     query = """
# #     UNWIND $chunks AS chunk

# #     MERGE (b:Book {book_id: chunk.book_id})
# #     SET b.name = chunk.book_name,
# #         b.source_file = chunk.source_file

# #     MERGE (c:Chapter {
# #         book_id: chunk.book_id,
# #         name: chunk.chapter
# #     })

# #     MERGE (h:Heading {
# #         book_id: chunk.book_id,
# #         chapter: chunk.chapter,
# #         name: chunk.heading
# #     })

# #     MERGE (p:Page {
# #         book_id: chunk.book_id,
# #         page_no: chunk.page_no
# #     })

# #     MERGE (ch:Chunk {
# #         chunk_id: chunk.chunk_id
# #     })
# #     SET ch.text_preview = chunk.text_preview,
# #         ch.page_no = chunk.page_no,
# #         ch.word_count = chunk.word_count,
# #         ch.global_paragraph_no = chunk.global_paragraph_no

# #     MERGE (b)-[:HAS_CHAPTER]->(c)
# #     MERGE (c)-[:HAS_HEADING]->(h)
# #     MERGE (h)-[:HAS_CHUNK]->(ch)
# #     MERGE (b)-[:HAS_PAGE]->(p)
# #     MERGE (p)-[:HAS_CHUNK]->(ch)
# #     """

# #     rows = []

# #     for chunk in chunks:
# #         rows.append({
# #             "chunk_id": chunk["chunk_id"],
# #             "book_id": chunk["book_id"],
# #             "book_name": chunk.get("book_name", ""),
# #             "source_file": chunk.get("source_file", ""),
# #             "chapter": chunk.get("chapter", "General Document"),
# #             "heading": chunk.get("heading_or_subtopic", "General Content"),
# #             "page_no": chunk.get("page_no", 0),
# #             "text_preview": chunk.get("text_preview", ""),
# #             "word_count": chunk.get("word_count", 0),
# #             "global_paragraph_no": chunk.get("global_paragraph_no", 0)
# #         })

# #     with neo4j_driver.session() as session:
# #         session.run(query, chunks=rows)


# # def update_mongo_index_status_batch(chunks: list[dict]):
# #     operations = []

# #     for chunk in chunks:
# #         operations.append(
# #             UpdateOne(
# #                 {"chunk_id": chunk["chunk_id"]},
# #                 {
# #                     "$set": {
# #                         "pinecone_indexed": True,
# #                         "neo4j_indexed": True,
# #                         "indexed_at": now_utc()
# #                     }
# #                 }
# #             )
# #         )

# #     if operations:
# #         chunks_col.bulk_write(operations, ordered=False)


# # @app.post("/v1/books/index-fast")
# # async def index_book_fast(payload: IndexBookRequest):
# #     chunks = list(
# #         chunks_col.find(
# #             {
# #                 "book_id": payload.book_id,
# #                 "$or": [
# #                     {"pinecone_indexed": {"$ne": True}},
# #                     {"neo4j_indexed": {"$ne": True}}
# #                 ]
# #             },
# #             {"_id": 0}
# #         ).sort([
# #             ("page_no", ASCENDING),
# #             ("global_paragraph_no", ASCENDING)
# #         ])
# #     )

# #     if not chunks:
# #         return {
# #             "success": True,
# #             "message": "Already indexed or no chunks found",
# #             "book_id": payload.book_id
# #         }

# #     total = len(chunks)
# #     indexed = 0
# #     failed_batches = 0

# #     for i in range(0, total, BATCH_SIZE):
# #         batch = chunks[i:i + BATCH_SIZE]

# #         try:
# #             save_chunks_to_pinecone_batch(batch)
# #             save_chunks_to_neo4j_batch(batch)
# #             update_mongo_index_status_batch(batch)

# #             indexed += len(batch)

# #         except Exception as e:
# #             failed_batches += 1

# #             for chunk in batch:
# #                 chunks_col.update_one(
# #                     {"chunk_id": chunk["chunk_id"]},
# #                     {
# #                         "$set": {
# #                             "indexing_error": str(e),
# #                             "indexed_at": now_utc()
# #                         }
# #                     }
# #                 )

# #     return {
# #         "success": True,
# #         "book_id": payload.book_id,
# #         "total_chunks": total,
# #         "indexed": indexed,
# #         "failed_batches": failed_batches,
# #         "batch_size": BATCH_SIZE,
# #         "pinecone_namespace": f"book_{payload.book_id}"
# #     }

# # # =========================
# # # NEO4J EXPAND NEARBY CONTEXT
# # # =========================





# # import math
# # from typing import List, Dict, Any


# # MIN_FINAL_CONFIDENCE = 0.55
# # VECTOR_TOP_K = 25
# # KEYWORD_TOP_K = 25
# # FINAL_TOP_K = 8


# # # =========================
# # # OPENAI JSON HELPER
# # # =========================

# # def call_openai_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
# #     response = openai_client.chat.completions.create(
# #         model="gpt-4.1-mini",
# #         temperature=0,
# #         response_format={"type": "json_object"},
# #         messages=[
# #             {"role": "system", "content": system_prompt},
# #             {"role": "user", "content": user_prompt}
# #         ]
# #     )

# #     return json.loads(response.choices[0].message.content)


# # # =========================
# # # STEP 1: QUERY PLANNER
# # # =========================

# # def openai_query_planner(user_query: str, book_name: str = "") -> Dict[str, Any]:
# #     system_prompt = """
# # You are a production-level educational search query planner.

# # Your job:
# # 1. Understand the user's actual learning intent.
# # 2. Fix spelling mistakes naturally.
# # 3. Expand the query into semantic and keyword search queries.
# # 4. Detect whether the query is about a topic, definition, formula, example, exercise, or explanation.
# # 5. Return JSON only.

# # Rules:
# # - Do not answer the question.
# # - Do not hallucinate book content.
# # - Keep search queries short and useful.
# # - If user query is unclear, still create best possible search queries.
# # """

# #     user_prompt = f"""
# # User query: {user_query}
# # Book name: {book_name}

# # Return JSON in this format:
# # {{
# #   "corrected_query": "",
# #   "intent": "definition|concept|formula|example|exercise|explanation|unknown",
# #   "main_topic": "",
# #   "semantic_queries": [],
# #   "keyword_queries": [],
# #   "must_have_terms": [],
# #   "avoid_terms": []
# # }}
# # """

# #     return call_openai_json(system_prompt, user_prompt)


# # # =========================
# # # STEP 2: VECTOR SEARCH
# # # =========================

# # def vector_search(namespace: str, queries: List[str], top_k: int = VECTOR_TOP_K):
# #     all_matches = {}

# #     for q in queries:
# #         query_vector = create_embeddings_batch(q)

# #         result = pinecone_index.query(
# #             namespace=namespace,
# #             vector=query_vector,
# #             top_k=top_k,
# #             include_metadata=True
# #         )

# #         for match in result.get("matches", []):
# #             chunk_id = match["id"]
# #             score = float(match.get("score", 0))

# #             if chunk_id not in all_matches or score > all_matches[chunk_id]["vector_score"]:
# #                 all_matches[chunk_id] = {
# #                     "chunk_id": chunk_id,
# #                     "vector_score": score
# #                 }

# #     return list(all_matches.values())


# # # =========================
# # # STEP 3: KEYWORD SEARCH
# # # Mongo text index required:
# # # db.chunks.createIndex({ text: "text", chapter: "text", heading_or_subtopic: "text" })
# # # =========================

# # def keyword_search(book_id: str, keyword_queries: List[str], top_k: int = KEYWORD_TOP_K):
# #     results = {}

# #     for q in keyword_queries:
# #         cursor = chunks_col.find(
# #             {
# #                 "book_id": book_id,
# #                 "$text": {"$search": q}
# #             },
# #             {
# #                 "_id": 0,
# #                 "chunk_id": 1,
# #                 "keyword_score": {"$meta": "textScore"}
# #             }
# #         ).sort([
# #             ("keyword_score", {"$meta": "textScore"})
# #         ]).limit(top_k)

# #         for doc in cursor:
# #             chunk_id = doc["chunk_id"]
# #             score = float(doc.get("keyword_score", 0))

# #             if chunk_id not in results or score > results[chunk_id]["keyword_score"]:
# #                 results[chunk_id] = {
# #                     "chunk_id": chunk_id,
# #                     "keyword_score": score
# #                 }

# #     return list(results.values())


# # # =========================
# # # STEP 4: NORMALIZE + MERGE
# # # =========================

# # def normalize_scores(items: List[Dict[str, Any]], score_key: str):
# #     if not items:
# #         return {}

# #     scores = [x.get(score_key, 0) for x in items]
# #     min_s, max_s = min(scores), max(scores)

# #     normalized = {}

# #     for item in items:
# #         chunk_id = item["chunk_id"]
# #         score = item.get(score_key, 0)

# #         if max_s == min_s:
# #             norm = 1.0
# #         else:
# #             norm = (score - min_s) / (max_s - min_s)

# #         normalized[chunk_id] = round(norm, 4)

# #     return normalized


# # def merge_hybrid_results(vector_results, keyword_results):
# #     vector_norm = normalize_scores(vector_results, "vector_score")
# #     keyword_norm = normalize_scores(keyword_results, "keyword_score")

# #     all_ids = set(vector_norm.keys()) | set(keyword_norm.keys())

# #     merged = []

# #     for chunk_id in all_ids:
# #         v = vector_norm.get(chunk_id, 0)
# #         k = keyword_norm.get(chunk_id, 0)

# #         hybrid_score = (0.65 * v) + (0.35 * k)

# #         merged.append({
# #             "chunk_id": chunk_id,
# #             "vector_score": v,
# #             "keyword_score": k,
# #             "hybrid_score": round(hybrid_score, 4)
# #         })

# #     merged.sort(key=lambda x: x["hybrid_score"], reverse=True)

# #     return merged


# # # =========================
# # # STEP 5: MMR DIVERSITY
# # # =========================

# # def cosine_similarity(vec1, vec2):
# #     dot = sum(a * b for a, b in zip(vec1, vec2))
# #     norm1 = math.sqrt(sum(a * a for a in vec1))
# #     norm2 = math.sqrt(sum(b * b for b in vec2))

# #     if norm1 == 0 or norm2 == 0:
# #         return 0

# #     return dot / (norm1 * norm2)


# # def apply_mmr(chunks: List[Dict[str, Any]], lambda_mult: float = 0.75, limit: int = 15):
# #     """
# #     MMR keeps relevant chunks but removes duplicate/same-same chunks.
# #     Needs chunk embedding saved in Mongo OR re-embed chunk text.
# #     """

# #     selected = []
# #     candidates = chunks[:]

# #     while candidates and len(selected) < limit:
# #         if not selected:
# #             selected.append(candidates.pop(0))
# #             continue

# #         best_candidate = None
# #         best_score = -999

# #         for candidate in candidates:
# #             relevance = candidate.get("hybrid_score", 0)

# #             diversity_penalty = max(
# #                 cosine_similarity(
# #                     candidate.get("embedding", []),
# #                     selected_item.get("embedding", [])
# #                 )
# #                 for selected_item in selected
# #             )

# #             mmr_score = (lambda_mult * relevance) - ((1 - lambda_mult) * diversity_penalty)

# #             if mmr_score > best_score:
# #                 best_score = mmr_score
# #                 best_candidate = candidate

# #         selected.append(best_candidate)
# #         candidates.remove(best_candidate)

# #     return selected


# # # =========================
# # # STEP 6: NEO4J RELATED CONTEXT
# # # =========================

# # def get_related_chunks_from_neo4j(chunk_ids: List[str], limit: int = 20):
# #     if not chunk_ids:
# #         return []

# #     query = """
# #     MATCH (ch:Chunk)
# #     WHERE ch.chunk_id IN $chunk_ids

# #     OPTIONAL MATCH (h:Heading)-[:HAS_CHUNK]->(ch)
# #     OPTIONAL MATCH (h)-[:HAS_CHUNK]->(related:Chunk)

# #     WITH DISTINCT related.chunk_id AS chunk_id
# #     WHERE chunk_id IS NOT NULL

# #     RETURN chunk_id
# #     LIMIT $limit
# #     """

# #     related_ids = set()

# #     with neo4j_driver.session() as session:
# #         result = session.run(query, chunk_ids=chunk_ids, limit=limit)

# #         for record in result:
# #             related_ids.add(record["chunk_id"])

# #     return list(related_ids)


# # # =========================
# # # STEP 7: OPENAI RERANKER
# # # =========================

# # def openai_rerank(query_plan: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
# #     compact_chunks = []

# #     for i, chunk in enumerate(chunks):
# #         compact_chunks.append({
# #             "rank_id": i,
# #             "chunk_id": chunk["chunk_id"],
# #             "chapter": chunk.get("chapter"),
# #             "heading": chunk.get("heading_or_subtopic"),
# #             "page_no": chunk.get("page_no"),
# #             "text": chunk.get("text", "")[:1200],
# #             "hybrid_score": chunk.get("hybrid_score", 0),
# #             "source": chunk.get("source")
# #         })

# #     system_prompt = """
# # You are a strict educational RAG reranker.

# # Your job:
# # 1. Select only chunks that are truly related to the user's query.
# # 2. Reject generic number/math chunks if they do not match the actual topic.
# # 3. Prefer exact topic, chapter, heading, definition, formula, examples.
# # 4. Return confidence from 0 to 1.
# # 5. If no chunk is relevant, return empty selected_chunk_ids and confidence below 0.5.

# # Rules:
# # - Do not answer the question.
# # - Do not select weakly related chunks.
# # - Do not select chunks only because they contain common words.
# # - JSON only.
# # """

# #     user_prompt = f"""
# # Query plan:
# # {json.dumps(query_plan, indent=2)}

# # Candidate chunks:
# # {json.dumps(compact_chunks, indent=2)}

# # Return JSON:
# # {{
# #   "confidence": 0.0,
# #   "reason": "",
# #   "selected_chunk_ids": [],
# #   "rejected_chunk_ids": []
# # }}
# # """

# #     return call_openai_json(system_prompt, user_prompt)


# # # =========================
# # # MAIN API
# # # =========================

# # @app.post("/v1/books/search")
# # async def search_book(payload: SearchRequest):
# #     if not openai_client:
# #         raise HTTPException(status_code=500, detail="OpenAI client not configured")

# #     if not payload.query or not payload.query.strip():
# #         raise HTTPException(status_code=400, detail="Search query cannot be empty")

# #     namespace = f"book_{payload.book_id}"

# #     query_plan = openai_query_planner(
# #         user_query=payload.query,
# #         book_name=getattr(payload, "book_name", "")
# #     )

# #     search_queries = list(set(
# #         [query_plan.get("corrected_query", payload.query)] +
# #         query_plan.get("semantic_queries", [])
# #     ))

# #     keyword_queries = list(set(
# #         [query_plan.get("main_topic", payload.query)] +
# #         query_plan.get("keyword_queries", []) +
# #         query_plan.get("must_have_terms", [])
# #     ))

# #     vector_results = vector_search(namespace, search_queries)
# #     keyword_results = keyword_search(payload.book_id, keyword_queries)

# #     hybrid_results = merge_hybrid_results(vector_results, keyword_results)

# #     if not hybrid_results:
# #         return {
# #             "success": True,
# #             "query": payload.query,
# #             "query_plan": query_plan,
# #             "confidence": 0,
# #             "results": [],
# #             "message": "Sorry, I have no topic related to your search in this book."
# #         }

# #     top_ids = [x["chunk_id"] for x in hybrid_results[:30]]

# #     related_ids = get_related_chunks_from_neo4j(top_ids[:10])

# #     final_ids = list(set(top_ids + related_ids))

# #     mongo_chunks = list(
# #         chunks_col.find(
# #             {
# #                 "book_id": payload.book_id,
# #                 "chunk_id": {"$in": final_ids}
# #             },
# #             {"_id": 0}
# #         ).sort([
# #             ("page_no", ASCENDING),
# #             ("global_paragraph_no", ASCENDING)
# #         ])
# #     )

# #     score_map = {x["chunk_id"]: x for x in hybrid_results}

# #     enriched_chunks = []

# #     for chunk in mongo_chunks:
# #         score_data = score_map.get(chunk["chunk_id"], {})

# #         enriched_chunks.append({
# #             **chunk,
# #             "vector_score": score_data.get("vector_score", 0),
# #             "keyword_score": score_data.get("keyword_score", 0),
# #             "hybrid_score": score_data.get("hybrid_score", 0),
# #             "source": "direct_hybrid_match" if chunk["chunk_id"] in score_map else "neo4j_related_context"
# #         })

# #     # Optional MMR only if embeddings are saved in Mongo chunks
# #     chunks_with_embedding = [
# #         c for c in enriched_chunks
# #         if c.get("embedding")
# #     ]

# #     if chunks_with_embedding:
# #         diverse_chunks = apply_mmr(chunks_with_embedding, limit=15)
# #     else:
# #         diverse_chunks = sorted(
# #             enriched_chunks,
# #             key=lambda x: x.get("hybrid_score", 0),
# #             reverse=True
# #         )[:15]

# #     rerank_result = openai_rerank(query_plan, diverse_chunks)

# #     confidence = float(rerank_result.get("confidence", 0))
# #     selected_ids = set(rerank_result.get("selected_chunk_ids", []))

# #     if confidence < MIN_FINAL_CONFIDENCE or not selected_ids:
# #         return {
# #             "success": True,
# #             "query": payload.query,
# #             "query_plan": query_plan,
# #             "confidence": confidence,
# #             "results": [],
# #             "message": "Sorry, I have no topic related to your search in this book."
# #         }

# #     final_results = []

# #     for chunk in diverse_chunks:
# #         if chunk["chunk_id"] in selected_ids:
# #             final_results.append({
# #                 "chunk_id": chunk["chunk_id"],
# #                 "book_name": chunk.get("book_name"),
# #                 "page_no": chunk.get("page_no"),
# #                 "chapter": chunk.get("chapter"),
# #                 "heading": chunk.get("heading_or_subtopic"),
# #                 "text": chunk.get("text"),
# #                 "text_preview": chunk.get("text_preview"),
# #                 "vector_score": chunk.get("vector_score"),
# #                 "keyword_score": chunk.get("keyword_score"),
# #                 "hybrid_score": chunk.get("hybrid_score"),
# #                 "source": chunk.get("source")
# #             })

# #     return {
# #         "success": True,
# #         "query": payload.query,
# #         "query_plan": query_plan,
# #         "confidence": confidence,
# #         "rerank_reason": rerank_result.get("reason"),
# #         "pinecone_vector_candidates": len(vector_results),
# #         "keyword_candidates": len(keyword_results),
# #         "neo4j_related_chunks": len(related_ids),
# #         "total_context_chunks": len(final_results),
# #         "results": final_results
# #     }



















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
import math

from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.errors import DuplicateKeyError

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI


# =========================
# APP
# =========================

app = FastAPI(title="GraphRAG Document API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CONFIG  (prefer env vars; fall back to defaults for local dev)
# =========================

MIN_TEXT_PER_PAGE    = int(os.getenv("MIN_TEXT_PER_PAGE", 40))
MIN_PARAGRAPH_CHARS  = int(os.getenv("MIN_PARAGRAPH_CHARS", 60))
CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE", 1200))
CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP", 180))

MONGO_URI      = os.getenv("MONGO_URI",      "mongodb://localhost:27017")
MONGO_DB_NAME  = os.getenv("MONGO_DB_NAME",  "data_edtech")
CHUNKS_COLLECTION = "chunks"
BOOKS_COLLECTION  = "books"

OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY",       "sk......")
OPENAI_MODEL         = os.getenv("OPENAI_MODEL",         "gpt-4.1-mini")
OPENAI_EMBED_MODEL   = os.getenv("OPENAI_EMBED_MODEL",   "text-embedding-3-small")
EMBEDDING_DIMENSION  = 1536

PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY",     "pcs....")
PINECONE_INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME",  "aieducation")

NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "secretgraph")

BATCH_SIZE          = int(os.getenv("BATCH_SIZE",          50))
VECTOR_TOP_K        = int(os.getenv("VECTOR_TOP_K",        25))
KEYWORD_TOP_K       = int(os.getenv("KEYWORD_TOP_K",       25))
MIN_FINAL_CONFIDENCE = float(os.getenv("MIN_FINAL_CONFIDENCE", 0.55))


PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "all_books")

# =========================
# DB
# =========================

mongo_client = MongoClient(MONGO_URI)
mongo_db     = mongo_client[MONGO_DB_NAME]
chunks_col   = mongo_db[CHUNKS_COLLECTION]
books_col    = mongo_db[BOOKS_COLLECTION]

# Standard indexes
chunks_col.create_index([("chunk_id", ASCENDING)], unique=True)
chunks_col.create_index([("book_id",  ASCENDING)])
chunks_col.create_index([("page_no",  ASCENDING)])
chunks_col.create_index([("chapter",  ASCENDING)])
chunks_col.create_index([("heading_or_subtopic", ASCENDING)])

# ── FIX: text index required for $text queries ──────────────────────────────
# Silently ignore if it already exists (IndexKeySpecsConflict / similar)
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
    pass  # index already exists or another harmless conflict
# ─────────────────────────────────────────────────────────────────────────────

books_col.create_index([("book_id", ASCENDING)], unique=True)

# =========================
# OPENAI + PINECONE + NEO4J
# =========================

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var is required")

openai_client  = OpenAI(api_key=OPENAI_API_KEY)

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY env var is required")

pc             = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

neo4j_driver   = GraphDatabase.driver(
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
# OCR
# =========================

def ocr_page(page: fitz.Page) -> str:
    try:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return normalize_text(pytesseract.image_to_string(img))
    except Exception:
        return ""


def extract_page_text(page: fitz.Page) -> Dict[str, Any]:
    try:
        normal_text = normalize_text(page.get_text("text"))
    except Exception:
        normal_text = ""

    if len(normal_text) >= MIN_TEXT_PER_PAGE:
        return {"text": normal_text, "extraction_method": "pymupdf"}

    ocr_text = ocr_page(page)
    return {
        "text": ocr_text,
        "extraction_method": "ocr_fallback" if ocr_text else "failed",
    }


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


# =========================
# SECTION DETECTION
# =========================

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

            for idx, chunk_text in enumerate(sentence_aware_chunking(paragraph_text), start=1):
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
                    "total_chunks_from_paragraph":  len(sentence_aware_chunking(paragraph_text)),
                    "extraction_method":            extraction_method,
                    "char_count":                   len(chunk_text),
                    "word_count":                   len(chunk_text.split()),
                    "text_preview":                 chunk_text[:250],
                    "text":                         chunk_text,
                    "status":                       "chunked",
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
    """Always receives a *list* of strings; returns a list of embedding vectors."""
    if not texts:
        return []
    response = openai_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_embedding_text(chunk: dict) -> str:
    return (
        f"Book: {chunk.get('book_name', '')}\n"
        f"Chapter: {chunk.get('chapter', 'General Document')}\n"
        f"Heading: {chunk.get('heading_or_subtopic', 'General Content')}\n"
        f"Page: {chunk.get('page_no', '')}\n\n"
        f"Content:\n{chunk.get('text', '')}"
    ).strip()


# =========================
# PINECONE UPSERT
# =========================

def save_chunks_to_pinecone_batch(chunks: List[dict]) -> None:
    texts = [build_embedding_text(c) for c in chunks]
    vectors = create_embeddings_batch(texts)

    pinecone_vectors = []

    for chunk, vector in zip(chunks, vectors):
        pinecone_vectors.append({
            "id": chunk["chunk_id"],
            "values": vector,
            "metadata": {
                "chunk_id": chunk["chunk_id"],
                "book_id": chunk["book_id"],
                "book_name": chunk.get("book_name", ""),
                "source_file": chunk.get("source_file", ""),
                "page_no": chunk.get("page_no", 0),
                "chapter": chunk.get("chapter", "General Document"),
                "heading": chunk.get("heading_or_subtopic", "General Content"),
                "text_preview": chunk.get("text_preview", ""),
            }
        })

    pinecone_index.upsert(
        vectors=pinecone_vectors,
        namespace=PINECONE_NAMESPACE
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
    SET ch.text_preview       = chunk.text_preview,
        ch.page_no            = chunk.page_no,
        ch.word_count         = chunk.word_count,
        ch.global_paragraph_no= chunk.global_paragraph_no

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
# SEARCH HELPERS
# =========================

def openai_query_planner(user_query: str, book_name: str = "") -> Dict[str, Any]:
    system = """You are a production-level educational search query planner.
1. Understand the user's learning intent.
2. Fix spelling mistakes naturally.
3. Expand the query into semantic and keyword search queries.
4. Detect intent: definition|concept|formula|example|exercise|explanation|unknown.
Return JSON only."""

    user = f"""User query: {user_query}
Book name: {book_name}

Return JSON:
{{
  "corrected_query": "",
  "intent": "definition|concept|formula|example|exercise|explanation|unknown",
  "main_topic": "",
  "semantic_queries": [],
  "keyword_queries": [],
  "must_have_terms": [],
  "avoid_terms": []
}}"""
    return call_openai_json(system, user)


def vector_search(
    queries: List[str],
    book_id: Optional[str] = None,
    top_k: int = VECTOR_TOP_K
) -> List[Dict[str, Any]]:

    all_matches = {}

    pinecone_filter = {}
    if book_id:
        pinecone_filter["book_id"] = {"$eq": book_id}

    for q in queries:
        query_vector = create_embeddings_batch([q])[0]

        result = pinecone_index.query(
            namespace=PINECONE_NAMESPACE,
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter if pinecone_filter else None
        )

        for match in result.get("matches", []):
            cid = match["id"]
            score = float(match.get("score", 0))

            if cid not in all_matches or score > all_matches[cid]["vector_score"]:
                all_matches[cid] = {
                    "chunk_id": cid,
                    "vector_score": score,
                    "metadata": match.get("metadata", {})
                }

    return list(all_matches.values())

def keyword_search(
    book_id: Optional[str],
    keyword_queries: List[str],
    top_k: int = KEYWORD_TOP_K
) -> List[Dict[str, Any]]:

    results = {}

    for q in keyword_queries:
        if not q or not q.strip():
            continue

        mongo_filter = {
            "$text": {"$search": q}
        }

        if book_id:
            mongo_filter["book_id"] = book_id

        cursor = (
            chunks_col.find(
                mongo_filter,
                {
                    "_id": 0,
                    "chunk_id": 1,
                    "keyword_score": {"$meta": "textScore"}
                }
            )
            .sort([("keyword_score", {"$meta": "textScore"})])
            .limit(top_k)
        )

        for doc in cursor:
            cid = doc["chunk_id"]
            score = float(doc.get("keyword_score", 0))

            if cid not in results or score > results[cid]["keyword_score"]:
                results[cid] = {
                    "chunk_id": cid,
                    "keyword_score": score
                }

    return list(results.values())

def normalize_scores(items: List[Dict], score_key: str) -> Dict[str, float]:
    if not items:
        return {}
    scores   = [x.get(score_key, 0) for x in items]
    min_s, max_s = min(scores), max(scores)
    out = {}
    for item in items:
        s = item.get(score_key, 0)
        norm = 1.0 if max_s == min_s else (s - min_s) / (max_s - min_s)
        out[item["chunk_id"]] = round(norm, 4)
    return out


def merge_hybrid_results(vector_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
    v_norm = normalize_scores(vector_results,  "vector_score")
    k_norm = normalize_scores(keyword_results, "keyword_score")
    all_ids = set(v_norm) | set(k_norm)

    merged = [
        {
            "chunk_id":      cid,
            "vector_score":  v_norm.get(cid, 0),
            "keyword_score": k_norm.get(cid, 0),
            "hybrid_score":  round(0.65 * v_norm.get(cid, 0) + 0.35 * k_norm.get(cid, 0), 4),
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
            "rank_id":     i,
            "chunk_id":    c["chunk_id"],
            "chapter":     c.get("chapter"),
            "heading":     c.get("heading_or_subtopic"),
            "page_no":     c.get("page_no"),
            "text":        c.get("text", "")[:1200],
            "hybrid_score":c.get("hybrid_score", 0),
            "source":      c.get("source"),
        }
        for i, c in enumerate(chunks)
    ]

    system = """You are a strict educational RAG reranker.
Select only chunks truly related to the query.
Return confidence 0–1.
JSON only."""

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


# =========================
# PYDANTIC MODELS
# =========================

class IndexBookRequest(BaseModel):
    book_id: str


class SearchRequest(BaseModel):
    query: str
    book_id: Optional[str] = None
    top_k: int = 10
    book_name: str = ""
    universal_search: bool = False


# =========================
# API — CHUNK (sync)
# =========================

@app.post("/v1/books/chunk")
async def upload_and_chunk_book(
    book_name: str      = Form(...),
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
            "book_id": book_id, "book_name": book_name,
            "source_file": file.filename, "status": "processing",
            "created_at": now_utc(), "updated_at": now_utc(),
        })

        pages = []
        for i in range(len(pdf)):
            page      = pdf.load_page(i)
            extracted = extract_page_text(page)
            pages.append({
                "page_no":          i + 1,
                "text":             extracted["text"],
                "extraction_method":extracted["extraction_method"],
                "char_count":       len(extracted["text"]),
            })

        chunks = build_chunks(book_id=book_id, book_name=book_name,
                              filename=file.filename, pages=pages, save_to_mongo=True)

        save_book_meta({
            "book_id": book_id, "book_name": book_name,
            "source_file": file.filename, "status": "chunked",
            "total_pages": len(pdf), "total_chunks": len(chunks),
            "updated_at": now_utc(),
        })

        return JSONResponse({
            "success": True, "book_id": book_id, "book_name": book_name,
            "total_pages": len(pdf), "total_chunks": len(chunks),
            "database": MONGO_DB_NAME, "collection": CHUNKS_COLLECTION,
            "chunks": chunks,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# =========================
# API — CHUNK (stream)
# =========================

@app.post("/v1/books/chunk/stream")
async def upload_and_chunk_book_stream(
    book_name: str      = Form(...),
    file:      UploadFile = File(...),
):
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
                "success": True, "book_id": book_id,
                "book_name": book_name, "source_file": file.filename,
            })

            pdf         = fitz.open(temp_path)
            total_pages = len(pdf)
            yield sse_event("pdf_info", {"book_id": book_id, "total_pages": total_pages})

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
                    "page_no": page_no, "char_count": len(page_text),
                    "extraction_method": exm,
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

                    yield sse_event("heading_detected", {
                        "page_no": page_no, "chapter": current_chapter,
                        "heading_or_subtopic": current_heading,
                        "detection_method": detected["detection_method"],
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
                            "created_at":                   now_utc(),
                            "updated_at":                   now_utc(),
                        }
                        safe_insert_chunk(chunk)
                        yield sse_event("chunk", chunk)
                        await asyncio.sleep(0)

                yield sse_event("page_done", {
                    "page_no": page_no, "total_pages": total_pages,
                    "total_chunks_so_far": total_chunks,
                })

            yield sse_event("done", {
                "success": True, "book_id": book_id, "book_name": book_name,
                "source_file": file.filename, "total_pages": total_pages,
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
            "Cache-Control":       "no-cache",
            "Connection":          "keep-alive",
            "X-Accel-Buffering":   "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# =========================
# API — INDEX FAST
# =========================

@app.post("/v1/books/index-fast")
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
        return {
            "success": True,
            "message": "Already indexed or no chunks found",
            "book_id": payload.book_id,
        }

    total = len(chunks)
    indexed = 0
    failed_batches = 0

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        try:
            save_chunks_to_pinecone_batch(batch)
            save_chunks_to_neo4j_batch(batch)
            update_mongo_index_status_batch(batch)
            indexed += len(batch)
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

    return {
        "success":           True,
        "book_id":           payload.book_id,
        "total_chunks":      total,
        "indexed":           indexed,
        "failed_batches":    failed_batches,
        "batch_size":        BATCH_SIZE,
        "pinecone_namespace":f"book_{payload.book_id}",
    }


# =========================
# API — SEARCH
# =========================

@app.post("/v1/books/search")
async def search_book(payload: SearchRequest):
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    route = classify_user_query(payload.query)

    if route.get("route") == "chitchat" or route.get("should_search_db") is False:
        reply = generate_chitchat_reply(payload.query)

        return {
            "success": True,
            "query": payload.query,
            "route": "chitchat",
            "searched_db": False,
            "answer": reply,
            "results": []
        }

    query_plan = openai_query_planner(
        user_query=payload.query,
        book_name=payload.book_name
    )

    search_queries = list(set(
        [query_plan.get("corrected_query", payload.query)]
        + query_plan.get("semantic_queries", [])
    ))

    keyword_queries = list(set(
        [query_plan.get("main_topic", payload.query)]
        + query_plan.get("keyword_queries", [])
        + query_plan.get("must_have_terms", [])
    ))

    search_book_id = None if payload.universal_search else payload.book_id

    vector_results = vector_search(
        queries=search_queries,
        book_id=search_book_id,
        top_k=VECTOR_TOP_K
    )

    keyword_results = keyword_search(
        book_id=search_book_id,
        keyword_queries=keyword_queries
    )

    hybrid_results = merge_hybrid_results(
        vector_results=vector_results,
        keyword_results=keyword_results
    )

    if not hybrid_results:
        return {
            "success": True,
            "query": payload.query,
            "route": "rag_search",
            "searched_db": True,
            "query_plan": query_plan,
            "confidence": 0,
            "results": [],
            "message": "Sorry, I have no topic related to your search."
        }

    top_ids = [x["chunk_id"] for x in hybrid_results[:30]]
    related_ids = get_related_chunks_from_neo4j(top_ids[:10])
    final_ids = list(set(top_ids + related_ids))

    mongo_filter = {"chunk_id": {"$in": final_ids}}

    if search_book_id:
        mongo_filter["book_id"] = search_book_id

    mongo_chunks = list(
        chunks_col.find(mongo_filter, {"_id": 0})
        .sort([
            ("page_no", ASCENDING),
            ("global_paragraph_no", ASCENDING)
        ])
    )

    if not mongo_chunks:
        return {
            "success": True,
            "query": payload.query,
            "route": "rag_search",
            "searched_db": True,
            "query_plan": query_plan,
            "confidence": 0,
            "results": [],
            "message": "Sorry, I have no topic related to your search."
        }

    score_map = {x["chunk_id"]: x for x in hybrid_results}

    enriched = [
        {
            **c,
            "vector_score": score_map.get(c["chunk_id"], {}).get("vector_score", 0),
            "keyword_score": score_map.get(c["chunk_id"], {}).get("keyword_score", 0),
            "hybrid_score": score_map.get(c["chunk_id"], {}).get("hybrid_score", 0),
            "source": "direct_hybrid_match" if c["chunk_id"] in score_map else "neo4j_related_context",
        }
        for c in mongo_chunks
    ]

    diverse_chunks = sorted(
        enriched,
        key=lambda x: x.get("hybrid_score", 0),
        reverse=True
    )[:15]

    rerank_result = openai_rerank(query_plan, diverse_chunks)

    confidence = float(rerank_result.get("confidence", 0))
    selected_ids = set(rerank_result.get("selected_chunk_ids", []))

    if confidence < MIN_FINAL_CONFIDENCE or not selected_ids:
        return {
            "success": True,
            "query": payload.query,
            "route": "rag_search",
            "searched_db": True,
            "query_plan": query_plan,
            "confidence": confidence,
            "results": [],
            "message": "Sorry, I have no topic related to your search."
        }

    final_results = [
        {
            "chunk_id": c["chunk_id"],
            "book_id": c.get("book_id"),
            "book_name": c.get("book_name"),
            "page_no": c.get("page_no"),
            "chapter": c.get("chapter"),
            "heading": c.get("heading_or_subtopic"),
            "text": c.get("text"),
            "text_preview": c.get("text_preview"),
            "vector_score": c.get("vector_score"),
            "keyword_score": c.get("keyword_score"),
            "hybrid_score": c.get("hybrid_score"),
            "source": c.get("source"),
        }
        for c in diverse_chunks
        if c["chunk_id"] in selected_ids
    ]

    return {
        "success": True,
        "query": payload.query,
        "route": "rag_search",
        "searched_db": True,
        "universal_search": payload.universal_search,
        "book_id_filter": search_book_id,
        "query_plan": query_plan,
        "confidence": confidence,
        "rerank_reason": rerank_result.get("reason"),
        "pinecone_vector_candidates": len(vector_results),
        "keyword_candidates": len(keyword_results),
        "neo4j_related_chunks": len(related_ids),
        "total_context_chunks": len(final_results),
        "results": final_results
    }


# =========================
# API — GRAPH DATA  (for frontend knowledge graph)
# =========================

@app.get("/v1/books/{book_id}/graph")
async def get_book_graph(book_id: str, chunk_ids: str = ""):
    """
    Returns graph nodes + edges for visualisation.
    Optional: pass comma-separated chunk_ids to highlight matched chunks.
    """
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

            def add_node(nid: str, label: str, props: dict, ntype: str):
                if nid not in nodes:
                    nodes[nid] = {
                        "id":        nid,
                        "label":     label,
                        "type":      ntype,
                        "props":     props,
                        "highlight": nid in highlight_ids,
                    }

            def add_edge(src: str, tgt: str, rel: str):
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
                    add_node(ch_id, f"Chunk p{ch.get('page_no','')}",
                             dict(ch), "chunk")
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
                        add_node(ch_id, f"Chunk p{ch2.get('page_no','')}",
                                 dict(ch2), "chunk")
                    add_edge(pid, ch_id, "HAS_CHUNK")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j error: {e}")

    return {
        "success": True,
        "book_id": book_id,
        "nodes":   list(nodes.values()),
        "edges":   edges,
    }
    
    
    
    
def classify_user_query(user_query: str) -> Dict[str, Any]:
    system = """
You are a query router for an educational AI app.

Classify user query.

Return JSON only:
{
  "route": "chitchat|rag_search",
  "reason": "",
  "should_search_db": true/false
}

Rules:
- greetings, thanks, casual talk, general AI questions = chitchat
- questions about uploaded books, topics, lessons, MCQs, PDF content = rag_search
"""

    user = f"User query: {user_query}"
    return call_openai_json(system, user)


def generate_chitchat_reply(user_query: str) -> str:
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.4,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful educational assistant. Reply naturally and briefly. Do not mention database or documents."
            },
            {"role": "user", "content": user_query}
        ]
    )
    return resp.choices[0].message.content.strip()






















<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>GraphRAG — Knowledge Explorer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Syne:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
<style>
/* ─── RESET & TOKENS ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #09090f;
  --bg2:       #0f0f1a;
  --bg3:       #151526;
  --border:    rgba(255,255,255,0.07);
  --border-h:  rgba(255,255,255,0.14);
  --text:      #e8e8f0;
  --muted:     #6b6b8a;
  --accent:    #7c6af7;
  --accent2:   #42d4c6;
  --accent3:   #f76a8c;
  --gold:      #f5c542;
  --r:         10px;
  --r-lg:      16px;

  /* node colours */
  --book:    #7c6af7;
  --chapter: #42d4c6;
  --heading: #f76a8c;
  --page:    #f5c542;
  --chunk:   #a0a0c8;
  --chunk-h: #ff8c42;
}

html, body {
  height: 100%; width: 100%;
  background: var(--bg);
  color: var(--text);
  font-family: 'Syne', sans-serif;
  overflow: hidden;
}

/* ─── LAYOUT ──────────────────────────────────────────────────────── */
#app {
  display: grid;
  grid-template-columns: 380px 1fr;
  grid-template-rows: 56px 1fr;
  height: 100vh;
}

/* ─── TOP BAR ─────────────────────────────────────────────────────── */
#topbar {
  grid-column: 1 / -1;
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 0 24px;
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  z-index: 10;
}
#topbar .logo {
  font-size: 17px;
  font-weight: 800;
  letter-spacing: -0.5px;
  color: var(--text);
}
#topbar .logo span { color: var(--accent); }
#topbar .pill {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  padding: 3px 10px;
  border-radius: 999px;
  background: var(--bg3);
  border: 1px solid var(--border);
  color: var(--muted);
}
#topbar .spacer { flex: 1; }
#api-base-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
}
#api-base-wrap label { font-size: 12px; color: var(--muted); }
#api-base {
  font-family: 'DM Mono', monospace;
  font-size: 12px;
  background: var(--bg3);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 5px 10px;
  border-radius: 6px;
  width: 220px;
  outline: none;
}
#api-base:focus { border-color: var(--accent); }

/* ─── SIDEBAR ─────────────────────────────────────────────────────── */
#sidebar {
  background: var(--bg2);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.sb-section {
  padding: 18px 20px;
  border-bottom: 1px solid var(--border);
}
.sb-title {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 14px;
}

/* file upload */
#drop-zone {
  border: 1.5px dashed var(--border-h);
  border-radius: var(--r);
  padding: 28px 16px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
  position: relative;
}
#drop-zone:hover, #drop-zone.over {
  border-color: var(--accent);
  background: rgba(124,106,247,0.06);
}
#drop-zone input[type=file] {
  position: absolute; inset: 0; opacity: 0; cursor: pointer;
}
#drop-zone .icon {
  font-size: 28px;
  margin-bottom: 8px;
}
#drop-zone p { font-size: 12px; color: var(--muted); line-height: 1.5; }
#drop-zone p strong { color: var(--text); }
#drop-zone .fname {
  margin-top: 6px;
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--accent2);
  word-break: break-all;
}

/* form inputs */
.field { margin-bottom: 12px; }
.field label {
  display: block;
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 5px;
  font-weight: 600;
  letter-spacing: 0.5px;
}
.field input, .field textarea {
  width: 100%;
  background: var(--bg3);
  border: 1px solid var(--border);
  color: var(--text);
  font-family: 'DM Mono', monospace;
  font-size: 13px;
  padding: 8px 12px;
  border-radius: var(--r);
  outline: none;
  resize: none;
  transition: border-color 0.2s;
}
.field input:focus, .field textarea:focus {
  border-color: var(--accent);
}
.field input::placeholder, .field textarea::placeholder {
  color: var(--muted);
}

/* buttons */
.btn {
  width: 100%;
  padding: 10px 14px;
  border-radius: var(--r);
  font-family: 'Syne', sans-serif;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.3px;
  cursor: pointer;
  border: none;
  transition: opacity 0.2s, transform 0.1s;
}
.btn:active { transform: scale(0.98); }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-primary {
  background: var(--accent);
  color: #fff;
}
.btn-primary:hover:not(:disabled) { opacity: 0.85; }
.btn-secondary {
  background: var(--bg3);
  color: var(--text);
  border: 1px solid var(--border-h);
  margin-top: 8px;
}
.btn-secondary:hover:not(:disabled) { border-color: var(--accent2); color: var(--accent2); }

/* progress log */
#log-wrap {
  flex: 1;
  overflow-y: auto;
  padding: 14px 20px;
}
#log-wrap::-webkit-scrollbar { width: 4px; }
#log-wrap::-webkit-scrollbar-thumb { background: var(--border-h); border-radius: 2px; }
.log-entry {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  line-height: 1.5;
  padding: 3px 0;
  color: var(--muted);
  animation: fadeIn 0.3s ease;
}
.log-entry .dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  margin-top: 4px;
  flex-shrink: 0;
  background: var(--muted);
}
.log-entry.ok .dot   { background: var(--accent2); }
.log-entry.err .dot  { background: var(--accent3); }
.log-entry.info .dot { background: var(--accent); }
.log-entry .msg { color: var(--text); }
@keyframes fadeIn { from { opacity:0; transform: translateY(4px); } to { opacity:1; transform:none; } }

/* book/search header in sidebar */
.book-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--bg3);
  border: 1px solid var(--border-h);
  border-radius: 999px;
  padding: 4px 10px;
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--accent2);
  margin-bottom: 12px;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* ─── MAIN PANEL ──────────────────────────────────────────────────── */
#main {
  display: flex;
  flex-direction: column;
  overflow: hidden;
  position: relative;
}

/* graph */
#graph-container {
  flex: 1;
  position: relative;
  overflow: hidden;
}
#graph-svg {
  width: 100%; height: 100%;
  cursor: grab;
}
#graph-svg:active { cursor: grabbing; }

/* empty state */
#empty-state {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  pointer-events: none;
}
#empty-state .big-icon { font-size: 64px; opacity: 0.15; }
#empty-state p { color: var(--muted); font-size: 14px; max-width: 300px; text-align: center; line-height: 1.6; }

/* legend */
#legend {
  position: absolute;
  top: 16px;
  right: 16px;
  background: rgba(9,9,15,0.85);
  backdrop-filter: blur(8px);
  border: 1px solid var(--border-h);
  border-radius: var(--r);
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 7px;
  z-index: 5;
}
.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  color: var(--muted);
}
.legend-dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}

/* graph controls */
#graph-controls {
  position: absolute;
  bottom: 16px;
  right: 16px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  z-index: 5;
}
.ctrl-btn {
  width: 32px; height: 32px;
  border-radius: 8px;
  background: var(--bg2);
  border: 1px solid var(--border-h);
  color: var(--text);
  font-size: 16px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: border-color 0.2s, color 0.2s;
}
.ctrl-btn:hover { border-color: var(--accent); color: var(--accent); }

/* tooltip */
#tooltip {
  position: fixed;
  background: var(--bg2);
  border: 1px solid var(--border-h);
  border-radius: var(--r);
  padding: 12px 14px;
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  line-height: 1.7;
  max-width: 320px;
  z-index: 100;
  pointer-events: none;
  display: none;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
#tooltip .tt-type {
  font-size: 9px;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 4px;
}
#tooltip .tt-name {
  font-family: 'Syne', sans-serif;
  font-size: 13px;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 6px;
}
#tooltip .tt-row { color: var(--muted); }
#tooltip .tt-row span { color: var(--text); }

/* results panel */
#results-panel {
  height: 0;
  overflow: hidden;
  transition: height 0.35s cubic-bezier(0.4,0,0.2,1);
  border-top: 1px solid var(--border);
  background: var(--bg2);
}
#results-panel.open {
  height: 280px;
  overflow-y: auto;
}
#results-panel::-webkit-scrollbar { width: 4px; }
#results-panel::-webkit-scrollbar-thumb { background: var(--border-h); border-radius: 2px; }
#results-inner {
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.result-card {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 14px;
  transition: border-color 0.2s;
  cursor: pointer;
}
.result-card:hover, .result-card.active {
  border-color: var(--accent);
}
.rc-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}
.rc-badge {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--bg);
  border: 1px solid var(--border-h);
  color: var(--muted);
}
.rc-badge.pg  { color: var(--gold); border-color: var(--gold); }
.rc-badge.sc  { color: var(--accent2); border-color: var(--accent2); }
.rc-badge.neo { color: var(--accent3); border-color: var(--accent3); }
.rc-chapter { font-size: 11px; font-weight: 700; color: var(--accent); margin-bottom: 3px; }
.rc-heading { font-size: 12px; font-weight: 600; color: var(--text); margin-bottom: 6px; }
.rc-text {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--muted);
  line-height: 1.6;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.rc-scores {
  display: flex;
  gap: 10px;
  margin-top: 8px;
}
.rc-score-item {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  color: var(--muted);
}
.rc-score-item span { color: var(--text); }

/* stat bar */
#stat-bar {
  height: 0;
  overflow: hidden;
  transition: height 0.3s;
  background: var(--bg);
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 0 20px;
}
#stat-bar.open {
  height: 44px;
}
.stat-item {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--muted);
}
.stat-item span {
  color: var(--accent2);
  font-weight: 500;
}
.stat-sep {
  width: 1px; height: 16px;
  background: var(--border);
}
#conf-bar-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
}
#conf-label { font-family: 'DM Mono', monospace; font-size: 11px; color: var(--muted); }
#conf-bar {
  width: 100px; height: 4px;
  background: var(--bg3);
  border-radius: 2px;
  overflow: hidden;
}
#conf-fill {
  height: 100%;
  border-radius: 2px;
  background: var(--accent2);
  transition: width 0.6s ease;
}

/* overlay spinner */
#overlay {
  position: fixed; inset: 0;
  background: rgba(9,9,15,0.7);
  backdrop-filter: blur(4px);
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 200;
}
#overlay.show { display: flex; }
.spinner {
  width: 40px; height: 40px;
  border: 3px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* graph edges */
.graph-link {
  stroke: rgba(255,255,255,0.1);
  stroke-width: 1.5;
}
.graph-link.highlighted {
  stroke: var(--accent);
  stroke-width: 2.5;
  stroke-dasharray: none;
}

/* arrow markers */
</style>
</head>
<body>

<div id="overlay"><div class="spinner"></div></div>
<div id="tooltip">
  <div class="tt-type" id="tt-type"></div>
  <div class="tt-name" id="tt-name"></div>
  <div id="tt-body"></div>
</div>

<div id="app">

  <!-- ── TOP BAR ─────────────────────────── -->
  <div id="topbar">
    <div class="logo">Graph<span>RAG</span></div>
    <div class="pill">Knowledge Explorer</div>
    <div class="spacer"></div>
    <div id="api-base-wrap">
      <label>API Base</label>
      <input id="api-base" value="http://localhost:8000" placeholder="http://localhost:8000"/>
    </div>
  </div>

  <!-- ── SIDEBAR ─────────────────────────── -->
  <div id="sidebar">

    <!-- Upload -->
    <div class="sb-section">
      <div class="sb-title">📄 Upload PDF</div>
      <div id="drop-zone">
        <input type="file" id="pdf-input" accept=".pdf"/>
        <div class="icon">⬆</div>
        <p><strong>Drop PDF here</strong> or click to browse</p>
        <div class="fname" id="fname-display" style="display:none"></div>
      </div>
      <div class="field" style="margin-top:12px">
        <label>Book Name</label>
        <input id="book-name" type="text" placeholder="e.g. Class 10 Mathematics"/>
      </div>
      <button class="btn btn-primary" id="btn-chunk" disabled>⚡ Chunk & Process</button>
      <button class="btn btn-secondary" id="btn-index" disabled>🔗 Index to Pinecone + Neo4j</button>
    </div>

    <!-- Search -->
    <div class="sb-section">
      <div class="sb-title">🔍 Semantic Search</div>
      <div id="book-id-tag" class="book-tag" style="display:none">
        <span>📚</span><span id="book-id-short"></span>
      </div>
      <div class="field">
        <label>Book ID</label>
        <input id="book-id-input" type="text" placeholder="Paste book_id or auto-filled after upload"/>
      </div>
      <div class="field">
        <label>Query</label>
        <textarea id="query-input" rows="3" placeholder="e.g. Explain Pythagoras theorem with examples"></textarea>
      </div>
      <button class="btn btn-primary" id="btn-search">🔎 Search</button>
      <button class="btn btn-secondary" id="btn-graph">🕸 Load Full Graph</button>
    </div>

    <!-- Log -->
    <div id="log-wrap">
      <div class="log-entry info"><div class="dot"></div><div class="msg">Ready — upload a PDF to begin.</div></div>
    </div>

  </div>

  <!-- ── MAIN GRAPH PANEL ─────────────────── -->
  <div id="main">
    <div id="graph-container">
      <svg id="graph-svg">
        <defs>
          <marker id="arrow" viewBox="0 -5 10 10" refX="22" refY="0"
            markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,-5L10,0L0,5" fill="rgba(255,255,255,0.2)"/>
          </marker>
          <marker id="arrow-h" viewBox="0 -5 10 10" refX="22" refY="0"
            markerWidth="6" markerHeight="6" orient="auto">
            <path d="M0,-5L10,0L0,5" fill="#7c6af7"/>
          </marker>
        </defs>
        <g id="graph-g"></g>
      </svg>
      <div id="empty-state">
        <div class="big-icon">🕸</div>
        <p>Upload a PDF and process it, then click <strong>Load Full Graph</strong> or run a search to see the knowledge graph.</p>
      </div>
      <div id="legend">
        <div class="legend-item"><div class="legend-dot" style="background:var(--book)"></div> Book</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--chapter)"></div> Chapter</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--heading)"></div> Heading</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--page)"></div> Page</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--chunk)"></div> Chunk</div>
        <div class="legend-item"><div class="legend-dot" style="background:var(--chunk-h)"></div> Matched Chunk</div>
      </div>
      <div id="graph-controls">
        <button class="ctrl-btn" id="ctrl-zoom-in"  title="Zoom in">+</button>
        <button class="ctrl-btn" id="ctrl-zoom-out" title="Zoom out">−</button>
        <button class="ctrl-btn" id="ctrl-fit"      title="Fit">⊡</button>
      </div>
    </div>

    <div id="stat-bar">
      <div class="stat-item">Chunks <span id="st-chunks">0</span></div>
      <div class="stat-sep"></div>
      <div class="stat-item">Vector <span id="st-vector">0</span></div>
      <div class="stat-sep"></div>
      <div class="stat-item">Keyword <span id="st-keyword">0</span></div>
      <div class="stat-sep"></div>
      <div class="stat-item">Neo4j context <span id="st-neo">0</span></div>
      <div class="conf-bar-wrap" id="conf-bar-wrap">
        <span id="conf-label">Confidence</span>
        <div id="conf-bar"><div id="conf-fill" style="width:0%"></div></div>
        <span id="conf-val" class="stat-item" style="color:var(--accent2)">0%</span>
      </div>
    </div>

    <div id="results-panel">
      <div id="results-inner"></div>
    </div>
  </div>

</div>

<script>
/* ─────────────────────────────────────────────────────────────────────
   STATE
──────────────────────────────────────────────────────────────────── */
const state = {
  bookId:    null,
  bookName:  '',
  file:      null,
  simulation: null,
  svg:       null,
  zoom:      null,
  highlightIds: new Set(),
};

/* ─── HELPERS ─────────────────────────────────────────────────────── */
const api = () => document.getElementById('api-base').value.replace(/\/$/, '');
const $   = id => document.getElementById(id);

function log(msg, type = '') {
  const wrap = $('log-wrap');
  const el   = document.createElement('div');
  el.className = `log-entry ${type}`;
  el.innerHTML = `<div class="dot"></div><div class="msg">${msg}</div>`;
  wrap.appendChild(el);
  wrap.scrollTop = wrap.scrollHeight;
}

function overlay(show) {
  $('overlay').classList.toggle('show', show);
}

/* ─── FILE DROP ───────────────────────────────────────────────────── */
const dropZone = $('drop-zone');
const fileInput = $('pdf-input');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

function setFile(f) {
  state.file = f;
  const d = $('fname-display');
  d.textContent = f.name;
  d.style.display = 'block';
  $('btn-chunk').disabled = false;
  log(`📄 Selected: <strong>${f.name}</strong> (${(f.size/1024).toFixed(1)} KB)`, 'ok');
}

/* ─── CHUNK ───────────────────────────────────────────────────────── */
$('btn-chunk').addEventListener('click', async () => {
  const bookName = $('book-name').value.trim() || state.file.name.replace('.pdf','');
  if (!state.file) return;

  $('btn-chunk').disabled = true;
  log('⚡ Starting chunking (stream mode)…', 'info');

  const fd = new FormData();
  fd.append('file', state.file);
  fd.append('book_name', bookName);

  try {
    const resp = await fetch(`${api()}/v1/books/chunk/stream`, { method: 'POST', body: fd });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let chunkCount = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop();

      for (const part of parts) {
        if (!part.startsWith('event:')) continue;
        const [evLine, dataLine] = part.split('\n');
        const evName = evLine.replace('event: ','').trim();
        let data = {};
        try { data = JSON.parse(dataLine.replace('data: ','')); } catch {}

        if (evName === 'start') {
          state.bookId   = data.book_id;
          state.bookName = data.book_name;
          setBookUI(data.book_id, data.book_name);
          log(`📦 book_id: <strong>${data.book_id}</strong>`, 'info');
        }
        if (evName === 'pdf_info') {
          log(`📋 ${data.total_pages} pages detected`, 'ok');
        }
        if (evName === 'page_extracted') {
          log(`Page ${data.page_no} → ${data.char_count} chars [${data.extraction_method}]`);
        }
        if (evName === 'chunk') {
          chunkCount++;
        }
        if (evName === 'done') {
          log(`✅ Done! ${data.total_chunks} chunks from ${data.total_pages} pages`, 'ok');
          $('btn-index').disabled = false;
          $('btn-chunk').disabled = false;
        }
        if (evName === 'error') {
          log(`❌ ${data.message}`, 'err');
          $('btn-chunk').disabled = false;
        }
      }
    }
  } catch(e) {
    log(`❌ ${e.message}`, 'err');
    $('btn-chunk').disabled = false;
  }
});

/* ─── INDEX ───────────────────────────────────────────────────────── */
$('btn-index').addEventListener('click', async () => {
  if (!state.bookId) return;
  overlay(true);
  log('🔗 Indexing to Pinecone + Neo4j…', 'info');
  try {
    const resp = await fetch(`${api()}/v1/books/index-fast`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ book_id: state.bookId }),
    });
    const data = await resp.json();
    if (data.success) {
      log(`✅ Indexed ${data.indexed}/${data.total_chunks} chunks. Failed batches: ${data.failed_batches}`, 'ok');
    } else {
      log(`❌ ${JSON.stringify(data)}`, 'err');
    }
  } catch(e) {
    log(`❌ ${e.message}`, 'err');
  } finally {
    overlay(false);
  }
});

/* ─── SEARCH ──────────────────────────────────────────────────────── */
$('btn-search').addEventListener('click', async () => {
  const bookId = $('book-id-input').value.trim() || state.bookId;
  const query  = $('query-input').value.trim();
  if (!bookId) { log('❌ Enter a book_id first', 'err'); return; }
  if (!query)  { log('❌ Enter a search query', 'err'); return; }

  overlay(true);
  log(`🔎 Searching: "${query}"…`, 'info');

  try {
    const resp = await fetch(`${api()}/v1/books/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ book_id: bookId, query, book_name: state.bookName }),
    });
    const data = await resp.json();

    renderStats(data);
    renderResults(data.results || []);

    const matchIds = (data.results || []).map(r => r.chunk_id);
    state.highlightIds = new Set(matchIds);
    log(`✅ Confidence ${(data.confidence*100).toFixed(0)}% — ${(data.results||[]).length} results`, 'ok');

    // load graph with highlights
    if (bookId) {
      await loadGraph(bookId, matchIds.join(','));
    }
  } catch(e) {
    log(`❌ ${e.message}`, 'err');
  } finally {
    overlay(false);
  }
});

/* ─── GRAPH LOAD ──────────────────────────────────────────────────── */
$('btn-graph').addEventListener('click', async () => {
  const bookId = $('book-id-input').value.trim() || state.bookId;
  if (!bookId) { log('❌ Enter a book_id first', 'err'); return; }
  overlay(true);
  await loadGraph(bookId, '');
  overlay(false);
});

async function loadGraph(bookId, chunkIds = '') {
  log(`🕸 Loading knowledge graph…`, 'info');
  try {
    const url = `${api()}/v1/books/${bookId}/graph${chunkIds ? '?chunk_ids='+chunkIds : ''}`;
    const resp = await fetch(url);
    const data = await resp.json();
    if (!data.success) throw new Error(JSON.stringify(data));
    log(`Graph: ${data.nodes.length} nodes, ${data.edges.length} edges`, 'ok');
    $('empty-state').style.display = 'none';
    renderGraph(data.nodes, data.edges);
  } catch(e) {
    log(`❌ Graph error: ${e.message}`, 'err');
  }
}

/* ─── STATS ───────────────────────────────────────────────────────── */
function renderStats(data) {
  $('stat-bar').classList.add('open');
  $('st-chunks').textContent  = data.total_context_chunks || 0;
  $('st-vector').textContent  = data.pinecone_vector_candidates || 0;
  $('st-keyword').textContent = data.keyword_candidates || 0;
  $('st-neo').textContent     = data.neo4j_related_chunks || 0;
  const pct = Math.round((data.confidence || 0) * 100);
  $('conf-fill').style.width  = pct + '%';
  $('conf-val').textContent   = pct + '%';
  $('conf-fill').style.background = pct > 70 ? 'var(--accent2)' : pct > 45 ? 'var(--gold)' : 'var(--accent3)';
}

/* ─── RESULTS ─────────────────────────────────────────────────────── */
function renderResults(results) {
  const panel = $('results-panel');
  const inner = $('results-inner');
  inner.innerHTML = '';

  if (!results.length) {
    panel.classList.remove('open');
    return;
  }
  panel.classList.add('open');

  results.forEach((r, i) => {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.dataset.chunkId = r.chunk_id;
    const sourceClass = r.source === 'neo4j_related_context' ? 'neo' : 'sc';
    const sourceLabel = r.source === 'neo4j_related_context' ? 'Neo4j' : 'Hybrid';
    card.innerHTML = `
      <div class="rc-meta">
        <span class="rc-badge pg">Page ${r.page_no}</span>
        <span class="rc-badge ${sourceClass}">${sourceLabel}</span>
        <span class="rc-badge">H: ${((r.hybrid_score||0)*100).toFixed(0)}%</span>
      </div>
      <div class="rc-chapter">${r.chapter||'—'}</div>
      <div class="rc-heading">${r.heading||'—'}</div>
      <div class="rc-text">${r.text||''}</div>
      <div class="rc-scores">
        <div class="rc-score-item">Vector <span>${((r.vector_score||0)*100).toFixed(0)}%</span></div>
        <div class="rc-score-item">Keyword <span>${((r.keyword_score||0)*100).toFixed(0)}%</span></div>
        <div class="rc-score-item">Hybrid <span>${((r.hybrid_score||0)*100).toFixed(0)}%</span></div>
      </div>
    `;
    card.addEventListener('click', () => {
      document.querySelectorAll('.result-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      highlightNodeInGraph(r.chunk_id);
    });
    inner.appendChild(card);
  });
}

/* ─── D3 GRAPH ────────────────────────────────────────────────────── */
const NODE_RADIUS = { book: 22, chapter: 16, heading: 12, page: 11, chunk: 7 };
const NODE_COLOR  = {
  book:    'var(--book)',
  chapter: 'var(--chapter)',
  heading: 'var(--heading)',
  page:    'var(--page)',
  chunk:   'var(--chunk)',
};

let d3Zoom, gEl;

function renderGraph(nodes, edges) {
  const svg  = d3.select('#graph-svg');
  const g    = d3.select('#graph-g');
  g.selectAll('*').remove();

  const W = $('graph-container').clientWidth;
  const H = $('graph-container').clientHeight;

  // Build adjacency for quick lookup
  const nodeMap = Object.fromEntries(nodes.map(n => [n.id, n]));

  const links = edges
    .filter(e => nodeMap[e.source] && nodeMap[e.target])
    .map(e => ({ ...e, source: e.source, target: e.target }));

  // Simulation
  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(d => {
      const types = [d.source.type, d.target.type].sort().join('-');
      const dist = { 'book-chapter':80, 'chapter-heading':70, 'heading-chunk':50, 'book-page':90, 'chunk-page':45 };
      return dist[types] || 60;
    }).strength(0.7))
    .force('charge', d3.forceManyBody().strength(d => d.type === 'book' ? -600 : d.type === 'chunk' ? -30 : -150))
    .force('center', d3.forceCenter(W/2, H/2))
    .force('collision', d3.forceCollide().radius(d => (NODE_RADIUS[d.type]||8) + 6));

  state.simulation = sim;

  // Zoom
  d3Zoom = d3.zoom()
    .scaleExtent([0.05, 4])
    .on('zoom', e => g.attr('transform', e.transform));

  svg.call(d3Zoom);
  gEl = g.node();

  // Links
  const link = g.append('g').attr('class','links')
    .selectAll('line')
    .data(links)
    .enter().append('line')
    .attr('class','graph-link')
    .attr('marker-end', d => state.highlightIds.size ? 'url(#arrow)' : 'url(#arrow)');

  // Node groups
  const node = g.append('g').attr('class','nodes')
    .selectAll('g')
    .data(nodes)
    .enter().append('g')
    .attr('class','graph-node')
    .call(d3.drag()
      .on('start', (e,d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag',  (e,d) => { d.fx=e.x; d.fy=e.y; })
      .on('end',   (e,d) => { if (!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; })
    )
    .on('mouseover', showTooltip)
    .on('mousemove', moveTooltip)
    .on('mouseout',  hideTooltip)
    .on('click', (e,d) => {
      if (d.type === 'chunk') highlightResultCard(d.id);
    });

  // Circle
  node.append('circle')
    .attr('r', d => NODE_RADIUS[d.type] || 8)
    .attr('fill', d => d.highlight ? 'var(--chunk-h)' : (NODE_COLOR[d.type] || '#888'))
    .attr('fill-opacity', d => d.type === 'chunk' ? 0.85 : 1)
    .attr('stroke', d => d.highlight ? '#fff' : 'rgba(255,255,255,0.2)')
    .attr('stroke-width', d => d.highlight ? 2.5 : 1)
    .style('filter', d => d.highlight ? 'drop-shadow(0 0 6px var(--chunk-h))' : '');

  // Glow pulse on highlighted nodes
  node.filter(d => d.highlight)
    .append('circle')
    .attr('r', d => (NODE_RADIUS[d.type]||8) + 6)
    .attr('fill', 'none')
    .attr('stroke', 'var(--chunk-h)')
    .attr('stroke-width', 1)
    .attr('stroke-opacity', 0.4);

  // Label — only for non-chunk types
  node.filter(d => d.type !== 'chunk')
    .append('text')
    .attr('dy', d => (NODE_RADIUS[d.type]||8) + 13)
    .attr('text-anchor','middle')
    .attr('font-family','Syne, sans-serif')
    .attr('font-size', d => d.type === 'book' ? 13 : d.type === 'chapter' ? 11 : 9)
    .attr('font-weight', d => d.type === 'book' ? 800 : 600)
    .attr('fill', d => NODE_COLOR[d.type] || '#aaa')
    .attr('fill-opacity', 0.9)
    .text(d => truncate(d.label, d.type === 'book' ? 30 : d.type === 'chapter' ? 22 : 18));

  sim.on('tick', () => {
    link
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  // Fit after sim settles
  setTimeout(() => fitGraph(W, H), 1200);
}

function truncate(s, n) {
  return s && s.length > n ? s.slice(0, n) + '…' : (s || '');
}

function fitGraph(W, H) {
  if (!gEl || !d3Zoom) return;
  const bounds = gEl.getBBox();
  if (!bounds.width || !bounds.height) return;
  const svg = d3.select('#graph-svg');
  const scale = Math.min(0.9, Math.min(W / bounds.width, H / bounds.height));
  const tx = W/2 - scale*(bounds.x + bounds.width/2);
  const ty = H/2 - scale*(bounds.y + bounds.height/2);
  svg.transition().duration(600)
    .call(d3Zoom.transform, d3.zoomIdentity.translate(tx,ty).scale(scale));
}

/* ─── GRAPH CONTROLS ──────────────────────────────────────────────── */
$('ctrl-zoom-in').addEventListener('click', () => {
  const svg = d3.select('#graph-svg');
  svg.transition().duration(250).call(d3Zoom.scaleBy, 1.4);
});
$('ctrl-zoom-out').addEventListener('click', () => {
  const svg = d3.select('#graph-svg');
  svg.transition().duration(250).call(d3Zoom.scaleBy, 0.7);
});
$('ctrl-fit').addEventListener('click', () => {
  const W = $('graph-container').clientWidth;
  const H = $('graph-container').clientHeight;
  fitGraph(W, H);
});

/* ─── TOOLTIP ─────────────────────────────────────────────────────── */
function showTooltip(e, d) {
  const tt = $('tooltip');
  $('tt-type').textContent = d.type.toUpperCase();
  $('tt-name').textContent = d.label;
  let body = '';
  if (d.type === 'chunk') {
    body += `<div class="tt-row">Page <span>${d.props.page_no||'—'}</span></div>`;
    body += `<div class="tt-row">Words <span>${d.props.word_count||'—'}</span></div>`;
    if (d.props.text_preview) {
      body += `<div class="tt-row" style="margin-top:6px;color:var(--muted)">${d.props.text_preview.slice(0,120)}…</div>`;
    }
    if (d.highlight) {
      body += `<div class="tt-row" style="color:var(--chunk-h);margin-top:4px">🎯 Matched result</div>`;
    }
  } else if (d.type === 'page') {
    body += `<div class="tt-row">Page No <span>${d.props.page_no||'—'}</span></div>`;
  } else if (d.type === 'chapter') {
    body += `<div class="tt-row">Chapter</div>`;
  }
  $('tt-body').innerHTML = body;
  tt.style.display = 'block';
  moveTooltip(e);
}

function moveTooltip(e) {
  const tt = $('tooltip');
  const pad = 12;
  let left = e.clientX + pad;
  let top  = e.clientY + pad;
  if (left + 330 > window.innerWidth)  left = e.clientX - 330 - pad;
  if (top  + 200 > window.innerHeight) top  = e.clientY - 200 - pad;
  tt.style.left = left + 'px';
  tt.style.top  = top  + 'px';
}

function hideTooltip() {
  $('tooltip').style.display = 'none';
}

/* ─── CROSS-LINK: graph ↔ results ─────────────────────────────────── */
function highlightNodeInGraph(chunkId) {
  d3.selectAll('.graph-node circle')
    .filter((d,i,els) => {
      return d.id === chunkId;
    })
    .attr('stroke','#fff')
    .attr('stroke-width', 3);
}

function highlightResultCard(chunkId) {
  const cards = document.querySelectorAll('.result-card');
  cards.forEach(c => {
    c.classList.toggle('active', c.dataset.chunkId === chunkId);
  });
  const active = document.querySelector(`.result-card[data-chunk-id="${chunkId}"]`);
  if (active) active.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ─── BOOK UI ─────────────────────────────────────────────────────── */
function setBookUI(bookId, bookName) {
  $('book-id-input').value = bookId;
  $('book-id-short').textContent = bookId.slice(0, 12) + '… — ' + (bookName || '');
  $('book-id-tag').style.display = 'inline-flex';
}

/* ─── RESIZE ──────────────────────────────────────────────────────── */
window.addEventListener('resize', () => {
  if (!gEl) return;
  const W = $('graph-container').clientWidth;
  const H = $('graph-container').clientHeight;
  fitGraph(W, H);
});
</script>
</body>
</html>












