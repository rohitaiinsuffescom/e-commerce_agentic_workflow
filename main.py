# import uuid
# import json
# import os
# import asyncio
# import httpx
# from datetime import datetime
# from typing import TypedDict, Optional, List, Dict, Any

# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.responses import HTMLResponse, JSONResponse
# from pydantic import BaseModel, Field

# # MongoDB / Beanie
# from beanie import Document, init_beanie
# from motor.motor_asyncio import AsyncIOMotorClient
# from pymongo import IndexModel, ASCENDING

# # ChromaDB
# import chromadb
# from chromadb.utils.embedding_functions import EmbeddingFunction

# # Groq
# from groq import AsyncGroq
# from sentence_transformers import SentenceTransformer
# import numpy as np

# # Load a pre-trained Sentence Transformer model
# st_model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight, fast model

# # Custom embedding function for ChromaDB
# def sentence_transformer_embed(texts: list[str]) -> list[list[float]]:
#     embeddings = st_model.encode(texts, convert_to_numpy=True)
#     return embeddings.tolist()
# # ==========================
# # 🔹 CONFIG
# # ==========================
# MONGO_URI      = os.getenv("MONGO_URI", "mongodb://localhost:27017")
# DB_NAME        = "ai_support"
# GROQ_API_KEY   = ''
# DUMMYJSON_BASE = "https://dummyjson.com"


# CATEGORIES = [
#     "beauty","fragrances","furniture","groceries","home-decoration",
#     "kitchen-accessories","laptops","mens-shirts","mens-shoes","mens-watches",
#     "mobile-accessories","motorcycle","skin-care","smartphones","sports-accessories",
#     "sunglasses","tablets","tops","vehicle","womens-bags","womens-dresses",
#     "womens-jewellery","womens-shoes","womens-watches"
# ]

# app    = FastAPI(title="Enterprise E-commerce AI Agent")
# groq   = AsyncGroq(api_key=GROQ_API_KEY)



# # ==========================
# # 🔹 CHROMADB + SENTENCE TRANSFORMER (FIXED)
# # ==========================

# # Custom embedding class (REQUIRED by Chroma)
# class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
#     def __init__(self, model):
#         self.model = model

#     def __call__(self, texts: List[str]) -> List[List[float]]:
#         # Safety: handle empty input
#         if not texts:
#             return []

#         embeddings = self.model.encode(
#             texts,
#             convert_to_numpy=True,
#             normalize_embeddings=True   # improves similarity search
#         )
#         return embeddings.tolist()

# # Initialize Chroma client
# chroma_client = chromadb.Client()

# # Initialize custom embedding function
# st_embed_fn = SentenceTransformerEmbeddingFunction(st_model)

# # Create / get collection with custom embeddings
# company_collection = chroma_client.get_or_create_collection(
#     name="company_knowledge",
#     embedding_function=st_embed_fn
# )

# # Optional debug (run once to verify)
# try:
#     test_vec = st_embed_fn(["test embedding"])
#     print(f"[DEBUG] Embedding dimension: {len(test_vec[0])}")
# except Exception as e:
#     print("[ERROR] Embedding test failed:", e)

# # Active WebSocket connections {session_id: websocket}
# active_connections: Dict[str, WebSocket] = {}

# # ==========================
# # 🔹 DB MODELS
# # ==========================
# class ChatMessage(Document):
#     session_id: str
#     role: str
#     message: str
#     timestamp: datetime = Field(default_factory=datetime.utcnow)

#     class Settings:
#         name = "chat_messages"
#         indexes = [
#             IndexModel([("session_id", ASCENDING)]),
#             IndexModel([("timestamp", ASCENDING)], expireAfterSeconds=2592000),
#         ]

# class ChatSummary(Document):
#     session_id: str
#     summary: str = ""
#     message_count: int = 0
#     last_updated: datetime = Field(default_factory=datetime.utcnow)

#     class Settings:
#         name = "chat_summaries"
#         indexes = [IndexModel([("session_id", ASCENDING)], unique=True)]

# class SessionProduct(Document):
#     session_id: str
#     product_id: str
#     product_name: str
#     category: str = ""
#     added_at: datetime = Field(default_factory=datetime.utcnow)

#     class Settings:
#         name = "session_products"
#         indexes = [IndexModel([("session_id", ASCENDING)])]

# class CartItem(Document):
#     session_id: str
#     product_id: str
#     product_name: str
#     price: float
#     quantity: int = 1
#     thumbnail: str = ""
#     added_at: datetime = Field(default_factory=datetime.utcnow)

#     class Settings:
#         name = "cart_items"
#         indexes = [IndexModel([("session_id", ASCENDING)])]

# class AdminProduct(Document):
#     product_name: str
#     description: str
#     price: float
#     category: str
#     stock: int = 0
#     created_at: datetime = Field(default_factory=datetime.utcnow)

#     class Settings:
#         name = "admin_products"

# # ==========================
# # 🔹 STARTUP
# # ==========================
# @app.on_event("startup")
# async def init_db():
#     client = AsyncIOMotorClient(MONGO_URI)
#     await init_beanie(
#         database=client[DB_NAME],
#         document_models=[ChatMessage, ChatSummary, SessionProduct, CartItem, AdminProduct]
#     )

# # ==========================
# # 🔹 LLM CALLER
# # ==========================
# async def call_llm(messages: List[Dict], temperature: float = 0.3) -> str:
#     resp = await groq.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=messages,
#         temperature=temperature,
#     )
#     return resp.choices[0].message.content

# # ==========================
# # 🔹 MEMORY
# # ==========================
# async def save_message(session_id: str, role: str, message: str):
#     await ChatMessage(session_id=session_id, role=role, message=message).insert()
#     count = await ChatMessage.find(ChatMessage.session_id == session_id).count()
#     if count > 0 and count % 10 == 0:   # summary every 10 msgs for testing
#         await _generate_and_save_summary(session_id)

# async def _generate_and_save_summary(session_id: str):
#     msgs = await ChatMessage.find(
#         ChatMessage.session_id == session_id
#     ).sort("timestamp").to_list()
#     if not msgs:
#         return
#     chat_text = "\n".join(
#         f"[{m.timestamp.strftime('%H:%M:%S')}] {m.role}: {m.message}" for m in msgs
#     )
#     system_prompt = (
#         "You are an e-commerce support summarizer. In 3-5 sentences summarize: "
#         "products asked, order issues, sentiment, resolutions. Be concise."
#     )
#     summary_text = await call_llm([
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": f"Chat:\n{chat_text}"}
#     ])
#     existing = await ChatSummary.find_one({"session_id": session_id})
#     if existing:
#         existing.summary = summary_text
#         existing.message_count = len(msgs)
#         existing.last_updated = datetime.utcnow()
#         await existing.save()
#     else:
#         await ChatSummary(
#             session_id=session_id,
#             summary=summary_text,
#             message_count=len(msgs)
#         ).insert()

# async def get_chat_context(session_id: str) -> str:
#     summary_doc = await ChatSummary.find_one({"session_id": session_id})
#     summary = summary_doc.summary if summary_doc else "No summary yet."
#     recent = await ChatMessage.find(
#         ChatMessage.session_id == session_id
#     ).sort("-timestamp").limit(12).to_list()
#     lines = [
#         f"[{m.timestamp.strftime('%H:%M')}] {m.role}: {m.message}"
#         for m in reversed(recent)
#     ]
#     return f"SUMMARY:\n{summary}\n\nRECENT:\n" + "\n".join(lines)

# # ==========================
# # 🔹 DUMMYJSON API TOOLS
# # ==========================
# async def detect_category(user_input: str) -> Optional[str]:
#     """Use LLM to map user query to a DummyJSON category."""
#     cats = ", ".join(CATEGORIES)
#     prompt = (
#         f"User said: \"{user_input}\"\n"
#         f"Available categories: {cats}\n"
#         "Return ONLY the single best matching category slug, or 'none' if no match."
#     )
#     result = await call_llm([
#         {"role": "system", "content": "You are a category classifier. Return only the slug string."},
#         {"role": "user", "content": prompt}
#     ], temperature=0.0)
#     result = result.strip().lower().strip('"').strip("'")
#     return result if result in CATEGORIES else None

# async def fetch_products_by_category(category: str, limit: int = 5) -> List[Dict]:
#     async with httpx.AsyncClient() as client:
#         resp = await client.get(f"{DUMMYJSON_BASE}/products/category/{category}?limit={limit}")
#         data = resp.json()
#         return data.get("products", [])

# async def search_products(query: str, limit: int = 5) -> List[Dict]:
#     async with httpx.AsyncClient() as client:
#         resp = await client.get(f"{DUMMYJSON_BASE}/products/search?q={query}&limit={limit}")
#         data = resp.json()
#         return data.get("products", [])

# async def get_product_by_id(product_id: str) -> Optional[Dict]:
#     async with httpx.AsyncClient() as client:
#         resp = await client.get(f"{DUMMYJSON_BASE}/products/{product_id}")
#         if resp.status_code == 200:
#             return resp.json()
#     return None

# # ==========================
# # 🔹 COMPANY KNOWLEDGE (ChromaDB)
# # ==========================
# def search_company_knowledge(query: str, n=3) -> str:
#     try:
#         results = company_collection.query(query_texts=[query], n_results=n)
#         docs = results.get("documents", [[]])[0]
#         return "\n".join(docs) if docs else ""
#     except Exception:
#         return ""

# # ==========================
# # 🔹 CART OPERATIONS
# # ==========================
# async def add_to_cart(session_id: str, product: Dict) -> str:
#     existing = await CartItem.find_one({
#         "session_id": session_id,
#         "product_id": str(product["id"])
#     })
#     if existing:
#         existing.quantity += 1
#         await existing.save()
#         return f"Updated cart: {product['title']} (qty: {existing.quantity})"
#     else:
#         await CartItem(
#             session_id=session_id,
#             product_id=str(product["id"]),
#             product_name=product["title"],
#             price=product["price"],
#             thumbnail=product.get("thumbnail", ""),
#         ).insert()
#         return f"Added to cart: {product['title']} @ ${product['price']}"

# async def get_cart(session_id: str) -> List[CartItem]:
#     return await CartItem.find(CartItem.session_id == session_id).to_list()

# # ==========================
# # 🔹 SESSION PRODUCT TABLE
# # ==========================
# async def save_session_product(session_id: str, product_id: str, product_name: str, category: str = ""):
#     exists = await SessionProduct.find_one({
#         "session_id": session_id, "product_id": product_id
#     })
#     if not exists:
#         await SessionProduct(
#             session_id=session_id,
#             product_id=product_id,
#             product_name=product_name,
#             category=category
#         ).insert()

# async def get_session_products(session_id: str) -> List[SessionProduct]:
#     return await SessionProduct.find(SessionProduct.session_id == session_id).to_list()

# # ==========================
# # 🔹 PRODUCTION DECISION ENGINE
# # ==========================
# async def decision_engine(session_id: str, user_input: str) -> Dict:
#     context = await get_chat_context(session_id)
#     session_prods = await get_session_products(session_id)
#     cart = await get_cart(session_id)

#     prod_table = ""
#     if session_prods:
#         prod_table = "Session Products: " + ", ".join(
#             f"{p.product_name}(ID:{p.product_id})" for p in session_prods
#         )
#     cart_info = f"Cart items: {len(cart)}" if cart else "Cart: empty"

#     system_prompt = """You are a production-grade e-commerce intent classifier.
# Analyze the user message and return ONLY valid JSON with these exact fields:

# {
#   "intent": one of [
#     "product_search",      // user wants to find/browse products
#     "product_detail",      // user wants details about a specific product
#     "add_to_cart",         // user wants to add something to cart
#     "view_cart",           // user wants to see their cart
#     "order_status",        // user asking about their order
#     "order_delay",         // user asking why order is delayed
#     "refund_return",       // refund or return request
#     "complaint",           // general complaint
#     "company_info",        // asking about the company, policies, etc.
#     "escalate",            // very angry, demands human, or complex unresolved issue
#     "off_topic",           // NOT related to e-commerce at all
#     "greeting"             // hello, hi, thanks, bye
#   ],
#   "sentiment": "positive" | "neutral" | "frustrated" | "angry" | "very_angry",
#   "needs_product_id": true | false,   // true if we need to ask for product ID
#   "category_hint": "category slug or empty string",
#   "add_to_cart_trigger": true | false,
#   "confidence": 0.0 to 1.0,
#   "escalate_now": true | false        // set true if sentiment is very_angry or angry + repeated issue
# }

# Rules:
# - If user asks about anything NOT related to shopping, products, orders, returns → intent = "off_topic"
# - If user is very_angry OR keeps repeating the same issue → escalate_now = true
# - needs_product_id = true only when user references a specific product without giving ID
# - Be strict: this is an e-commerce bot only

# Important 
# -always remebber chat history what going on in chat
# -if user asked about product like tell me about smartphone any thing serach via apis get data 
# -if user say tell me about my product then tell hime product
# -

# """

#     user_prompt = f"""CONTEXT:
# {context}

# {prod_table}
# {cart_info}

# USER MESSAGE: {user_input}"""

#     raw = await call_llm([
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt}
#     ], temperature=0.0)

#     try:
#         # Strip markdown fences if any
#         clean = raw.strip().strip("```json").strip("```").strip()
#         return json.loads(clean)
#     except Exception:
#         return {
#             "intent": "complaint",
#             "sentiment": "neutral",
#             "needs_product_id": False,
#             "category_hint": "",
#             "add_to_cart_trigger": False,
#             "confidence": 0.5,
#             "escalate_now": False
#         }

# # ==========================
# # 🔹 RESPONSE GENERATOR
# # ==========================
# async def generate_response(session_id: str, user_input: str, decision: Dict, products: List[Dict] = None) -> str:
#     context = await get_chat_context(session_id)
#     prod_str = json.dumps(products, indent=2) if products else "No products found."
#     company_info = search_company_knowledge(user_input)

#     system_prompt = """You are a professional e-commerce support agent.
# Rules:
# - ONLY answer questions related to products, orders, shipping, returns, company info.
# - If off-topic: politely redirect to e-commerce topics ONLY.
# - Be warm, concise, and helpful.
# - For product queries, present products clearly with name, price, ID, description.
# - For order delays: apologize, give standard ETA policy, offer to escalate if needed.
# - NEVER make up product data. Use only provided data.
# - End with a relevant follow-up question or offer."""

#     user_prompt = f"""CONTEXT:
# {context}

# COMPANY KNOWLEDGE:
# {company_info if company_info else "N/A"}

# PRODUCT DATA:
# {prod_str}

# USER: {user_input}
# INTENT: {decision.get('intent')}
# SENTIMENT: {decision.get('sentiment')}

# Provide a helpful, professional response:"""

#     return await call_llm([
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt}
#     ], temperature=0.4)

# # ==========================
# # 🔹 WEBSOCKET — CHAT
# # ==========================
# @app.websocket("/ws/chat")
# async def chat_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     session_id = str(uuid.uuid4())
#     active_connections[session_id] = websocket

#     await websocket.send_json({
#         "type": "connected",
#         "session_id": session_id,
#         "message": "👋 Welcome! I'm your AI shopping assistant. How can I help you today?"
#     })

#     # Track state for multi-turn product ID collection
#     pending_product_request: Optional[Dict] = None

#     try:
#         while True:
#             data = await websocket.receive_json()
#             user_input = data.get("message", "").strip()
#             if not user_input:
#                 continue

#             await save_message(session_id, "user", user_input)

#             # ── If we were waiting for product ID/name ──
#             if pending_product_request and pending_product_request.get("awaiting") == "product_id":
#                 product_id = user_input.strip()
#                 product = await get_product_by_id(product_id)
#                 if product:
#                     await save_session_product(session_id, str(product["id"]), product["title"], product.get("category",""))
#                     prods = await get_session_products(session_id)
#                     response_text = f"✅ Got it! Here are details for **{product['title']}** (ID: {product['id']}):\n\n💰 Price: ${product['price']}\n📦 Stock: {product.get('stock', 'N/A')}\n📝 {product.get('description','')}\n\nWould you like to add this to your cart?"
#                     pending_product_request = None
#                 else:
#                     response_text = f"❌ I couldn't find a product with ID **{product_id}**. Please double-check and try again."
#                 await save_message(session_id, "assistant", response_text)
#                 prods = await get_session_products(session_id)
#                 await websocket.send_json({
#                     "type": "response",
#                     "response": response_text,
#                     "session_products": [{"id": p.product_id, "name": p.product_name, "category": p.category} for p in prods]
#                 })
#                 continue

#             # ── Intent Decision ──
#             await websocket.send_json({"type": "status", "message": "🤔 Analyzing your query..."})
#             decision = await decision_engine(session_id, user_input)
#             intent = decision.get("intent", "complaint")
#             sentiment = decision.get("sentiment", "neutral")

#             # ── Escalate to Human ──
#             if decision.get("escalate_now") or sentiment == "very_angry":
#                 response_text = (
#                     "🔴 I understand you're frustrated. I'm connecting you to a **Human Agent** right now.\n\n"
#                     "Please hold on — a support specialist will join this chat shortly."
#                 )
#                 await save_message(session_id, "assistant", response_text)
#                 await websocket.send_json({
#                     "type": "human_takeover",
#                     "response": response_text,
#                     "session_id": session_id
#                 })
#                 continue

#             # ── Off-topic ──
#             if intent == "off_topic":
#                 response_text = (
#                     "I'm your dedicated **shopping assistant** and can only help with:\n"
#                     "🛍️ Products & browsing • 📦 Orders & shipping • 🔄 Returns & refunds • 🏢 Company info\n\n"
#                     "What can I help you shop for today?"
#                 )
#                 await save_message(session_id, "assistant", response_text)
#                 await websocket.send_json({"type": "response", "response": response_text})
#                 continue

#             # ── Greeting ──
#             if intent == "greeting":
#                 response_text = await call_llm([
#                     {"role": "system", "content": "You are a friendly e-commerce support agent. Keep it brief and warm. Mention you can help with products, orders, returns."},
#                     {"role": "user", "content": user_input}
#                 ], temperature=0.5)
#                 await save_message(session_id, "assistant", response_text)
#                 await websocket.send_json({"type": "response", "response": response_text})
#                 continue

#             # ── View Cart ──
#             if intent == "view_cart":
#                 cart = await get_cart(session_id)
#                 if not cart:
#                     response_text = "🛒 Your cart is empty. Browse our products and add something you like!"
#                 else:
#                     total = sum(c.price * c.quantity for c in cart)
#                     items = "\n".join([f"• {c.product_name} × {c.quantity} — ${c.price}" for c in cart])
#                     response_text = f"🛒 **Your Cart:**\n\n{items}\n\n💰 **Total: ${total:.2f}**\n\nReady to checkout?"
#                 await save_message(session_id, "assistant", response_text)
#                 await websocket.send_json({"type": "response", "response": response_text, "cart": [serialize_doc(c) for c in cart]})
#                 continue

#             # ── Company Info ──
#             if intent == "company_info":
#                 company_info = search_company_knowledge(user_input)
#                 if company_info:
#                     response_text = await generate_response(session_id, user_input, decision)
#                 else:
#                     response_text = (
#                         "We are a premium e-commerce platform offering a wide range of products. "
#                         "Our policies include 30-day returns, free shipping on orders over $50, and 24/7 support. "
#                         "Is there anything specific you'd like to know?"
#                     )
#                 await save_message(session_id, "assistant", response_text)
#                 await websocket.send_json({"type": "response", "response": response_text})
#                 continue

#             # ── Product Search / Detail ──
#             if intent in ["product_search", "product_detail"]:
#                 await websocket.send_json({"type": "status", "message": "🔍 Searching products..."})

#                 category = decision.get("category_hint", "").strip()
#                 if not category:
#                     category = await detect_category(user_input)

#                 products = []
#                 if category:
#                     products = await fetch_products_by_category(category, limit=5)
#                 if not products:
#                     products = await search_products(user_input, limit=5)

#                 # Save session products
#                 for p in products[:3]:
#                     await save_session_product(session_id, str(p["id"]), p["title"], p.get("category",""))

#                 response_text = await generate_response(session_id, user_input, decision, products)

#                 prods = await get_session_products(session_id)
#                 await save_message(session_id, "assistant", response_text)
#                 await websocket.send_json({
#                     "type": "response",
#                     "response": response_text,
#                     "products": products[:5],
#                     "session_products": [{"id": p.product_id, "name": p.product_name, "category": p.category} for p in prods]
#                 })
#                 continue

#             # ── Add to Cart ──
#             if intent == "add_to_cart" or decision.get("add_to_cart_trigger"):
#                 await websocket.send_json({"type": "status", "message": "🛒 Processing cart..."})
#                 category = decision.get("category_hint", "")
#                 if not category:
#                     category = await detect_category(user_input)
#                 products = []
#                 if category:
#                     products = await fetch_products_by_category(category, limit=3)
#                 if not products:
#                     products = await search_products(user_input, limit=3)

#                 if products:
#                     p = products[0]
#                     msg = await add_to_cart(session_id, p)
#                     await save_session_product(session_id, str(p["id"]), p["title"], p.get("category",""))
#                     response_text = (
#                         f"✅ {msg}\n\n"
#                         f"🖼️ Product: **{p['title']}**\n"
#                         f"💰 Price: ${p['price']}\n"
#                         f"🔗 [View Product]({DUMMYJSON_BASE}/products/{p['id']})\n\n"
#                         "Keep shopping? 🛍️"
#                     )
#                 else:
#                     response_text = "I couldn't find that product. Can you give me more details?"
#                 await save_message(session_id, "assistant", response_text)
#                 cart = await get_cart(session_id)
#                 await websocket.send_json({
#                     "type": "response",
#                     "response": response_text,
#                     "cart": [serialize_doc(c) for c in cart]
#                 })
#                 continue

#             # ── Order Issues (delay, status, refund, complaint) ──
#             if intent in ["order_status", "order_delay", "refund_return", "complaint"]:
#                 # Ask for product ID if not given
#                 session_prods = await get_session_products(session_id)
#                 if not session_prods and decision.get("needs_product_id", True):
#                     response_text = (
#                         "I'd love to help with that! To pull up your order details, could you please share:\n\n"
#                         "1️⃣ **Product ID** (found in your order confirmation)\n"
#                         "2️⃣ **Product Name**\n\n"
#                         "Please enter your **Product ID** first:"
#                     )
#                     pending_product_request = {"awaiting": "product_id", "intent": intent}
#                     await save_message(session_id, "assistant", response_text)
#                     await websocket.send_json({"type": "response", "response": response_text})
#                     continue

#                 response_text = await generate_response(session_id, user_input, decision)
#                 await save_message(session_id, "assistant", response_text)
#                 await websocket.send_json({"type": "response", "response": response_text})
#                 continue

#             # ── Fallback ──
#             response_text = await generate_response(session_id, user_input, decision)
#             await save_message(session_id, "assistant", response_text)
#             await websocket.send_json({"type": "response", "response": response_text})

#     except WebSocketDisconnect:
#         active_connections.pop(session_id, None)
#     except Exception as e:
#         print(f"[ERROR] {e}")
#         try:
#             await websocket.send_json({"type": "error", "message": "Something went wrong. Please try again."})
#         except Exception:
#             pass


# # ==========================
# # 🔹 ADMIN APIs
# # ==========================
# class ProductIn(BaseModel):
#     product_name: str
#     description: str
#     price: float
#     category: str
#     stock: int = 0

# class CompanyKnowledgeIn(BaseModel):
#     text: str
#     doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# @app.post("/admin/products")
# async def admin_add_product(product: ProductIn):
#     doc = AdminProduct(**product.dict())
#     await doc.insert()
#     return {"status": "created", "id": str(doc.id)}

# @app.get("/admin/products")
# async def admin_list_products():
#     products = await AdminProduct.find_all().to_list()
#     return [{"id": str(p.id), "name": p.product_name, "price": p.price, "category": p.category, "stock": p.stock} for p in products]

# @app.post("/admin/company-knowledge")
# async def admin_add_company_knowledge(payload: CompanyKnowledgeIn):
#     company_collection.add(
#         documents=[payload.text],
#         ids=[payload.doc_id]
#     )
#     return {"status": "added", "doc_id": payload.doc_id}

# @app.get("/admin/sessions")
# async def admin_sessions():
#     summaries = await ChatSummary.find_all().to_list()
#     result = []
#     for s in summaries:
#         prods = await SessionProduct.find(SessionProduct.session_id == s.session_id).to_list()
#         cart  = await CartItem.find(CartItem.session_id == s.session_id).to_list()
#         result.append({
#             "session_id": s.session_id,
#             "summary": s.summary,
#             "message_count": s.message_count,
#             "last_updated": s.last_updated.isoformat(),
#             "products_viewed": [{"id": p.product_id, "name": p.product_name} for p in prods],
#             "cart_items": [{"id": c.product_id, "name": c.product_name, "qty": c.quantity} for c in cart]
#         })
#     return result


# # ==========================
# # 🔹 FRONTEND — FULLY FIXED
# # ==========================
# @app.get("/")
# async def get_frontend():
#     html = r"""<!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <meta name="viewport" content="width=device-width, initial-scale=1.0">
# <title>ShopBot AI • E-commerce Assistant</title>
# <script src="https://cdn.tailwindcss.com"></script>
# <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
# <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
# <style>
#   * { box-sizing: border-box; }
#   body { font-family: 'Inter', sans-serif; background: #0a0f1e; color: #e2e8f0; }
#   ::-webkit-scrollbar { width: 6px; }
#   ::-webkit-scrollbar-track { background: transparent; }
#   ::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }

#   .glass { background: rgba(255,255,255,0.04); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08); }
#   .user-bubble { background: linear-gradient(135deg, #3b82f6, #6366f1); color: white; border-radius: 18px 18px 4px 18px; }
#   .bot-bubble  { background: #1e293b; color: #e2e8f0; border-radius: 18px 18px 18px 4px; border: 1px solid #334155; }
#   .human-bubble { background: linear-gradient(135deg, #dc2626, #f97316); color: white; border-radius: 18px; }

#   .product-card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 12px; transition: all 0.2s; cursor: pointer; }
#   .product-card:hover { border-color: #3b82f6; transform: translateY(-2px); }

#   .session-row { background: #1e293b; border-radius: 8px; font-size: 12px; }
#   .pill { display: inline-block; background: #0f172a; border: 1px solid #334155; border-radius: 999px; padding: 2px 8px; font-size: 11px; }

#   .send-btn { background: linear-gradient(135deg, #3b82f6, #6366f1); }
#   .send-btn:hover { opacity: 0.9; transform: scale(1.05); }

#   .typing-dot { animation: bounce 0.8s infinite; }
#   .typing-dot:nth-child(2) { animation-delay: 0.1s; }
#   .typing-dot:nth-child(3) { animation-delay: 0.2s; }
#   @keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-6px)} }

#   .tab-btn { padding: 8px 20px; border-radius: 8px; font-size: 14px; font-weight: 500; transition: all 0.2s; }
#   .tab-btn.active { background: #3b82f6; color: white; }
#   .tab-btn:not(.active) { color: #94a3b8; }
#   .tab-btn:not(.active):hover { color: white; }

#   .status-pulse { width:8px; height:8px; border-radius:50%; background:#22c55e; animation: pls 2s infinite; }
#   @keyframes pls { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.3)} }
# </style>
# </head>
# <body class="min-h-screen">

# <div class="flex h-[100dvh] overflow-hidden">

#   <!-- SIDEBAR -->
#   <div class="w-72 glass border-r border-white/5 flex flex-col p-4 gap-4 flex-shrink-0">
#     <div class="flex items-center gap-3 px-2 pt-2">
#       <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-violet-600 rounded-xl flex items-center justify-center text-xl">🛍️</div>
#       <div>
#         <h1 class="font-bold text-lg leading-tight">ShopBot AI</h1>
#         <p class="text-xs text-emerald-400 flex items-center gap-1"><span class="status-pulse"></span> Online</p>
#       </div>
#     </div>

#     <div class="flex gap-2">
#       <button class="tab-btn active flex-1" onclick="switchTab('chat')">💬 Chat</button>
#       <button class="tab-btn flex-1" onclick="switchTab('admin')">⚙️ Admin</button>
#     </div>

#     <div>
#       <p class="text-xs text-slate-400 font-semibold uppercase tracking-wider mb-2">Session Products</p>
#       <div id="session-products" class="space-y-1 text-xs">
#         <p class="text-slate-500 italic">No products yet</p>
#       </div>
#     </div>

#     <div class="mt-auto">
#       <button onclick="viewCart()" class="w-full flex items-center justify-between px-4 py-3 bg-slate-800 hover:bg-slate-700 rounded-xl transition text-sm">
#         <span class="flex items-center gap-2"><i class="fa-solid fa-cart-shopping text-blue-400"></i> My Cart</span>
#         <span id="cart-count" class="bg-blue-600 text-white text-xs px-2 py-0.5 rounded-full">0</span>
#       </button>
#       <div id="session-id-display" class="mt-2 text-xs text-slate-500 font-mono truncate px-1"></div>
#     </div>
#   </div>

#   <!-- MAIN AREA -->
#   <div class="flex-1 flex flex-col min-w-0">

#     <!-- CHAT TAB -->
#     <div id="tab-chat" class="flex-1 flex flex-col min-h-0">
#       <div id="messages" class="flex-1 overflow-y-auto p-6 space-y-4 min-h-0"></div>
#       <div id="typing-indicator" class="hidden px-6 pb-2">
#         <div class="flex items-center gap-3">
#           <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-violet-600 rounded-full flex items-center justify-center text-sm">🤖</div>
#           <div class="bot-bubble px-4 py-3 flex gap-1 items-center">
#             <span class="typing-dot w-2 h-2 bg-slate-400 rounded-full"></span>
#             <span class="typing-dot w-2 h-2 bg-slate-400 rounded-full"></span>
#             <span class="typing-dot w-2 h-2 bg-slate-400 rounded-full"></span>
#           </div>
#         </div>
#       </div>
#       <div id="status-bar" class="px-6 py-1 text-xs text-emerald-400 font-medium min-h-[24px]"></div>
#       <div class="p-4 border-t border-white/5 glass shrink-0">
#         <div class="flex gap-3 max-w-4xl mx-auto">
#           <input id="user-input" type="text" placeholder="Ask about products, orders, returns..."
#             class="flex-1 bg-slate-800 text-white placeholder:text-slate-500 rounded-2xl px-5 py-3.5 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
#             onkeydown="if(event.key==='Enter') sendMessage()">
#           <button onclick="sendMessage()" class="send-btn w-12 h-12 rounded-2xl flex items-center justify-center transition-all flex-shrink-0">
#             <i class="fa-solid fa-paper-plane text-white"></i>
#           </button>
#         </div>
#       </div>
#     </div>

#     <!-- ADMIN TAB -->
#     <div id="tab-admin" class="flex-1 overflow-y-auto p-6 hidden">
#       <h2 class="text-xl font-bold mb-6">⚙️ Admin Panel</h2>
#       <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

#         <!-- Add Product -->
#         <div class="glass rounded-2xl p-5">
#           <h3 class="font-semibold mb-4 flex items-center gap-2"><i class="fa-solid fa-box text-blue-400"></i> Add Product to DB</h3>
#           <div class="space-y-3">
#             <input id="adm-pname" placeholder="Product Name" class="w-full bg-slate-800 text-white rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
#             <input id="adm-pdesc" placeholder="Description" class="w-full bg-slate-800 text-white rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
#             <div class="flex gap-2">
#               <input id="adm-pprice" type="number" placeholder="Price" class="flex-1 bg-slate-800 text-white rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
#               <input id="adm-pstock" type="number" placeholder="Stock" class="flex-1 bg-slate-800 text-white rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
#             </div>
#             <input id="adm-pcat" placeholder="Category (e.g. mens-shoes)" class="w-full bg-slate-800 text-white rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
#             <button onclick="adminAddProduct()" class="w-full bg-blue-600 hover:bg-blue-700 text-white rounded-xl py-2.5 text-sm font-medium transition">Add Product</button>
#             <div id="adm-prod-msg" class="text-xs mt-2"></div>
#           </div>
#         </div>

#         <!-- Add Company Knowledge -->
#         <div class="glass rounded-2xl p-5">
#           <h3 class="font-semibold mb-4 flex items-center gap-2"><i class="fa-solid fa-brain text-violet-400"></i> Train Company Knowledge (ChromaDB)</h3>
#           <div class="space-y-3">
#             <textarea id="adm-know" rows="5" placeholder="e.g. We offer free shipping on orders above $50. Return policy: 30 days no questions asked..."
#               class="w-full bg-slate-800 text-white rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500 resize-none"></textarea>
#             <button onclick="adminAddKnowledge()" class="w-full bg-violet-600 hover:bg-violet-700 text-white rounded-xl py-2.5 text-sm font-medium transition">Add to Vector DB</button>
#             <div id="adm-know-msg" class="text-xs mt-2"></div>
#           </div>
#         </div>

#         <!-- Sessions List -->
#         <div class="glass rounded-2xl p-5 lg:col-span-2">
#           <div class="flex items-center justify-between mb-4">
#             <h3 class="font-semibold flex items-center gap-2"><i class="fa-solid fa-users text-emerald-400"></i> Active Sessions</h3>
#             <button onclick="loadSessions()" class="text-xs bg-slate-700 hover:bg-slate-600 px-3 py-1.5 rounded-lg transition">Refresh</button>
#           </div>
#           <div id="adm-sessions" class="space-y-3 text-sm text-slate-400">Click refresh to load sessions.</div>
#         </div>
#       </div>
#     </div>
#   </div>
# </div>

# <script>
# // Full JavaScript with correct endpoints
# let ws = null;
# let sessionId = "";
# let cartCount = 0;

# function switchTab(tab) {
#   document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
#   const activeBtn = [...document.querySelectorAll('.tab-btn')].find(btn => btn.getAttribute('onclick').includes(tab));
#   if (activeBtn) activeBtn.classList.add('active');

#   document.getElementById('tab-chat').classList.toggle('hidden', tab !== 'chat');
#   document.getElementById('tab-admin').classList.toggle('hidden', tab !== 'admin');
#   if (tab === 'admin') loadSessions();
# }

# function scrollToBottom() {
#   const container = document.getElementById("messages");
#   requestAnimationFrame(() => container.scrollTop = container.scrollHeight);
# }

# function forceScrollAfterContent() {
#   setTimeout(scrollToBottom, 100);
#   setTimeout(scrollToBottom, 300);
# }

# function formatText(text) {
#   let html = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
#                  .replace(/\*(.*?)\*/g, '<em>$1</em>')
#                  .replace(/\n/g, '<br>');
#   return html;
# }

# function appendMessage(role, text, extra = {}) {
#   const wrap = document.getElementById("messages");
#   const messageDiv = document.createElement("div");
#   messageDiv.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'} items-start gap-3 mb-6`;

#   const isUser = role === 'user';
#   const bubbleClass = isUser ? 'user-bubble' : role === 'human' ? 'human-bubble' : 'bot-bubble';
#   const avatar = isUser 
#     ? `<div class="w-8 h-8 bg-slate-700 rounded-2xl flex items-center justify-center text-lg flex-shrink-0 mt-1">👤</div>`
#     : `<div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-violet-600 rounded-2xl flex items-center justify-center text-lg flex-shrink-0 mt-1">${role === 'human' ? '👤' : '🤖'}</div>`;

#   let innerHTML = `<div class="max-w-[75%]"><div class="${bubbleClass} px-5 py-3.5 text-[15px] leading-relaxed">${formatText(text)}</div>`;

#   if (extra?.products?.length > 0) {
#     innerHTML += `<div class="grid grid-cols-2 md:grid-cols-3 gap-3 mt-4">` + 
#       extra.products.slice(0,6).map(p => `
#         <div onclick="addToCartQuick(${p.id}, '${(p.title||'').replace(/'/g,"\\'")}')" class="product-card">
#           <img src="${p.thumbnail||''}" class="w-full h-32 object-cover rounded-xl mb-2" onerror="this.src='https://via.placeholder.com/300x200?text=No+Image'">
#           <p class="font-medium text-sm">${p.title||p.name||'Product'}</p>
#           <p class="text-blue-400 font-bold">$${p.price}</p>
#           <button class="mt-3 w-full bg-blue-600 hover:bg-blue-700 py-2 rounded-xl text-xs">+ Add to Cart</button>
#         </div>`).join('') + `</div>`;
#   }

#   innerHTML += `</div>`;
#   messageDiv.innerHTML = `${!isUser ? avatar : ''}${innerHTML}${isUser ? avatar : ''}`;
#   wrap.appendChild(messageDiv);
#   forceScrollAfterContent();
# }

# function showTyping(show) {
#   document.getElementById("typing-indicator").classList.toggle('hidden', !show);
# }

# function setStatus(msg) {
#   const el = document.getElementById("status-bar");
#   el.textContent = msg;
#   if (msg) setTimeout(() => el.textContent = '', 3500);
# }

# function sendMessage() {
#   const input = document.getElementById("user-input");
#   const text = input.value.trim();
#   if (!text || !ws) return;
#   appendMessage("user", text);
#   input.value = "";
#   showTyping(true);
#   ws.send(JSON.stringify({ message: text }));
# }

# function viewCart() {
#   if (!ws) return;
#   appendMessage("user", "Show me my cart");
#   showTyping(true);
#   ws.send(JSON.stringify({ message: "show me my cart" }));
# }

# function addToCartQuick(id, name) {
#   if (!ws) return;
#   const msg = `Add product ${name} (ID: ${id}) to my cart`;
#   appendMessage("user", msg);
#   showTyping(true);
#   ws.send(JSON.stringify({ message: msg }));
# }

# // ====================== ADMIN FUNCTIONS ======================
# async function adminAddProduct() {
#   const name = document.getElementById('adm-pname').value.trim();
#   const desc = document.getElementById('adm-pdesc').value.trim();
#   const price = parseFloat(document.getElementById('adm-pprice').value);
#   const stock = parseInt(document.getElementById('adm-pstock').value) || 0;
#   const category = document.getElementById('adm-pcat').value.trim();

#   if (!name || !price || !category) {
#     alert("Please fill Product Name, Price and Category");
#     return;
#   }

#   const msgDiv = document.getElementById('adm-prod-msg');
#   msgDiv.textContent = "Adding...";
#   msgDiv.style.color = '#eab308';

#   try {
#     const res = await fetch('/admin/add-product', {
#       method: 'POST',
#       headers: {'Content-Type': 'application/json'},
#       body: JSON.stringify({name, description: desc, price, stock, category})
#     });

#     const data = await res.json();
#     if (res.ok && data.status === "success") {
#       msgDiv.textContent = "✅ Product added successfully!";
#       msgDiv.style.color = '#10b981';
#       // Clear form
#       document.getElementById('adm-pname').value = '';
#       document.getElementById('adm-pdesc').value = '';
#       document.getElementById('adm-pprice').value = '';
#       document.getElementById('adm-pstock').value = '';
#       document.getElementById('adm-pcat').value = '';
#     } else {
#       msgDiv.textContent = "❌ " + (data.message || "Failed");
#       msgDiv.style.color = '#ef4444';
#     }
#   } catch (e) {
#     msgDiv.textContent = "❌ Connection error";
#     msgDiv.style.color = '#ef4444';
#   }
# }

# async function adminAddKnowledge() {
#   const text = document.getElementById('adm-know').value.trim();
#   if (!text) return alert("Enter knowledge text");

#   const msgDiv = document.getElementById('adm-know-msg');
#   msgDiv.textContent = "Adding...";
#   msgDiv.style.color = '#eab308';

#   try {
#     const res = await fetch('/admin/add-knowledge', {
#       method: 'POST',
#       headers: {'Content-Type': 'application/json'},
#       body: JSON.stringify({text})
#     });

#     const data = await res.json();
#     if (res.ok && data.status === "success") {
#       msgDiv.textContent = "✅ Knowledge added to ChromaDB!";
#       msgDiv.style.color = '#10b981';
#       document.getElementById('adm-know').value = '';
#     } else {
#       msgDiv.textContent = "❌ " + (data.message || "Failed");
#       msgDiv.style.color = '#ef4444';
#     }
#   } catch (e) {
#     msgDiv.textContent = "❌ Connection error";
#     msgDiv.style.color = '#ef4444';
#   }
# }

# async function loadSessions() {
#   const container = document.getElementById('adm-sessions');
#   container.innerHTML = "Loading...";

#   try {
#     const res = await fetch('/admin/sessions');
#     const data = await res.json();

#     if (data.sessions && data.sessions.length > 0) {
#       container.innerHTML = data.sessions.map(s => `
#         <div class="glass p-4 rounded-xl">
#           <div class="font-mono text-xs text-emerald-400">${s.session_id.slice(0,8)}...</div>
#           <div class="text-slate-300 text-sm mt-1">${s.last_message}</div>
#           <div class="text-xs text-slate-500 mt-2">${new Date(s.updated_at).toLocaleString()}</div>
#         </div>
#       `).join('');
#     } else {
#       container.innerHTML = `<p class="text-slate-500 italic">No active sessions found.</p>`;
#     }
#   } catch (err) {
#     container.innerHTML = `<p class="text-red-400">Failed to load sessions.</p>`;
#   }
# }

# // WebSocket Connection
# function connectWS() {
#   const proto = location.protocol === 'https:' ? 'wss' : 'ws';
#   ws = new WebSocket(`${proto}://${location.host}/ws/chat`);

#   ws.onopen = () => console.log("✅ WebSocket Connected");

#   ws.onmessage = (event) => {
#     const data = JSON.parse(event.data);
#     showTyping(false);

#     if (data.session_id) {
#       sessionId = data.session_id;
#       document.getElementById("session-id-display").textContent = `Session: ${sessionId.slice(0,8)}…`;
#     }

#     if (data.type === "status") {
#       setStatus(data.message);
#       return;
#     }

#     if (data.type === "connected" || data.type === "response") {
#       appendMessage("bot", data.response || data.message, {
#         products: data.products || [],
#         cart: data.cart || []
#       });
#       if (data.cart) {
#         document.getElementById("cart-count").textContent = data.cart.length;
#       }
#     }
#   };

#   ws.onclose = () => setTimeout(connectWS, 2000);
# }

# window.onload = connectWS;
# </script>
# </body>
# </html>"""
#     return HTMLResponse(content=html)





import uuid
import json
import os
import asyncio
import httpx
from datetime import datetime
from typing import TypedDict, Optional, List, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# MongoDB / Beanie
from beanie import Document, init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel, ASCENDING

# ChromaDB
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction

# Groq
from groq import AsyncGroq
from sentence_transformers import SentenceTransformer
import numpy as np

# Load Sentence Transformer model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================
# 🔹 CONFIG
# ==========================
MONGO_URI      = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME        = "ai_support"
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
DUMMYJSON_BASE = "https://dummyjson.com"

CATEGORIES = [
    "beauty","fragrances","furniture","groceries","home-decoration",
    "kitchen-accessories","laptops","mens-shirts","mens-shoes","mens-watches",
    "mobile-accessories","motorcycle","skin-care","smartphones","sports-accessories",
    "sunglasses","tablets","tops","vehicle","womens-bags","womens-dresses",
    "womens-jewellery","womens-shoes","womens-watches"
]

app  = FastAPI(title="Enterprise E-commerce AI Agent")
groq = AsyncGroq(api_key=GROQ_API_KEY)

# ==========================
# 🔹 CHROMADB + EMBEDDINGS
# ==========================
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model):
        self.model = model

    def __call__(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

chroma_client  = chromadb.Client()
st_embed_fn    = SentenceTransformerEmbeddingFunction(st_model)
company_collection = chroma_client.get_or_create_collection(
    name="company_knowledge",
    embedding_function=st_embed_fn
)

try:
    test_vec = st_embed_fn(["test embedding"])
    print(f"[DEBUG] Embedding dimension: {len(test_vec[0])}")
except Exception as e:
    print("[ERROR] Embedding test failed:", e)

# Active WebSocket connections: session_id -> websocket
active_connections: Dict[str, WebSocket] = {}

# Admin WebSocket connections: admin_id -> websocket
admin_connections: Dict[str, WebSocket] = {}

# Human takeover map: session_id -> True (AI disabled)
human_takeover_sessions: Dict[str, bool] = {}

# Map: session_id -> admin_session_id (which admin is handling it)
session_admin_map: Dict[str, str] = {}

# ==========================
# 🔹 DB MODELS
# ==========================
class ChatMessage(Document):
    session_id: str
    role: str           # "user" | "assistant" | "admin"
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "chat_messages"
        indexes = [
            IndexModel([("session_id", ASCENDING)]),
            IndexModel([("timestamp", ASCENDING)], expireAfterSeconds=7776000),  # 90 days TTL
        ]

class ChatSummary(Document):
    session_id: str
    summary: str = ""
    message_count: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "chat_summaries"
        indexes = [IndexModel([("session_id", ASCENDING)], unique=True)]

class SessionProduct(Document):
    session_id: str
    product_id: str
    product_name: str
    category: str = ""
    added_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "session_products"
        indexes = [IndexModel([("session_id", ASCENDING)])]

class CartItem(Document):
    session_id: str
    product_id: str
    product_name: str
    price: float
    quantity: int = 1
    thumbnail: str = ""
    added_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "cart_items"
        indexes = [IndexModel([("session_id", ASCENDING)])]

class AdminProduct(Document):
    product_name: str
    description: str
    price: float
    category: str
    stock: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "admin_products"

# NEW: Customer Order Model (admin-managed, searched by email)
class CustomerOrder(Document):
    email: str
    product_name: str
    product_id: str
    status: str          # "delivered" | "processing" | "shipped" | "cancelled"
    order_date: datetime = Field(default_factory=datetime.utcnow)
    delivery_date: Optional[datetime] = None
    price: float = 0.0
    notes: str = ""

    class Settings:ss 50+ websites, achieving 98% accuracy using Selenium with headless Chrome.
        name = "customer_orders"
        indexes = [
            IndexModel([("email", ASCENDING)]),
            IndexModel([("order_date", ASCENDING)]),
        ]

# ==========================
# 🔹 STARTUP
# ==========================
@app.on_event("startup")
async def init_db():
    client = AsyncIOMotorClient(MONGO_URI)
    await init_beanie(
        database=client[DB_NAME],
        document_models=[ChatMessage, ChatSummary, SessionProduct, CartItem, AdminProduct, CustomerOrder]
    )

# ==========================
# 🔹 LLM CALLER
# ==========================
async def call_llm(messages: List[Dict], temperature: float = 0.3) -> str:
    resp = await groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content

# ==========================
# 🔹 MEMORY — Save, Summarize, Context
# ==========================
async def save_message(session_id: str, role: str, message: str):
    await ChatMessage(session_id=session_id, role=role, message=message).insert()
    count = await ChatMessage.find(ChatMessage.session_id == session_id).count()
    # Always update summary every 6 messages (or every message if small window)
    if count > 0 and count % 6 == 0:
        await _generate_and_save_summary(session_id)
    else:
        # Incremental: always keep summary fresh with latest info
        await _incremental_summary_update(session_id)

async def _incremental_summary_update(session_id: str):
    """Light-weight summary refresh: append latest message to existing summary."""
    existing = await ChatSummary.find_one({"session_id": session_id})
    count = await ChatMessage.find(ChatMessage.session_id == session_id).count()
    if not existing:
        existing = ChatSummary(session_id=session_id, summary="", message_count=0)
    existing.message_count = count
    existing.last_updated = datetime.utcnow()
    await existing.save()

async def _generate_and_save_summary(session_id: str):
    """Full LLM-based rolling summary of all messages."""
    msgs = await ChatMessage.find(
        ChatMessage.session_id == session_id
    ).sort("timestamp").to_list()
    if not msgs:
        return

    chat_text = "\n".join(
        f"[{m.timestamp.strftime('%H:%M:%S')}] {m.role}: {m.message}" for m in msgs
    )

    # Get existing summary to do rolling summarization
    existing = await ChatSummary.find_one({"session_id": session_id})
    prev_summary = existing.summary if existing else ""

    system_prompt = (
        "You are an expert e-commerce support conversation summarizer.\n"
        "Your job is to create a ROLLING SUMMARY that captures:\n"
        "1. Customer's name/email if mentioned\n"
        "2. Products browsed, asked about, or added to cart (with IDs if given)\n"
        "3. Any order issues, complaints, or specific problems raised\n"
        "4. Customer's emotional state and sentiment trend\n"
        "5. Any resolutions offered or pending actions\n"
        "6. Whether a human agent took over\n"
        "7. Key context needed for the next AI response\n\n"
        "Write the summary in 5-7 sentences. Be specific. Include product names/IDs/emails if present.\n"
        "This summary will be injected into every AI response, so make it DECISION-RELEVANT."
    )

    user_prompt = f"PREVIOUS SUMMARY:\n{prev_summary}\n\nFULL CHAT:\n{chat_text}\n\nWrite updated rolling summary:"

    summary_text = await call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ], temperature=0.2)

    if existing:
        existing.summary      = summary_text
        existing.message_count = len(msgs)
        existing.last_updated  = datetime.utcnow()
        await existing.save()
    else:
        await ChatSummary(
            session_id=session_id,
            summary=summary_text,
            message_count=len(msgs)
        ).insert()

async def get_chat_context(session_id: str) -> str:
    """Build full context: rolling summary + recent 15 messages."""
    summary_doc = await ChatSummary.find_one({"session_id": session_id})
    summary     = summary_doc.summary if (summary_doc and summary_doc.summary) else "No prior context."

    recent = await ChatMessage.find(
        ChatMessage.session_id == session_id
    ).sort("-timestamp").limit(15).to_list()

    lines = [
        f"[{m.timestamp.strftime('%H:%M')}] {m.role.upper()}: {m.message}"
        for m in reversed(recent)
    ]
    return (
        f"=== ROLLING CONVERSATION SUMMARY ===\n{summary}\n\n"
        f"=== RECENT MESSAGES (last {len(lines)}) ===\n" + "\n".join(lines)
    )

# ==========================
# 🔹 DUMMYJSON API TOOLS
# ==========================
async def detect_category(user_input: str) -> Optional[str]:
    cats = ", ".join(CATEGORIES)
    prompt = (
        f"User said: \"{user_input}\"\n"
        f"Available categories: {cats}\n"
        "Return ONLY the single best matching category slug, or 'none' if no match."
    )
    result = await call_llm([
        {"role": "system", "content": "You are a category classifier. Return only the slug string, nothing else."},
        {"role": "user", "content": prompt}
    ], temperature=0.0)
    result = result.strip().lower().strip('"').strip("'")
    return result if result in CATEGORIES else None

async def fetch_products_by_category(category: str, limit: int = 5) -> List[Dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{DUMMYJSON_BASE}/products/category/{category}?limit={limit}")
        data = resp.json()
        return data.get("products", [])

async def search_products(query: str, limit: int = 5) -> List[Dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{DUMMYJSON_BASE}/products/search?q={query}&limit={limit}")
        data = resp.json()
        return data.get("products", [])

async def get_product_by_id(product_id: str) -> Optional[Dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{DUMMYJSON_BASE}/products/{product_id}")
        if resp.status_code == 200:
            return resp.json()
    return None

# ==========================
# 🔹 MCP TOOL: Search Customer Orders by Email
# ==========================
async def mcp_get_orders_by_email(email: str) -> List[Dict]:
    """MCP Tool: Fetch all orders for a customer by email, descending by order_date. 
    Completed/delivered orders clearly marked."""
    orders = await CustomerOrder.find(
        CustomerOrder.email == email.lower().strip()
    ).sort("-order_date").to_list()

    result = []
    for o in orders:
        result.append({
            "order_id":     str(o.id),
            "product_name": o.product_name,
            "product_id":   o.product_id,
            "status":       o.status,
            "is_delivered": o.status.lower() == "delivered",
            "order_date":   o.order_date.strftime("%Y-%m-%d %H:%M"),
            "delivery_date": o.delivery_date.strftime("%Y-%m-%d") if o.delivery_date else "Pending",
            "price":        o.price,
            "notes":        o.notes,
        })
    return result

def format_orders_for_display(orders: List[Dict], email: str) -> str:
    if not orders:
        return f"No orders found for **{email}**."

    lines = [f"📦 Orders for **{email}** (newest first):\n"]
    for i, o in enumerate(orders, 1):
        status_emoji = {
            "delivered": "✅",
            "shipped": "🚚",
            "processing": "⏳",
            "cancelled": "❌"
        }.get(o["status"].lower(), "📦")

        delivered_note = " *(Successfully Delivered)*" if o["is_delivered"] else ""
        lines.append(
            f"**{i}. {o['product_name']}** {status_emoji}{delivered_note}\n"
            f"   Order Date: {o['order_date']} | Status: {o['status'].upper()}\n"
            f"   Delivery: {o['delivery_date']} | Price: ${o['price']}\n"
            + (f"   Note: {o['notes']}\n" if o['notes'] else "")
        )
    return "\n".join(lines)

# ==========================
# 🔹 COMPANY KNOWLEDGE (ChromaDB)
# ==========================
def search_company_knowledge(query: str, n: int = 3) -> str:
    try:
        results = company_collection.query(query_texts=[query], n_results=n)
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs) if docs else ""
    except Exception:
        return ""

# ==========================
# 🔹 CART OPERATIONS
# ==========================
async def add_to_cart(session_id: str, product: Dict) -> str:
    existing = await CartItem.find_one({
        "session_id": session_id,
        "product_id": str(product["id"])
    })
    if existing:
        existing.quantity += 1
        await existing.save()
        return f"Updated cart: {product['title']} (qty: {existing.quantity})"
    else:
        await CartItem(
            session_id=session_id,
            product_id=str(product["id"]),
            product_name=product["title"],
            price=product["price"],
            thumbnail=product.get("thumbnail", ""),
        ).insert()
        return f"Added to cart: {product['title']} @ ${product['price']}"

async def get_cart(session_id: str) -> List[CartItem]:
    return await CartItem.find(CartItem.session_id == session_id).to_list()

# ==========================
# 🔹 SESSION PRODUCT TABLE
# ==========================
async def save_session_product(session_id: str, product_id: str, product_name: str, category: str = ""):
    exists = await SessionProduct.find_one({
        "session_id": session_id, "product_id": product_id
    })
    if not exists:
        await SessionProduct(
            session_id=session_id,
            product_id=product_id,
            product_name=product_name,
            category=category
        ).insert()

async def get_session_products(session_id: str) -> List[SessionProduct]:
    return await SessionProduct.find(SessionProduct.session_id == session_id).to_list()

# ==========================
# 🔹 BROADCAST TO ADMIN — send session chat to all admin connections
# ==========================
async def broadcast_to_admins(payload: Dict):
    dead = []
    for admin_id, ws in admin_connections.items():
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(admin_id)
    for d in dead:
        admin_connections.pop(d, None)

# ==========================
# 🔹 DECISION ENGINE — Iron-clad intent detection with full context
# ==========================
async def decision_engine(session_id: str, user_input: str) -> Dict:
    context      = await get_chat_context(session_id)
    session_prods = await get_session_products(session_id)
    cart         = await get_cart(session_id)

    prod_table = ""
    if session_prods:
        prod_table = "Currently tracked products in this session:\n" + "\n".join(
            f"  - {p.product_name} (ID: {p.product_id}, Category: {p.category})" for p in session_prods
        )

    cart_info = "Cart: empty"
    if cart:
        total = sum(c.price * c.quantity for c in cart)
        cart_info = f"Cart has {len(cart)} item(s), total ${total:.2f}:\n" + "\n".join(
            f"  - {c.product_name} × {c.quantity} @ ${c.price}" for c in cart
        )

    system_prompt = """You are a world-class e-commerce intent classifier with deep NLU capabilities.

Your task: analyze the user's message carefully using ALL context provided, then return ONLY valid JSON.

Intent Options (choose the single BEST match):
- "product_search"    → wants to find/browse/discover products
- "product_detail"    → wants details about a specific product by name or ID
- "add_to_cart"       → explicitly wants to add something to cart
- "view_cart"         → wants to see or review their cart
- "order_status"      → asking about the status of their order
- "order_delay"       → asking why their order is delayed or late
- "refund_return"     → wants a refund, exchange, or return
- "complaint"         → expressing a grievance or dissatisfaction
- "company_info"      → asking about policies, shipping, returns policy, company details
- "connect_human"     → explicitly wants to talk to a human agent / live support
- "escalate"          → extremely angry, demanding help, or complex unresolved multi-issue
- "off_topic"         → completely unrelated to shopping, products, or orders
- "greeting"          → hello, hi, good morning, thanks, bye, small talk
- "provide_email"     → the user just typed what looks like an email address
- "provide_product_id" → the user just provided a product ID number

Return this JSON and NOTHING else:
{
  "intent": "<one of the above>",
  "sentiment": "positive" | "neutral" | "slightly_frustrated" | "frustrated" | "angry" | "very_angry",
  "needs_email": true | false,
  "needs_product_id": true | false,
  "category_hint": "<category slug or empty string>",
  "add_to_cart_trigger": true | false,
  "confidence": 0.0-1.0,
  "escalate_now": true | false,
  "email_detected": "<email if detected in message, else empty string>",
  "product_id_detected": "<product ID if detected in message, else empty string>",
  "wants_human": true | false
}

CRITICAL RULES:
- NEVER forget prior context — if user already gave their email earlier, do not ask again
- "connect_human" → wants_human = true (user said things like "talk to agent", "connect me to support", "human please", "real person")
- "escalate_now" = true ONLY if sentiment is very_angry OR user has repeated the same issue 3+ times
- "needs_email" = true if user is asking about their order/return/refund AND we don't have their email yet
- "needs_product_id" = true only when user references a specific product without providing ID
- "provide_email" intent when the user's message is just or mostly an email address
- "off_topic" = user asking about weather, politics, coding help, jokes — anything non-ecommerce
- product IDs are typically numeric (e.g. "1", "42", "100") from dummyjson
- Detect email pattern: anything with @ and a domain
- If user says "yes", "sure", "ok", "add it", "go ahead" — likely continuing prior intent from context
"""

    user_prompt = (
        f"{context}\n\n"
        f"{prod_table}\n{cart_info}\n\n"
        f"USER'S LATEST MESSAGE: \"{user_input}\"\n\n"
        "Classify intent and return JSON:"
    )

    raw = await call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ], temperature=0.0)

    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        return json.loads(clean)
    except Exception:
        return {
            "intent":             "complaint",
            "sentiment":          "neutral",
            "needs_email":        False,
            "needs_product_id":   False,
            "category_hint":      "",
            "add_to_cart_trigger": False,
            "confidence":         0.5,
            "escalate_now":       False,
            "email_detected":     "",
            "product_id_detected":"",
            "wants_human":        False,
        }

# ==========================
# 🔹 RESPONSE GENERATOR — Full human-like AI responses
# ==========================
async def generate_response(
    session_id: str,
    user_input: str,
    decision: Dict,
    products: List[Dict] = None,
    orders: List[Dict] = None,
    extra_context: str = ""
) -> str:
    context      = await get_chat_context(session_id)
    prod_str     = json.dumps(products, indent=2) if products else "No products retrieved."
    company_info = search_company_knowledge(user_input)
    order_str    = format_orders_for_display(orders, "") if orders else "No order data."

    system_prompt = """You are an expert, empathetic e-commerce support agent named "ShopBot".

Your personality:
- Warm, professional, and genuinely helpful — like a knowledgeable friend who works in retail
- Never robotic or generic. Sound like a real human who READS the conversation
- Adapt your tone: casual for greetings, empathetic for complaints, enthusiastic for shopping
- NEVER repeat yourself across messages — always add new value

Your capabilities:
- Help customers find products and make purchase decisions
- Handle order issues, delays, refunds, and returns
- Answer questions about company policies using provided knowledge base
- Escalate to human agents when needed

Strict rules:
- ONLY discuss e-commerce topics (products, orders, shipping, returns, company info)
- NEVER make up product data — only use what is provided
- If order data is provided, ALWAYS present it clearly (newest first, highlight "delivered" orders)
- For product queries: mention name, price, ID, key specs, and a buying recommendation
- Always end with a natural, relevant follow-up question OR a clear next step
- If the customer seems frustrated, acknowledge their feelings FIRST before solving
- If you don't know something, admit it and offer to connect them to a human agent

DO NOT:
- Use bullet points for everything — vary your response style
- Start every message with "Great!" or "Sure!" — be natural
- Copy-paste product JSON directly — summarize it naturally
"""

    user_prompt = (
        f"FULL CONVERSATION CONTEXT:\n{context}\n\n"
        f"COMPANY KNOWLEDGE BASE:\n{company_info if company_info else 'Standard policies apply.'}\n\n"
        f"PRODUCT DATA:\n{prod_str}\n\n"
        f"ORDER DATA:\n{order_str}\n\n"
        f"EXTRA CONTEXT:\n{extra_context}\n\n"
        f"DETECTED INTENT: {decision.get('intent')}\n"
        f"CUSTOMER SENTIMENT: {decision.get('sentiment')}\n\n"
        f"CUSTOMER'S MESSAGE: \"{user_input}\"\n\n"
        "Write your response (be natural, warm, and helpful):"
    )

    return await call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
    ], temperature=0.45)



def serialize_doc(doc):
    data = doc.dict()
    data["id"] = str(doc.id)
    return data

# ==========================
# 🔹 WEBSOCKET — CUSTOMER CHAT
# ==========================
@app.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    active_connections[session_id] = websocket

    await websocket.send_json({
        "type":       "connected",
        "session_id": session_id,
        "message":    "👋 Hey there! Welcome to ShopBot — your personal shopping assistant. Whether you're looking for products, tracking an order, or need help with a return, I've got you covered. What can I help you with today?"
    })

    # Notify admins of new session
    await broadcast_to_admins({
        "type":       "new_session",
        "session_id": session_id,
        "message":    f"New customer session started: {session_id[:8]}…"
    })

    # Conversation state
    pending: Dict = {}   # tracks what we're waiting for: {"awaiting": "product_id"|"email"|"human_confirm"}
    customer_email: Optional[str] = None

    try:
        while True:
            data       = await websocket.receive_json()
            user_input = data.get("message", "").strip()
            if not user_input:
                continue

            # Save user message
            await save_message(session_id, "user", user_input)

            # Forward message to admin in real-time
            await broadcast_to_admins({
                "type":       "customer_message",
                "session_id": session_id,
                "role":       "user",
                "message":    user_input,
                "timestamp":  datetime.utcnow().isoformat()
            })

            # ==== IF AI IS DISABLED (human takeover) ====
            if human_takeover_sessions.get(session_id):
                # Message goes only to admin; AI stays silent
                await websocket.send_json({
                    "type":    "human_active",
                    "message": "💬 You're connected with a support specialist. They'll respond shortly."
                })
                continue

            # ==== PENDING STATE HANDLERS ====

            # Waiting for email
            if pending.get("awaiting") == "email":
                email_val = user_input.strip().lower()
                if "@" in email_val and "." in email_val:
                    customer_email = email_val
                    pending = {}
                    orders  = await mcp_get_orders_by_email(customer_email)
                    if orders:
                        display = format_orders_for_display(orders, customer_email)
                        response_text = (
                            f"Perfect, I found your orders! Here's what I have on file for **{customer_email}**:\n\n"
                            f"{display}\n\n"
                            "Is there a specific order you'd like help with? I can assist with delays, returns, or any other issues."
                        )
                    else:
                        response_text = (
                            f"Hmm, I couldn't find any orders associated with **{customer_email}**. "
                            "Could you double-check the email you used when placing the order? "
                            "Or would you like me to connect you with a human agent for further help?"
                        )
                    await save_message(session_id, "assistant", response_text)
                    await broadcast_to_admins({"type": "bot_message", "session_id": session_id, "message": response_text, "timestamp": datetime.utcnow().isoformat()})
                    await websocket.send_json({"type": "response", "response": response_text})
                    continue
                else:
                    await websocket.send_json({
                        "type":     "response",
                        "response": "That doesn't look like a valid email address. Could you please enter a valid email? (e.g. john@example.com)"
                    })
                    continue

            # Waiting for product ID
            if pending.get("awaiting") == "product_id":
                product_id = user_input.strip()
                product    = await get_product_by_id(product_id)
                if product:
                    await save_session_product(session_id, str(product["id"]), product["title"], product.get("category",""))
                    response_text = (
                        f"Got it! Here are the details for **{product['title']}** (ID: {product['id']}):\n\n"
                        f"💰 **Price:** ${product['price']}\n"
                        f"📦 **In Stock:** {product.get('stock', 'N/A')} units\n"
                        f"⭐ **Rating:** {product.get('rating', 'N/A')}/5\n"
                        f"📝 **Description:** {product.get('description','')}\n\n"
                        "Would you like to add this to your cart, or do you need help with an order related to this product?"
                    )
                    pending = {}
                else:
                    response_text = f"I couldn't find a product with ID **{product_id}**. Could you double-check the ID from your order confirmation?"

                await save_message(session_id, "assistant", response_text)
                await broadcast_to_admins({"type": "bot_message", "session_id": session_id, "message": response_text, "timestamp": datetime.utcnow().isoformat()})
                prods = await get_session_products(session_id)
                await websocket.send_json({
                    "type":             "response",
                    "response":         response_text,
                    "session_products": [{"id": p.product_id, "name": p.product_name, "category": p.category} for p in prods]
                })
                continue

            # ==== DECISION ENGINE ====
            await websocket.send_json({"type": "status", "message": "🤔 Thinking..."})
            decision = await decision_engine(session_id, user_input)
            intent   = decision.get("intent", "complaint")
            sentiment = decision.get("sentiment", "neutral")

            # Auto-capture email from message if detected
            if decision.get("email_detected"):
                customer_email = decision["email_detected"].lower().strip()

            # ==== 1. WANTS HUMAN AGENT ====
            if decision.get("wants_human") or intent == "connect_human":
                response_text = (
                    "Of course! I completely understand — sometimes you just need to speak with a real person. 💙\n\n"
                    "I'm connecting you to one of our support specialists right now. Please hold on for a moment..."
                )
                human_takeover_sessions[session_id] = True
                await save_message(session_id, "assistant", response_text)
                await websocket.send_json({
                    "type":       "human_takeover",
                    "response":   response_text,
                    "session_id": session_id
                })
                await broadcast_to_admins({
                    "type":       "takeover_request",
                    "session_id": session_id,
                    "message":    f"⚡ Customer {session_id[:8]} requested human agent!",
                    "urgent":     True
                })
                continue

            # ==== 2. AUTO ESCALATE (very angry / repeated issue) ====
            if decision.get("escalate_now") or sentiment == "very_angry":
                response_text = (
                    "I can hear how frustrated you are, and I sincerely apologize for the experience you've had. "
                    "This isn't the kind of service we want to provide. 🙏\n\n"
                    "I'm escalating this to a **Senior Support Specialist** right now — they have the authority to resolve this for you immediately."
                )
                human_takeover_sessions[session_id] = True
                await save_message(session_id, "assistant", response_text)
                await websocket.send_json({
                    "type":       "human_takeover",
                    "response":   response_text,
                    "session_id": session_id
                })
                await broadcast_to_admins({
                    "type":       "takeover_request",
                    "session_id": session_id,
                    "message":    f"🔴 URGENT: Angry customer {session_id[:8]} needs immediate escalation!",
                    "urgent":     True
                })
                continue

            # ==== 3. OFF-TOPIC ====
            if intent == "off_topic":
                response_text = (
                    "I appreciate you chatting with me, but I'm specifically trained to help with shopping-related topics! 😊\n\n"
                    "I can help you with:\n"
                    "🛍️ **Finding products** — browse by category or search\n"
                    "📦 **Order tracking** — check your order status\n"
                    "🔄 **Returns & refunds** — hassle-free return assistance\n"
                    "🏢 **Policies** — shipping, warranty, and more\n\n"
                    "What can I help you shop for today?"
                )
                await save_message(session_id, "assistant", response_text)
                await broadcast_to_admins({"type": "bot_message", "session_id": session_id, "message": response_text, "timestamp": datetime.utcnow().isoformat()})
                await websocket.send_json({"type": "response", "response": response_text})
                continue

            # ==== 4. GREETING ====
            if intent == "greeting":
                response_text = await call_llm([
                    {"role": "system", "content": (
                        "You are ShopBot, a friendly e-commerce AI assistant. "
                        "Respond warmly and briefly to the greeting. "
                        "Mention you can help with products, orders, returns, and company policies. "
                        "Be natural — not corporate. Max 3 sentences."
                    )},
                    {"role": "user", "content": user_input}
                ], temperature=0.6)
                await save_message(session_id, "assistant", response_text)
                await broadcast_to_admins({"type": "bot_message", "session_id": session_id, "message": response_text, "timestamp": datetime.utcnow().isoformat()})
                await websocket.send_json({"type": "response", "response": response_text})
                continue

            # ==== 5. VIEW CART ====
            if intent == "view_cart":
                cart = await get_cart(session_id)
                if not cart:
                    response_text = "Your cart is currently empty! 🛒 Browse some products and add the ones you like — I'm happy to help you find something great."
                else:
                    total = sum(c.price * c.quantity for c in cart)
                    items = "\n".join([f"• **{c.product_name}** × {c.quantity} — ${c.price:.2f}" for c in cart])
                    response_text = f"Here's what's in your cart right now:\n\n{items}\n\n💰 **Total: ${total:.2f}**\n\nReady to checkout, or would you like to keep browsing?"
                await save_message(session_id, "assistant", response_text)
                await broadcast_to_admins({"type": "bot_message", "session_id": session_id, "message": response_text, "timestamp": datetime.utcnow().isoformat()})
                await websocket.send_json({"type": "response", "response": response_text, "cart": [serialize_doc(c) for c in cart]})
                continue

            # ==== 6. COMPANY INFO ====
            if intent == "company_info":
                response_text = await generate_response(session_id, user_input, decision)
                await save_message(session_id, "assistant", response_text)
                await broadcast_to_admins({"type": "bot_message", "session_id": session_id, "message": response_text, "timestamp": datetime.utcnow().isoformat()})
                await websocket.send_json({"type": "response", "response": response_text})
                continue

            # ==== 7. ORDER STATUS / DELAY / REFUND / COMPLAINT ====
            if intent in ["order_status", "order_delay", "refund_return", "complaint"]:
                # Step 1: Need email?
                if not customer_email and decision.get("needs_email", True):
                    response_text = (
                        "I'd be happy to help you with that! To pull up your order details, "
                        "could you please share the **email address** you used when placing the order?"
                    )
                    pending = {"awaiting": "email", "intent": intent}
                    await save_message(session_id, "assistant", response_text)
                    await broadcast_to_admins({"type": "bot_message", "session_id": session_id, "message": response_text, "timestamp": datetime.utcnow().isoformat()})
                    await websocket.send_json({"type": "response", "response": response_text})
                    continue

                # Step 2: Have email — fetch orders
                orders = []
                if customer_email:
                    await websocket.send_json({"type": "status", "message": f"🔍 Looking up orders for {customer_email}..."})
                    orders = await mcp_get_orders_by_email(customer_email)

                response_text = await generate_response(session_id, user_input, decision, orders=orders, extra_context=f"Customer email: {customer_email}")
                await save_message(session_id, "assistant", response_text)
                await broadcast_to_admins({"type": "bot_message", "session_id": session_id, "message": response_text, "timestamp": datetime.utcnow().isoformat()})
                await websocket.send_json({
                    "type":     "response",
                    "response": response_text,
                    "orders":   orders
                })
                continue

            # ==== 8. PRODUCT SEARCH / DETAIL ====
            if intent in ["product_search", "product_detail"]:
                await websocket.send_json({"type": "status", "message": "🔍 Searching products for you..."})

                # Check if product ID was detected
                if decision.get("product_id_detected"):
                    pid     = decision["product_id_detected"]
                    product = await get_product_by_id(pid)
                    if product:
                        products = [product]
                        await save_session_product(session_id, str(product["id"]), product["title"], product.get("category",""))
                    else:
                        products = await search_products(user_input, limit=5)
                else:
                    category = decision.get("category_hint", "").strip()
                    if not category:
                        category = await detect_category(user_input)
                    products = []
                    if category:
                        products = await fetch_products_by_category(category, limit=5)
                    if not products:
                        products = await search_products(user_input, limit=5)

                for p in products[:3]:
                    await save_session_product(session_id, str(p["id"]), p["title"], p.get("category",""))

                response_text = await generate_response(session_id, user_input, decision, products=products)
                await save_message(session_id, "assistant", response_text)
                await broadcast_to_admins({"type": "bot_message", "session_id": session_id, "message": response_text, "timestamp": datetime.utcnow().isoformat()})
                prods = await get_session_products(session_id)
                await websocket.send_json({
                    "type":             "response",
                    "response":         response_text,
                    "products":         products[:10],
                    "session_products": [{"id": p.product_id, "name": p.product_name, "category": p.category} for p in prods]
                })
                continue

            # ==== 9. ADD TO CART ====
            if intent == "add_to_cart" or decision.get("add_to_cart_trigger"):
                await websocket.send_json({"type": "status", "message": "🛒 Adding to your cart..."})

                # Try to use session product first
                session_prods = await get_session_products(session_id)
                product       = None

                if decision.get("product_id_detected"):
                    product = await get_product_by_id(decision["product_id_detected"])

                if not product and session_prods:
                    # Use the most recently tracked product
                    last_prod = session_prods[-1]
                    product   = await get_product_by_id(last_prod.product_id)

                if not product:
                    category = decision.get("category_hint", "")
                    if not category:
                        category = await detect_category(user_input)
                    products_list = []
                    if category:
                        products_list = await fetch_products_by_category(category, limit=3)
                    if not products_list:
                        products_list = await search_products(user_input, limit=3)
                    if products_list:
                        product = products_list[0]

                if product:
                    msg = await add_to_cart(session_id, product)
                    await save_session_product(session_id, str(product["id"]), product["title"], product.get("category",""))
                    cart         = await get_cart(session_id)
                    cart_total   = sum(c.price * c.quantity for c in cart)
                    response_text = (
                        f"Done! 🛒 **{product['title']}** has been added to your cart.\n\n"
                        f"💰 Price: ${product['price']} | Cart total: ${cart_total:.2f} ({len(cart)} item{'s' if len(cart) > 1 else ''})\n\n"
                        "Want to keep shopping, or are you ready to checkout?"
                    )
                else:
                    response_text = "I wasn't sure which product you meant. Could you give me the product name or ID so I can add the right one?"

                await save_message(session_id, "assistant", response_text)
                await broadcast_to_admins({"type": "bot_message", "session_id": session_id, "message": response_text, "timestamp": datetime.utcnow().isoformat()})
                cart = await get_cart(session_id)
                await websocket.send_json({
                    "type":     "response",
                    "response": response_text,
                    "cart":     [c.dict() for c in cart]
                })
                continue

            # ==== 10. PROVIDE EMAIL (user typed just their email) ====
            if intent == "provide_email" or decision.get("email_detected"):
                email_val = decision.get("email_detected", user_input).strip().lower()
                if "@" in email_val:
                    customer_email = email_val
                    pending        = {}
                    await websocket.send_json({"type": "status", "message": f"🔍 Looking up orders for {customer_email}..."})
                    orders = await mcp_get_orders_by_email(customer_email)
                    if orders:
                        response_text = (
                            f"Found your account! Here are your orders for **{customer_email}**:\n\n"
                            f"{format_orders_for_display(orders, customer_email)}\n\n"
                            "Is there anything specific you need help with?"
                        )
                    else:
                        response_text = f"Thanks! I looked up **{customer_email}** but couldn't find any orders. Are you sure this is the email you used? I can also connect you with a human agent if needed."
                    await save_message(session_id, "assistant", response_text)
                    await broadcast_to_admins({"type": "bot_message", "session_id": session_id, "message": response_text, "timestamp": datetime.utcnow().isoformat()})
                    await websocket.send_json({"type": "response", "response": response_text, "orders": orders})
                    continue

            # ==== FALLBACK ====
            response_text = await generate_response(session_id, user_input, decision)
            await save_message(session_id, "assistant", response_text)
            await broadcast_to_admins({"type": "bot_message", "session_id": session_id, "message": response_text, "timestamp": datetime.utcnow().isoformat()})
            await websocket.send_json({"type": "response", "response": response_text})

    except WebSocketDisconnect:
        active_connections.pop(session_id, None)
        await broadcast_to_admins({
            "type":       "session_disconnected",
            "session_id": session_id,
            "message":    f"Customer {session_id[:8]} disconnected."
        })
    except Exception as e:
        print(f"[ERROR] session={session_id} error={e}")
        try:
            await websocket.send_json({"type": "error", "message": "Something went wrong on my end. Please try again!"})
        except Exception:
            pass

# ==========================
# 🔹 WEBSOCKET — ADMIN (live monitoring + reply)
# ==========================
@app.websocket("/ws/admin")
async def admin_ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    admin_id = str(uuid.uuid4())
    admin_connections[admin_id] = websocket

    await websocket.send_json({
        "type":     "admin_connected",
        "admin_id": admin_id,
        "message":  "✅ Admin panel connected. You will receive live customer messages."
    })

    # Send currently active sessions
    active_sids = list(active_connections.keys())
    await websocket.send_json({
        "type":     "active_sessions",
        "sessions": active_sids
    })

    try:
        while True:
            data = await websocket.receive_json()
            msg_type   = data.get("type", "")
            session_id = data.get("session_id", "")
            message    = data.get("message", "").strip()

            # Admin sends message to customer
            if msg_type == "admin_reply" and session_id and message:
                await save_message(session_id, "admin", message)
                customer_ws = active_connections.get(session_id)
                if customer_ws:
                    try:
                        await customer_ws.send_json({
                            "type":     "admin_message",
                            "response": f"👤 **Support Agent:** {message}",
                            "role":     "admin"
                        })
                    except Exception:
                        pass

                # Broadcast to all other admins too
                for aid, aws in admin_connections.items():
                    if aid != admin_id:
                        try:
                            await aws.send_json({
                                "type":       "admin_echo",
                                "session_id": session_id,
                                "message":    message,
                                "admin_id":   admin_id
                            })
                        except Exception:
                            pass

            # Admin takes over a session (disable AI)
            elif msg_type == "take_over" and session_id:
                human_takeover_sessions[session_id] = True
                session_admin_map[session_id] = admin_id
                customer_ws = active_connections.get(session_id)
                if customer_ws:
                    try:
                        await customer_ws.send_json({
                            "type":     "human_takeover",
                            "response": "👤 You're now connected with a live support specialist. How can I help you?",
                        })
                    except Exception:
                        pass
                await websocket.send_json({"type": "take_over_confirmed", "session_id": session_id})

            # Admin releases session back to AI
            elif msg_type == "release_to_ai" and session_id:
                human_takeover_sessions.pop(session_id, None)
                session_admin_map.pop(session_id, None)
                customer_ws = active_connections.get(session_id)
                if customer_ws:
                    try:
                        await customer_ws.send_json({
                            "type":     "ai_restored",
                            "response": "🤖 You're back with ShopBot AI. How can I continue helping you?",
                        })
                    except Exception:
                        pass
                await websocket.send_json({"type": "ai_restored", "session_id": session_id})

            # Admin requests full chat history of a session
            elif msg_type == "get_history" and session_id:
                msgs = await ChatMessage.find(
                    ChatMessage.session_id == session_id
                ).sort("timestamp").to_list()
                history = [{"role": m.role, "message": m.message, "timestamp": m.timestamp.isoformat()} for m in msgs]
                await websocket.send_json({
                    "type":     "session_history",
                    "session_id": session_id,
                    "history":  history
                })

    except WebSocketDisconnect:
        admin_connections.pop(admin_id, None)


# ==========================
# 🔹 REST APIs
# ==========================
class ProductIn(BaseModel):
    product_name: str
    description: str
    price: float
    category: str
    stock: int = 0

class CompanyKnowledgeIn(BaseModel):
    text: str
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class OrderIn(BaseModel):
    email: str
    product_name: str
    product_id: str
    status: str = "processing"
    price: float = 0.0
    delivery_date: Optional[str] = None
    notes: str = ""

@app.post("/admin/products")
async def admin_add_product(product: ProductIn):
    doc = AdminProduct(**product.dict())
    await doc.insert()
    return {"status": "created", "id": str(doc.id)}

@app.get("/admin/products")
async def admin_list_products():
    products = await AdminProduct.find_all().to_list()
    return [{"id": str(p.id), "name": p.product_name, "price": p.price, "category": p.category, "stock": p.stock} for p in products]

@app.post("/admin/company-knowledge")
async def admin_add_company_knowledge(payload: CompanyKnowledgeIn):
    company_collection.add(documents=[payload.text], ids=[payload.doc_id])
    return {"status": "added", "doc_id": payload.doc_id}

@app.post("/admin/orders")
async def admin_add_order(order: OrderIn):
    delivery_dt = None
    if order.delivery_date:
        try:
            delivery_dt = datetime.strptime(order.delivery_date, "%Y-%m-%d")
        except Exception:
            pass
    doc = CustomerOrder(
        email=order.email.lower().strip(),
        product_name=order.product_name,
        product_id=order.product_id,
        status=order.status,
        price=order.price,
        delivery_date=delivery_dt,
        notes=order.notes
    )
    await doc.insert()
    return {"status": "created", "id": str(doc.id)}

@app.get("/admin/orders")
async def admin_list_orders(email: Optional[str] = None):
    if email:
        orders = await CustomerOrder.find(CustomerOrder.email == email.lower().strip()).sort("-order_date").to_list()
    else:
        orders = await CustomerOrder.find_all().sort("-order_date").to_list()
    return [{
        "id":            str(o.id),
        "email":         o.email,
        "product_name":  o.product_name,
        "product_id":    o.product_id,
        "status":        o.status,
        "order_date":    o.order_date.isoformat(),
        "delivery_date": o.delivery_date.isoformat() if o.delivery_date else None,
        "price":         o.price,
        "notes":         o.notes
    } for o in orders]

@app.get("/admin/sessions")
async def admin_sessions():
    summaries = await ChatSummary.find_all().sort("-last_updated").to_list()
    result = []
    for s in summaries:
        prods = await SessionProduct.find(SessionProduct.session_id == s.session_id).to_list()
        cart  = await CartItem.find(CartItem.session_id == s.session_id).to_list()
        is_active = s.session_id in active_connections
        result.append({
            "session_id":       s.session_id,
            "summary":          s.summary,
            "message_count":    s.message_count,
            "last_updated":     s.last_updated.isoformat(),
            "is_active":        is_active,
            "human_takeover":   human_takeover_sessions.get(s.session_id, False),
            "products_viewed":  [{"id": p.product_id, "name": p.product_name} for p in prods],
            "cart_items":       [{"id": c.product_id, "name": c.product_name, "qty": c.quantity} for c in cart]
        })
    return result


# ==========================
# 🔹 FRONTEND — Customer Chat
# ==========================
@app.get("/")
async def get_frontend():
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ShopBot AI • Shopping Assistant</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
  * { box-sizing: border-box; }
  body { font-family: 'Inter', sans-serif; background: #0a0f1e; color: #e2e8f0; }
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-thumb { background: #334155; border-radius: 10px; }
  .glass { background: rgba(255,255,255,0.04); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.08); }
  .user-bubble { background: linear-gradient(135deg, #3b82f6, #6366f1); color: white; border-radius: 18px 18px 4px 18px; }
  .bot-bubble   { background: #1e293b; color: #e2e8f0; border-radius: 18px 18px 18px 4px; border: 1px solid #334155; }
  .admin-bubble { background: linear-gradient(135deg, #059669, #0284c7); color: white; border-radius: 18px 18px 18px 4px; }
  .human-banner { background: linear-gradient(135deg, #dc2626, #f97316); border-radius: 10px; padding: 10px 16px; font-size:13px; margin: 8px 0; }
  .product-card { background: #1e293b; border: 1px solid #334155; border-radius: 14px; overflow:hidden; transition: all 0.2s; cursor: pointer; }
  .product-card:hover { border-color: #3b82f6; transform: translateY(-2px); box-shadow: 0 8px 25px rgba(59,130,246,0.15); }
  .send-btn { background: linear-gradient(135deg, #3b82f6, #6366f1); }
  .send-btn:hover { opacity: 0.9; transform: scale(1.05); }
  .typing-dot { animation: bounce 0.8s infinite; }
  .typing-dot:nth-child(2){animation-delay:.1s} .typing-dot:nth-child(3){animation-delay:.2s}
  @keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px)}}
  .status-pulse{width:8px;height:8px;border-radius:50%;background:#22c55e;animation:pls 2s infinite}
  @keyframes pls{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.5;transform:scale(1.3)}}
  .tab-btn{padding:8px 18px;border-radius:8px;font-size:13px;font-weight:500;transition:all .2s}
  .tab-btn.active{background:#3b82f6;color:white}
  .tab-btn:not(.active){color:#94a3b8}
  .tab-btn:not(.active):hover{color:white}
  strong { font-weight: 600; }
</style>
</head>
<body class="min-h-screen">
<div class="flex h-[100dvh] overflow-hidden">

  <!-- SIDEBAR -->
  <div class="w-64 glass border-r border-white/5 flex flex-col p-4 gap-4 flex-shrink-0">
    <div class="flex items-center gap-3 px-1 pt-2">
      <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-violet-600 rounded-xl flex items-center justify-center text-xl">🛍️</div>
      <div>
        <h1 class="font-bold text-lg leading-tight">ShopBot AI</h1>
        <p class="text-xs text-emerald-400 flex items-center gap-1.5"><span class="status-pulse"></span> Online</p>
      </div>
    </div>

    <div>
      <p class="text-xs text-slate-400 font-semibold uppercase tracking-wider mb-2">Quick Actions</p>
      <div class="space-y-1">
        <button onclick="sendQuick('Show me my cart')" class="w-full text-left px-3 py-2 rounded-lg text-sm text-slate-300 hover:bg-slate-700 transition flex items-center gap-2">🛒 My Cart <span id="cart-badge" class="ml-auto bg-blue-600 text-xs px-2 py-0.5 rounded-full hidden">0</span></button>
        <button onclick="sendQuick('I want to connect with your support agent')" class="w-full text-left px-3 py-2 rounded-lg text-sm text-slate-300 hover:bg-slate-700 transition">👤 Talk to Human</button>
        <button onclick="sendQuick('What is your return policy?')" class="w-full text-left px-3 py-2 rounded-lg text-sm text-slate-300 hover:bg-slate-700 transition">📋 Return Policy</button>
      </div>
    </div>

    <div>
      <p class="text-xs text-slate-400 font-semibold uppercase tracking-wider mb-2">Browsed Products</p>
      <div id="session-products" class="space-y-1 text-xs text-slate-500 italic">None yet</div>
    </div>

    <div class="mt-auto text-xs font-mono text-slate-600 px-1 truncate" id="session-id-display"></div>
  </div>

  <!-- MAIN CHAT -->
  <div class="flex-1 flex flex-col min-w-0">
    <div id="human-banner" class="hidden mx-4 mt-3 human-banner text-white flex items-center gap-2">
      <i class="fa-solid fa-headset"></i> <span id="human-banner-text">You're connected with a live support agent</span>
    </div>

    <div id="messages" class="flex-1 overflow-y-auto p-5 space-y-4 min-h-0"></div>

    <div id="typing-indicator" class="hidden px-6 pb-2">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-violet-600 rounded-full flex items-center justify-center text-sm">🤖</div>
        <div class="bot-bubble px-4 py-3 flex gap-1 items-center">
          <span class="typing-dot w-2 h-2 bg-slate-400 rounded-full"></span>
          <span class="typing-dot w-2 h-2 bg-slate-400 rounded-full"></span>
          <span class="typing-dot w-2 h-2 bg-slate-400 rounded-full"></span>
        </div>
      </div>
    </div>

    <div id="status-bar" class="px-5 py-1 text-xs text-blue-400 font-medium min-h-[22px]"></div>

    <div class="p-4 border-t border-white/5 glass shrink-0">
      <div class="flex gap-3 max-w-3xl mx-auto">
        <input id="user-input" type="text" placeholder="Type your message..."
          class="flex-1 bg-slate-800 text-white placeholder:text-slate-500 rounded-2xl px-5 py-3.5 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
          onkeydown="if(event.key==='Enter') sendMessage()">
        <button onclick="sendMessage()" class="send-btn w-12 h-12 rounded-2xl flex items-center justify-center transition-all flex-shrink-0">
          <i class="fa-solid fa-paper-plane text-white"></i>
        </button>
      </div>
    </div>
  </div>
</div>

<script>
let ws = null, sessionId = "", cartCount = 0, isHumanMode = false;

function scrollBot() { const c = document.getElementById("messages"); setTimeout(()=>c.scrollTop=c.scrollHeight,80); }

function fmt(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.*?)\*/g,'<em>$1</em>')
    .replace(/\n/g,'<br>');
}

function appendMessage(role, text, extra={}) {
  const wrap = document.getElementById("messages");
  const div  = document.createElement("div");
  div.className = `flex ${role==='user'?'justify-end':'justify-start'} items-start gap-3 mb-5`;

  const isUser  = role === 'user';
  const isAdmin = role === 'admin';
  const bubClass = isUser ? 'user-bubble' : isAdmin ? 'admin-bubble' : 'bot-bubble';
  const ava  = isUser ? '👤' : isAdmin ? '🎧' : '🤖';
  const avatarEl = `<div class="w-8 h-8 ${isUser?'bg-slate-700':'bg-gradient-to-br from-blue-500 to-violet-600'} rounded-full flex items-center justify-center text-sm flex-shrink-0 mt-1">${ava}</div>`;

  let html = `<div class="max-w-[78%]"><div class="${bubClass} px-5 py-3.5 text-[14px] leading-relaxed">${fmt(text)}</div>`;

  if (extra.products && extra.products.length > 0) {
    html += `<div class="grid grid-cols-2 md:grid-cols-3 gap-3 mt-3">` +
      extra.products.slice(0,6).map(p=>`
        <div onclick="addToCartQuick(${p.id},'${(p.title||'').replace(/'/g,"\\'")}','${p.category||''}')" class="product-card">
          <img src="${p.thumbnail||''}" class="w-full h-28 object-cover" onerror="this.src='https://placehold.co/300x200/1e293b/94a3b8?text=No+Image'">
          <div class="p-3">
            <p class="font-medium text-sm leading-snug">${p.title}</p>
            <p class="text-blue-400 font-bold mt-1">$${p.price}</p>
            <p class="text-xs text-slate-400 mt-1 truncate">${p.description||''}</p>
            <button class="mt-2 w-full bg-blue-600 hover:bg-blue-500 py-1.5 rounded-lg text-xs font-medium transition">+ Add to Cart</button>
          </div>
        </div>`).join('') + `</div>`;
  }

  if (extra.orders && extra.orders.length > 0) {
    html += `<div class="mt-3 space-y-2">` +
      extra.orders.map(o => {
        const em  = {delivered:'✅',shipped:'🚚',processing:'⏳',cancelled:'❌'}[o.status.toLowerCase()]||'📦';
        const tag = o.is_delivered ? '<span class="ml-2 bg-emerald-900 text-emerald-300 text-xs px-2 py-0.5 rounded-full">Delivered</span>' : '';
        return `<div class="bg-slate-800 rounded-xl p-3 text-sm border border-slate-700">
          <div class="font-medium">${em} ${o.product_name}${tag}</div>
          <div class="text-xs text-slate-400 mt-1">Status: ${o.status.toUpperCase()} | Date: ${o.order_date} | $${o.price}</div>
          ${o.notes?`<div class="text-xs text-slate-500 mt-1">Note: ${o.notes}</div>`:''}
        </div>`;
      }).join('') + `</div>`;
  }

  html += `</div>`;
  div.innerHTML = `${!isUser?avatarEl:''}${html}${isUser?avatarEl:''}`;
  wrap.appendChild(div);
  scrollBot();
}

function showTyping(v) { document.getElementById("typing-indicator").classList.toggle("hidden",!v); }
function setStatus(m)  { const el=document.getElementById("status-bar"); el.textContent=m; if(m) setTimeout(()=>el.textContent='',4000); }

function sendMessage() {
  const inp = document.getElementById("user-input");
  const txt = inp.value.trim();
  if (!txt || !ws) return;
  appendMessage("user", txt);
  inp.value = "";
  if (!isHumanMode) showTyping(true);
  ws.send(JSON.stringify({message: txt}));
}

function sendQuick(msg) {
  document.getElementById("user-input").value = msg;
  sendMessage();
}

function addToCartQuick(id, name, cat) {
  sendQuick(`Add product ${name} (ID: ${id}) to my cart`);
}

function updateCartBadge(count) {
  const badge = document.getElementById("cart-badge");
  badge.textContent = count;
  if (count > 0) badge.classList.remove("hidden");
  else badge.classList.add("hidden");
}

function updateSessionProducts(prods) {
  const el = document.getElementById("session-products");
  if (!prods || prods.length === 0) { el.innerHTML = '<span class="italic">None yet</span>'; return; }
  el.innerHTML = prods.slice(-5).reverse().map(p=>
    `<div class="py-0.5 text-slate-300 truncate" title="${p.name}">• ${p.name}</div>`
  ).join('');
}

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/chat`);

  ws.onopen = () => console.log("WS Connected");

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    showTyping(false);

    if (data.session_id && !sessionId) {
      sessionId = data.session_id;
      document.getElementById("session-id-display").textContent = `ID: ${sessionId.slice(0,8)}…`;
    }

    if (data.type === "status") { setStatus(data.message); return; }

    if (data.type === "connected") {
      appendMessage("bot", data.message);
      return;
    }

    if (data.type === "human_takeover") {
      isHumanMode = true;
      document.getElementById("human-banner").classList.remove("hidden");
      appendMessage("bot", data.response || data.message);
      return;
    }

    if (data.type === "ai_restored") {
      isHumanMode = false;
      document.getElementById("human-banner").classList.add("hidden");
      appendMessage("bot", data.response || "🤖 ShopBot AI is back online!");
      return;
    }

    if (data.type === "human_active") {
      appendMessage("bot", data.message);
      return;
    }

    if (data.type === "admin_message") {
      appendMessage("admin", data.response);
      return;
    }

    if (data.type === "response") {
      appendMessage("bot", data.response, {products: data.products||[], orders: data.orders||[]});
      if (data.cart) updateCartBadge(data.cart.length);
      if (data.session_products) updateSessionProducts(data.session_products);
      return;
    }
  };

  ws.onclose = () => setTimeout(connectWS, 2500);
  ws.onerror = () => ws.close();
}

window.onload = connectWS;
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


# ==========================
# 🔹 ADMIN DASHBOARD — Full Panel
# ==========================
@app.get("/admin")
async def get_admin_panel():
    html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ShopBot Admin Panel</title>
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<style>
  * { box-sizing: border-box; }
  body { font-family:'Inter',sans-serif; background:#0a0f1e; color:#e2e8f0; }
  ::-webkit-scrollbar{width:5px} ::-webkit-scrollbar-thumb{background:#334155;border-radius:10px}
  .glass{background:rgba(255,255,255,0.04);backdrop-filter:blur(12px);border:1px solid rgba(255,255,255,0.08)}
  .session-card{background:#1e293b;border:1px solid #334155;border-radius:12px;cursor:pointer;transition:all .2s}
  .session-card:hover, .session-card.selected{border-color:#3b82f6}
  .session-card.active-session{border-left:3px solid #22c55e}
  .session-card.takeover-session{border-left:3px solid #f97316}
  .msg-user{background:linear-gradient(135deg,#3b82f6,#6366f1);color:white;border-radius:12px 12px 4px 12px;margin-left:auto}
  .msg-bot{background:#1e293b;border:1px solid #334155;border-radius:12px 12px 12px 4px}
  .msg-admin{background:linear-gradient(135deg,#059669,#0284c7);color:white;border-radius:12px 12px 12px 4px}
  .tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px}
  .tab-active{background:#3b82f6;color:white}
  input,textarea,select{background:#1e293b;border:1px solid #334155;color:#e2e8f0;border-radius:10px;padding:8px 12px;width:100%;outline:none;font-size:13px}
  input:focus,textarea:focus{border-color:#3b82f6}
  .btn{padding:8px 16px;border-radius:10px;font-size:13px;font-weight:500;cursor:pointer;transition:all .2s}
  .btn-blue{background:#3b82f6;color:white} .btn-blue:hover{background:#2563eb}
  .btn-green{background:#059669;color:white} .btn-green:hover{background:#047857}
  .btn-red{background:#dc2626;color:white} .btn-red:hover{background:#b91c1c}
  .btn-orange{background:#f97316;color:white} .btn-orange:hover{background:#ea580c}
  .btn-slate{background:#334155;color:#e2e8f0} .btn-slate:hover{background:#475569}
  .notif{position:fixed;top:20px;right:20px;z-index:999;padding:12px 20px;border-radius:12px;font-size:13px;font-weight:500;animation:slide-in .3s ease}
  @keyframes slide-in{from{transform:translateX(100px);opacity:0}to{transform:translateX(0);opacity:1}}
</style>
</head>
<body class="min-h-screen">

<div class="flex h-[100dvh] overflow-hidden">

  <!-- LEFT: Session List -->
  <div class="w-72 glass border-r border-white/5 flex flex-col p-3 gap-3 flex-shrink-0">
    <div class="flex items-center gap-2 px-1 pt-1">
      <div class="w-9 h-9 bg-gradient-to-br from-orange-500 to-red-600 rounded-xl flex items-center justify-center text-lg">⚙️</div>
      <div>
        <h1 class="font-bold text-base">Admin Panel</h1>
        <p id="admin-status" class="text-xs text-slate-500">Connecting...</p>
      </div>
    </div>

    <div class="flex gap-1">
      <button onclick="showView('sessions')" class="btn btn-blue flex-1 text-xs py-1.5">💬 Chats</button>
      <button onclick="showView('manage')" class="btn btn-slate flex-1 text-xs py-1.5">🗃️ Manage</button>
    </div>

    <div>
      <div class="flex items-center justify-between mb-2">
        <p class="text-xs text-slate-400 font-semibold uppercase tracking-wider">Live Sessions</p>
        <span id="session-count" class="tag bg-slate-800 text-slate-400">0</span>
      </div>
      <div id="session-list" class="space-y-2 overflow-y-auto max-h-[calc(100vh-200px)]">
        <p class="text-xs text-slate-500 italic px-2">No active sessions</p>
      </div>
    </div>
  </div>

  <!-- MAIN AREA -->
  <div class="flex-1 flex flex-col min-w-0">

    <!-- SESSIONS VIEW: Chat Pane -->
    <div id="view-sessions" class="flex-1 flex flex-col">

      <!-- Chat Header -->
      <div id="chat-header" class="hidden px-5 py-3 border-b border-white/5 glass flex items-center justify-between flex-shrink-0">
        <div>
          <h2 class="font-semibold" id="chat-title">Select a session</h2>
          <p class="text-xs text-slate-400" id="chat-subtitle"></p>
        </div>
        <div class="flex gap-2" id="chat-actions">
          <button onclick="takeOver()" id="btn-takeover" class="btn btn-orange text-xs hidden">
            <i class="fa-solid fa-headset mr-1"></i> Take Over
          </button>
          <button onclick="releaseToAI()" id="btn-release" class="btn btn-green text-xs hidden">
            <i class="fa-solid fa-robot mr-1"></i> Release to AI
          </button>
          <button onclick="refreshHistory()" class="btn btn-slate text-xs">
            <i class="fa-solid fa-rotate-right"></i> Refresh
          </button>
        </div>
      </div>

      <!-- No session selected -->
      <div id="no-session-msg" class="flex-1 flex items-center justify-center text-slate-500">
        <div class="text-center">
          <div class="text-5xl mb-4">💬</div>
          <p class="text-lg font-medium">Select a customer session</p>
          <p class="text-sm mt-1">Live messages will appear here</p>
        </div>
      </div>

      <!-- Chat History -->
      <div id="chat-pane" class="flex-1 overflow-y-auto p-4 space-y-3 hidden min-h-0"></div>

      <!-- Admin Reply Box -->
      <div id="reply-box" class="hidden p-4 border-t border-white/5 glass shrink-0">
        <div class="flex gap-2">
          <input id="admin-reply-input" type="text" placeholder="Type your reply to the customer..."
            onkeydown="if(event.key==='Enter') adminSend()"
            class="flex-1 text-sm py-3 px-4 rounded-xl">
          <button onclick="adminSend()" class="btn btn-green px-5">Send</button>
        </div>
      </div>
    </div>

    <!-- MANAGE VIEW -->
    <div id="view-manage" class="flex-1 overflow-y-auto p-5 hidden">
      <h2 class="text-lg font-bold mb-5">🗃️ Manage Store Data</h2>
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-5">

        <!-- Add Order (MCP Tool) -->
        <div class="glass rounded-2xl p-5">
          <h3 class="font-semibold mb-4 flex items-center gap-2 text-emerald-400"><i class="fa-solid fa-box"></i> Add Customer Order</h3>
          <div class="space-y-2">
            <input id="o-email"     placeholder="Customer Email *" required>
            <input id="o-pname"     placeholder="Product Name *">
            <input id="o-pid"       placeholder="Product ID">
            <select id="o-status">
              <option value="processing">Processing</option>
              <option value="shipped">Shipped</option>
              <option value="delivered">Delivered</option>
              <option value="cancelled">Cancelled</option>
            </select>
            <div class="flex gap-2">
              <input id="o-price" type="number" placeholder="Price ($)" class="flex-1">
              <input id="o-date"  type="date" class="flex-1">
            </div>
            <textarea id="o-notes" rows="2" placeholder="Internal notes (optional)"></textarea>
            <button onclick="addOrder()" class="btn btn-green w-full">Add Order</button>
            <div id="o-msg" class="text-xs mt-1"></div>
          </div>
        </div>

        <!-- Search Orders by Email -->
        <div class="glass rounded-2xl p-5">
          <h3 class="font-semibold mb-4 flex items-center gap-2 text-blue-400"><i class="fa-solid fa-search"></i> Search Orders by Email</h3>
          <div class="space-y-2">
            <div class="flex gap-2">
              <input id="search-email" placeholder="customer@email.com" class="flex-1">
              <button onclick="searchOrders()" class="btn btn-blue">Search</button>
            </div>
            <div id="order-results" class="space-y-2 max-h-64 overflow-y-auto mt-2"></div>
          </div>
        </div>

        <!-- Add Product -->
        <div class="glass rounded-2xl p-5">
          <h3 class="font-semibold mb-4 flex items-center gap-2 text-violet-400"><i class="fa-solid fa-tag"></i> Add Product to DB</h3>
          <div class="space-y-2">
            <input id="p-name" placeholder="Product Name *">
            <textarea id="p-desc" rows="2" placeholder="Description"></textarea>
            <div class="flex gap-2">
              <input id="p-price" type="number" placeholder="Price" class="flex-1">
              <input id="p-stock" type="number" placeholder="Stock" class="flex-1">
            </div>
            <input id="p-cat" placeholder="Category (e.g. smartphones)">
            <button onclick="addProduct()" class="btn btn-blue w-full">Add Product</button>
            <div id="p-msg" class="text-xs mt-1"></div>
          </div>
        </div>

        <!-- Company Knowledge -->
        <div class="glass rounded-2xl p-5">
          <h3 class="font-semibold mb-4 flex items-center gap-2 text-orange-400"><i class="fa-solid fa-brain"></i> Train Knowledge Base</h3>
          <div class="space-y-2">
            <textarea id="k-text" rows="5" placeholder="e.g. We offer 30-day no-questions-asked returns. Free shipping on orders above $50..."></textarea>
            <button onclick="addKnowledge()" class="btn btn-orange w-full">Add to Vector DB</button>
            <div id="k-msg" class="text-xs mt-1"></div>
          </div>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
let adminWs = null, selectedSession = null, sessions = {}, adminId = "";
const takeovers = {};

function showView(v) {
  document.getElementById('view-sessions').classList.toggle('hidden', v !== 'sessions');
  document.getElementById('view-manage').classList.toggle('hidden', v !== 'manage');
}

function notify(msg, color='blue') {
  const colors = {blue:'#3b82f6',green:'#059669',red:'#dc2626',orange:'#f97316'};
  const n = document.createElement('div');
  n.className = 'notif';
  n.style.background = colors[color]||colors.blue;
  n.textContent = msg;
  document.body.appendChild(n);
  setTimeout(()=>n.remove(), 4000);
}

function renderSessionList() {
  const el = document.getElementById('session-list');
  const ids = Object.keys(sessions);
  document.getElementById('session-count').textContent = ids.length;
  if (!ids.length) { el.innerHTML='<p class="text-xs text-slate-500 italic px-2">No active sessions</p>'; return; }

  el.innerHTML = ids.map(sid => {
    const s = sessions[sid];
    const isActive   = s.isActive;
    const isTakeover = takeovers[sid];
    const label      = sid.slice(0,8);
    const lastMsg    = (s.lastMsg||'').slice(0,40);
    return `<div onclick="selectSession('${sid}')" class="session-card p-3 ${isActive?'active-session':''} ${isTakeover?'takeover-session':''} ${selectedSession===sid?'selected':''}">
      <div class="flex items-center justify-between">
        <span class="font-mono text-xs text-emerald-400">${label}…</span>
        <div class="flex gap-1">
          ${isActive?'<span class="tag bg-emerald-900 text-emerald-300">Live</span>':'<span class="tag bg-slate-700 text-slate-400">Offline</span>'}
          ${isTakeover?'<span class="tag bg-orange-900 text-orange-300">Human</span>':''}
        </div>
      </div>
      ${lastMsg?`<p class="text-xs text-slate-400 mt-1 truncate">${lastMsg}</p>`:''}
    </div>`;
  }).join('');
}

function selectSession(sid) {
  selectedSession = sid;
  renderSessionList();
  document.getElementById('no-session-msg').classList.add('hidden');
  document.getElementById('chat-pane').classList.remove('hidden');
  document.getElementById('chat-header').classList.remove('hidden');
  document.getElementById('reply-box').classList.remove('hidden');
  document.getElementById('chat-title').textContent = `Session: ${sid.slice(0,8)}…`;
  document.getElementById('chat-subtitle').textContent = sessions[sid]?.isActive ? '🟢 Active' : '⚫ Offline';

  const isTakeover = takeovers[sid];
  document.getElementById('btn-takeover').classList.toggle('hidden', !!isTakeover);
  document.getElementById('btn-release').classList.toggle('hidden', !isTakeover);

  // Load history
  if (adminWs && adminWs.readyState === 1)
    adminWs.send(JSON.stringify({type:'get_history', session_id: sid}));
}

function renderHistory(history) {
  const pane = document.getElementById('chat-pane');
  pane.innerHTML = '';
  history.forEach(m => appendChatMsg(m.role, m.message, m.timestamp));
  pane.scrollTop = pane.scrollHeight;
}

function appendChatMsg(role, msg, ts='') {
  if (selectedSession === null) return;
  const pane = document.getElementById('chat-pane');
  const div  = document.createElement('div');
  const cls  = role==='user'?'msg-user':role==='admin'?'msg-admin':'msg-bot';
  const align= role==='user'?'flex justify-end':'flex justify-start';
  const time = ts ? new Date(ts).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}) : '';
  div.className = align;
  div.innerHTML = `<div class="${cls} px-4 py-2.5 max-w-[75%] text-sm">
    <div class="text-xs opacity-60 mb-1">${role.toUpperCase()}${time?' · '+time:''}</div>
    ${msg.replace(/\n/g,'<br>')}
  </div>`;
  pane.appendChild(div);
  pane.scrollTop = pane.scrollHeight;
}

function takeOver() {
  if (!selectedSession || !adminWs) return;
  adminWs.send(JSON.stringify({type:'take_over', session_id: selectedSession}));
  takeovers[selectedSession] = true;
  document.getElementById('btn-takeover').classList.add('hidden');
  document.getElementById('btn-release').classList.remove('hidden');
  notify('You are now handling this session', 'orange');
}

function releaseToAI() {
  if (!selectedSession || !adminWs) return;
  adminWs.send(JSON.stringify({type:'release_to_ai', session_id: selectedSession}));
  delete takeovers[selectedSession];
  document.getElementById('btn-takeover').classList.remove('hidden');
  document.getElementById('btn-release').classList.add('hidden');
  notify('Session released back to AI', 'green');
}

function refreshHistory() {
  if (!selectedSession || !adminWs) return;
  adminWs.send(JSON.stringify({type:'get_history', session_id: selectedSession}));
}

function adminSend() {
  const inp = document.getElementById('admin-reply-input');
  const msg = inp.value.trim();
  if (!msg || !selectedSession || !adminWs) return;
  adminWs.send(JSON.stringify({type:'admin_reply', session_id: selectedSession, message: msg}));
  appendChatMsg('admin', msg);
  inp.value = '';
}

async function addOrder() {
  const email   = document.getElementById('o-email').value.trim();
  const pname   = document.getElementById('o-pname').value.trim();
  const pid     = document.getElementById('o-pid').value.trim();
  const status  = document.getElementById('o-status').value;
  const price   = parseFloat(document.getElementById('o-price').value)||0;
  const ddate   = document.getElementById('o-date').value;
  const notes   = document.getElementById('o-notes').value.trim();
  const msgEl   = document.getElementById('o-msg');
  if (!email||!pname) { msgEl.textContent='⚠️ Email and Product Name are required'; msgEl.style.color='#f97316'; return; }
  msgEl.textContent='Adding...'; msgEl.style.color='#94a3b8';
  const res = await fetch('/admin/orders',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({email,product_name:pname,product_id:pid||'N/A',status,price,delivery_date:ddate||null,notes})});
  if(res.ok){ msgEl.textContent='✅ Order added!'; msgEl.style.color='#22c55e'; }
  else      { msgEl.textContent='❌ Failed'; msgEl.style.color='#ef4444'; }
}

async function searchOrders() {
  const email = document.getElementById('search-email').value.trim();
  const el    = document.getElementById('order-results');
  if (!email) return;
  el.innerHTML='<p class="text-xs text-slate-400">Searching...</p>';
  const res  = await fetch(`/admin/orders?email=${encodeURIComponent(email)}`);
  const data = await res.json();
  if (!data.length) { el.innerHTML='<p class="text-xs text-slate-500 italic">No orders found.</p>'; return; }
  const statusIcon={delivered:'✅',shipped:'🚚',processing:'⏳',cancelled:'❌'};
  el.innerHTML=data.map(o=>`
    <div class="bg-slate-800 rounded-lg p-3 text-xs border border-slate-700">
      <div class="font-medium text-sm">${statusIcon[o.status]||'📦'} ${o.product_name}</div>
      <div class="text-slate-400 mt-1">Status: <strong>${o.status.toUpperCase()}</strong> | Date: ${o.order_date?.split('T')[0]||'N/A'} | $${o.price}</div>
      ${o.delivery_date?`<div class="text-slate-400">Delivery: ${o.delivery_date.split('T')[0]}</div>`:''}
      ${o.notes?`<div class="text-slate-500 mt-1">${o.notes}</div>`:''}
    </div>`).join('');
}

async function addProduct() {
  const name  = document.getElementById('p-name').value.trim();
  const desc  = document.getElementById('p-desc').value.trim();
  const price = parseFloat(document.getElementById('p-price').value);
  const stock = parseInt(document.getElementById('p-stock').value)||0;
  const cat   = document.getElementById('p-cat').value.trim();
  const el    = document.getElementById('p-msg');
  if(!name||!price||!cat){el.textContent='⚠️ Name, price, category required';el.style.color='#f97316';return}
  el.textContent='Adding...';el.style.color='#94a3b8';
  const res=await fetch('/admin/products',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({product_name:name,description:desc,price,stock,category:cat})});
  if(res.ok){el.textContent='✅ Product added!';el.style.color='#22c55e';}
  else{el.textContent='❌ Failed';el.style.color='#ef4444';}
}

async function addKnowledge() {
  const text=document.getElementById('k-text').value.trim();
  const el=document.getElementById('k-msg');
  if(!text){el.textContent='⚠️ Enter knowledge text';el.style.color='#f97316';return}
  el.textContent='Adding...';el.style.color='#94a3b8';
  const res=await fetch('/admin/company-knowledge',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text})});
  if(res.ok){el.textContent='✅ Added to ChromaDB!';el.style.color='#22c55e';document.getElementById('k-text').value='';}
  else{el.textContent='❌ Failed';el.style.color='#ef4444';}
}

// Admin WebSocket
function connectAdminWS() {
  const proto = location.protocol==='https:'?'wss':'ws';
  adminWs = new WebSocket(`${proto}://${location.host}/ws/admin`);

  adminWs.onopen = () => {
    document.getElementById('admin-status').textContent = '🟢 Connected';
    document.getElementById('admin-status').style.color = '#22c55e';
  };

  adminWs.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'admin_connected') {
      adminId = data.admin_id;
    }

    if (data.type === 'active_sessions') {
      data.sessions.forEach(sid => {
        if (!sessions[sid]) sessions[sid] = {isActive:true, lastMsg:''};
      });
      renderSessionList();
    }

    if (data.type === 'new_session') {
      sessions[data.session_id] = {isActive:true, lastMsg:'New session started'};
      notify(`New customer: ${data.session_id.slice(0,8)}`, 'green');
      renderSessionList();
    }

    if (data.type === 'session_disconnected') {
      if (sessions[data.session_id]) sessions[data.session_id].isActive = false;
      renderSessionList();
    }customer_support_agent.zip

    if (data.type === 'customer_message') {
      if (!sessions[data.session_id]) sessions[data.session_id] = {isActive:true, lastMsg:''};
      sessions[data.session_id].lastMsg = data.message;
      renderSessionList();
      if (selectedSession === data.session_id) appendChatMsg('user', data.message, data.timestamp);
      else notify(`💬 ${data.session_id.slice(0,8)}: ${data.message.slice(0,40)}`, 'blue');
    }

    if (data.type === 'bot_message') {
      if (sessions[data.session_id]) sessions[data.session_id].lastMsg = '[Bot] '+data.message.slice(0,30);
      renderSessionList();
      if (selectedSession === data.session_id) appendChatMsg('assistant', data.message, data.timestamp);
    }

    if (data.type === 'takeover_request') {
      notify(`🔴 ${data.message}`, 'red');
      if (!sessions[data.session_id]) sessions[datcustomer_support_agent.zipa.session_id]={isActive:true,lastMsg:''};
      renderSessionList();
    }

    if (data.type === 'session_history') {
      if (selectedSession === data.session_id) renderHistory(data.history);
    }

    if (data.type === 'take_over_confirmed') {
      takeovers[data.session_id] = true;
    }
  };

  adminWs.onclose = () => {
    document.getElementById('admin-status').textContent = '🔴 Disconnected';
    setTimeout(connectAdminWS, 2500);
  };
}

// Load existing sessions on start
async function loadExistingSessions() {
  const res  = await fetch('/admin/sessions');
  const data = await res.json();
  data.forEach(s => {
    sessions[s.session_id] = {isActive: s.is_active, lastMsg: s.summary?.slice(0,50)||''};
    if (s.human_takeover) takeovers[s.session_id] = true;
  });
  renderSessionList();
}

window.onload = () => { connectAdminWS(); loadExistingSessions(); };
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
