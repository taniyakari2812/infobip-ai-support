"""
AI Customer Support Assistant (prototype)
Flow: customer SMS -> Infobip webhook -> FastAPI -> LLM (answers from FAQ)
      (also works from the website chat box at "/")
      -> reply sent back via Infobip SMS API. Unknown questions go to a human.
"""
import os
import re
import json
import sqlite3
import time

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

INFOBIP_BASE_URL = os.getenv("INFOBIP_BASE_URL", "").replace("https://", "").strip("/")
INFOBIP_API_KEY = os.getenv("INFOBIP_API_KEY", "")
INFOBIP_SENDER = os.getenv("INFOBIP_SENDER", "InfoSMS")

# Any OpenAI-compatible LLM endpoint works (Groq, Gemini, OpenAI, OpenRouter...).
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

app = FastAPI(title="AI Support Assistant")

# ---------- Knowledge base (simple RAG: FAQ is placed in the prompt) ----------
with open("faq.json", encoding="utf-8") as f:
    FAQ = json.load(f)
FAQ_TEXT = "\n\n".join(f"Q: {item['q']}\nA: {item['a']}" for item in FAQ)

SYSTEM_PROMPT = f"""You are a friendly customer support assistant for a business.
Answer the customer's question using ONLY the FAQ below.
Keep replies short (under 300 characters) because they are sent by SMS.
If the FAQ does not contain the answer, or the customer is angry, or asks for a
human, reply with exactly the single word: HANDOFF

FAQ:
{FAQ_TEXT}
"""

HANDOFF_REPLY = "Thanks for your message. I'm connecting you to a human agent who will reply soon."

# ---------- Logging (SQLite) ----------
db = sqlite3.connect("conversations.db", check_same_thread=False)
db.execute(
    """CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, phone TEXT, message TEXT, reply TEXT,
        handoff INTEGER, latency_ms INTEGER, sms_status TEXT
    )"""
)
db.commit()


def log_conversation(phone, message, reply, handoff, latency_ms, sms_status):
    db.execute(
        "INSERT INTO logs (ts, phone, message, reply, handoff, latency_ms, sms_status) "
        "VALUES (?,?,?,?,?,?,?)",
        (time.strftime("%Y-%m-%d %H:%M:%S"), phone, message, reply,
         int(handoff), latency_ms, sms_status),
    )
    db.commit()


# ---------- Fallback: FAQ keyword matching (works with no API key) ----------
STOPWORDS = {"the", "a", "an", "is", "are", "do", "does", "you", "your", "i", "my",
             "can", "how", "what", "to", "of", "for", "in", "on", "and", "or", "it",
             "me", "we", "have", "get", "there", "with", "at", "be"}


SYNONYMS = {"timing": "hour", "time": "hour", "open": "hour", "opening": "hour",
            "cost": "fee", "price": "fee", "charge": "fee", "rate": "fee",
            "address": "locat", "where": "locat", "location": "locat", "located": "locat"}


def _normalize(word: str) -> str:
    if word in SYNONYMS:
        return SYNONYMS[word]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]  # crude plural removal: timings -> timing
    return SYNONYMS.get(word, word)


def _tokens(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_normalize(w) for w in words if w not in STOPWORDS}


def ask_faq_match(customer_text: str) -> str:
    """Return the best FAQ answer, or HANDOFF if nothing matches well."""
    words = _tokens(customer_text)
    if not words:
        return "HANDOFF"
    best_answer, best_score = None, 0.0
    for item in FAQ:
        overlap = len(words & _tokens(item["q"] + " " + item["a"]))
        score = overlap / len(words)
        if score > best_score:
            best_answer, best_score = item["a"], score
    return best_answer if best_score >= 0.5 else "HANDOFF"


# ---------- LLM ----------
LAST_LLM_ERROR = ""


def ask_llm(customer_text: str):
    """Use the LLM if a key is set; otherwise (or on failure) use FAQ matching.
    Returns (answer, engine) so the website can show which one answered."""
    global LAST_LLM_ERROR
    if not LLM_API_KEY:
        LAST_LLM_ERROR = "LLM_API_KEY is empty, so FAQ matching is used"
        return ask_faq_match(customer_text), "FAQ matching (no key)"
    try:
        answer = _ask_llm_api(customer_text)
        if not answer:
            raise ValueError("empty reply from the LLM")
        LAST_LLM_ERROR = ""
        return answer, "Gemini/LLM"
    except Exception as e:
        LAST_LLM_ERROR = f"{type(e).__name__}: {e}"[:400]
        print("LLM ERROR:", LAST_LLM_ERROR)
        return ask_faq_match(customer_text), "FAQ matching (LLM failed)"


def _call_model(model: str, customer_text: str) -> str:
    r = httpx.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={
            "model": model,
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": customer_text},
            ],
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"{model} -> HTTP {r.status_code}: {r.text[:250]}")
    return r.json()["choices"][0]["message"]["content"].strip()


def _ask_llm_api(customer_text: str) -> str:
    """Try each model in LLM_MODEL (comma-separated). Retry once on busy servers."""
    models = [m.strip() for m in LLM_MODEL.split(",") if m.strip()]
    last_error = None
    for model in models:
        for attempt in range(2):
            try:
                return _call_model(model, customer_text)
            except Exception as e:
                last_error = e
                busy = any(code in str(e) for code in ("HTTP 429", "HTTP 500", "HTTP 503"))
                if busy and attempt == 0:
                    time.sleep(1.5)  # short wait, then retry the same model
                    continue
                break  # try the next model
    raise last_error


# ---------- Core logic ----------
def handle_message(phone: str, text: str, send: bool = True) -> dict:
    start = time.time()
    answer, engine = ask_llm(text)
    handoff = answer.upper().startswith("HANDOFF")
    reply = HANDOFF_REPLY if handoff else answer
    sms_status = send_sms(phone, reply) if send else "NOT SENT (test mode)"
    latency_ms = int((time.time() - start) * 1000)
    log_conversation(phone, text, reply, handoff, latency_ms, sms_status)
    return {"reply": reply, "handoff": handoff, "engine": engine,
            "latency_ms": latency_ms, "sms_status": sms_status}


# ---------- Endpoints ----------
@app.get("/")
def home():
    """The chat website."""
    return FileResponse("index.html")


@app.get("/admin")
def admin():
    """Admin page: conversations, handoffs, response time."""
    return FileResponse("admin.html")


@app.get("/health")
def health():
    return {"status": "running", "llm_key_set": bool(LLM_API_KEY),
            "llm_model": LLM_MODEL, "llm_base_url": LLM_BASE_URL,
            "last_llm_error": LAST_LLM_ERROR}


class ChatMessage(BaseModel):
    text: str


@app.post("/chat")
def chat(m: ChatMessage):
    """Used by the website chat box."""
    return handle_message("web-user", m.text[:500], send=False)


@app.post("/webhook/infobip")
async def infobip_webhook(request: Request):
    """Infobip calls this URL when a customer sends an inbound SMS."""
    body = await request.json()
    results = []
    for msg in body.get("results", []):
        phone = msg.get("from", "")
        text = msg.get("cleanText") or msg.get("text", "")
        if phone and text:
            results.append(handle_message(phone, text))
    return {"processed": len(results)}


class TestMessage(BaseModel):
    phone: str
    text: str
    send: bool = False  # set true to really send SMS through Infobip


@app.post("/simulate")
def simulate(m: TestMessage):
    """Test without an inbound number: pretend a customer sent a message."""
    return handle_message(m.phone, m.text, send=m.send)


@app.get("/logs")
def logs(limit: int = 50):
    rows = db.execute(
        "SELECT ts, phone, message, reply, handoff, latency_ms, sms_status "
        "FROM logs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    keys = ["time", "phone", "message", "reply", "handoff", "latency_ms", "sms_status"]
    total = db.execute("SELECT COUNT(*), SUM(handoff), AVG(latency_ms) FROM logs").fetchone()
    return {
        "stats": {"total": total[0], "handoffs": total[1] or 0,
                  "avg_latency_ms": int(total[2] or 0)},
        "conversations": [dict(zip(keys, r)) for r in rows],
    }
