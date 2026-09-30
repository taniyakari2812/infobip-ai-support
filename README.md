# AI Customer Support Assistant (Infobip + an LLM)

A prototype that answers customer questions over SMS using Infobip's messaging API and an LLM.
Built to learn how Infobip's API, authentication and webhooks work from a developer's side,
and to see what a business would face when adopting them.

## Problem
Small businesses (for example a clinic) get the same questions again and again by SMS:
timings, fees, location, booking. Staff waste time replying manually, and customers wait.

## Solution
1. Customer sends an SMS.
2. Infobip forwards it to my FastAPI webhook (`/webhook/infobip`).
3. an LLM answers using only the business FAQ (`faq.json`), so it does not make things up.
4. If the answer is not in the FAQ, or the customer is angry, the bot hands off to a human.
5. The reply is sent back through the Infobip SMS API.
6. Every conversation is logged in SQLite and shown at `/logs` (handoff count, average latency).

## Architecture
Customer -> Infobip -> FastAPI webhook -> an LLM (FAQ in prompt) -> Infobip SMS API -> Customer
                                   |
                                   +-> SQLite log -> /logs

## Tech stack
Python, FastAPI, Gemini API, Infobip SMS API, SQLite, httpx

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env      # then add your keys
uvicorn main:app --reload
```

## Website
Run `uvicorn main:app --reload`, then open:
- http://127.0.0.1:8000/  (customer chat page)
- http://127.0.0.1:8000/admin  (admin dashboard: conversations, handoff rate, response time)

## Test without an inbound number
```bash
curl -X POST http://127.0.0.1:8000/simulate -H "Content-Type: application/json" \
  -d '{"phone":"91XXXXXXXXXX","text":"What are your timings?","send":false}'
```
Set `"send": true` to actually send the reply SMS through Infobip (trial accounts can only
send to the phone number you registered with).

## Results (fill in after testing)
- Tested with __ sample questions, answered correctly: __
- Handoff triggered correctly for out-of-scope questions: __
- Average response time: __ ms

## Limitations and next steps
- FAQ is placed in the prompt; for large knowledge bases use a vector database (RAG).
- No conversation memory yet.
- Add WhatsApp channel, webhook signature verification, and a proper dashboard.
