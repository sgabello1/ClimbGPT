### ClimbGPT 2.0 — Telegram Bot Backend (FastAPI)
# Starter Code: Connects Telegram to OpenRouter LLM + RAG

from fastapi import FastAPI, Request
from pydantic import BaseModel
import httpx
import os
import uvicorn
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

# ENV VARS
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BOT_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Dummy knowledge base (replace with real RAG later)
CLIMB_KNOWLEDGE = [
    {"topic": "Gyms in Munich", "content": "Boulderwelt Ost: 5A–7A, quiet mornings. Heavens Gate: Great rope climbing, less crowded midday."},
    {"topic": "Shoe resoling Germany", "content": "Sohlenprofi and Kletterschuhexpress offer resoling, avg price ~30€."},
    {"topic": "Arco sport climbing", "content": "Arco has sectors like Massi di Prabi and Nago. Best in spring and autumn for 5c–7a routes."}
]

app = FastAPI()

class TelegramMessage(BaseModel):
    message: Dict[str, Any]

# Util: Call OpenRouter LLM (Claude or Mixtral)
async def ask_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://climbgpt-bot.onrender.com/webhook",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mixtral-8x7b",
        "messages": [
            {"role": "system", "content": "You are ClimbGPT, a helpful climbing assistant for gyms, crags, and gear in Europe."},
            {"role": "user", "content": prompt}
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        return response.json()['choices'][0]['message']['content']

# Simple info retrieval (simulate RAG)
def find_relevant_info(user_text: str) -> str:
    relevant = []
    for item in CLIMB_KNOWLEDGE:
        if any(word.lower() in user_text.lower() for word in item['topic'].split()):
            relevant.append(item['content'])
    return "\n".join(relevant) if relevant else ""

@app.post("/webhook")
async def telegram_webhook(update: TelegramMessage):
    chat_id = update.message["chat"]["id"]
    user_text = update.message.get("text", "")

    base_knowledge = find_relevant_info(user_text)
    prompt = f"User asked: {user_text}\nRelevant info:\n{base_knowledge}\nRespond helpfully."

    llm_reply = await ask_llm(prompt)

    send_payload = {
        "chat_id": chat_id,
        "text": llm_reply
    }
    async with httpx.AsyncClient() as client:
        await client.post(f"{BOT_URL}/sendMessage", json=send_payload)
    return {"ok": True}

@app.get("/")
def health():
    return {"status": "ClimbGPT Telegram bot is running!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
