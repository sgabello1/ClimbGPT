### ClimbGPT 2.0 — Telegram Bot Backend (FastAPI)
# Starter Code: Connects Telegram to OpenRouter LLM + RAG

from fastapi import FastAPI, Request
from pydantic import BaseModel
import httpx
import os
import uvicorn
from typing import Dict, Any
import json

# Load climbing knowledge base
with open("climb_knowledge.json", "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = json.load(f)["data"]


from dotenv import load_dotenv
load_dotenv()

# ENV VARS
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BOT_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


app = FastAPI()

class TelegramMessage(BaseModel):
    message: Dict[str, Any]

# Util: Call OpenRouter LLM (Claude or Mixtral)
async def ask_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://yourdomain.com",  # Can be anything
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mixtral-8x7b-instruct",
        "messages": [
            {"role": "system", "content": "You are ClimbGPT, a helpful climbing assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()  # Raise error if HTTP error
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ OpenRouter call failed:", e)
        return "Sorry, I ran into an error trying to answer. Try again later!"


# Simple info retrieval (simulate RAG)
def find_relevant_info(user_text: str) -> str:
    relevant = []
    lowered = user_text.lower()
    for entry in KNOWLEDGE_BASE:
        # Match against topic or tags
        if any(word in lowered for word in entry["topic"].lower().split()) or \
           any(tag in lowered for tag in entry["tags"]):
            relevant.append(entry["content"])
    return "\n\n".join(relevant) if relevant else "Sorry, I don't have specific info on that yet!"


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
    uvicorn.run("ClimbGPT:app", host="0.0.0.0", port=8000, reload=True)
