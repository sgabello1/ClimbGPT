### ClimbGPT 2.0 — Telegram Bot Backend (FastAPI + Crag Weather)

from fastapi import FastAPI, Request
from pydantic import BaseModel
import httpx
import os
import uvicorn
from typing import Dict, Any
import json

from dotenv import load_dotenv
load_dotenv()

# ENV VARS
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BOT_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load climbing knowledge base
with open("climb_knowledge.json", "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = json.load(f)["data"]

# Load crag index with topo + weather
with open("crag_index.json", "r", encoding="utf-8") as f:
    CRAG_INDEX = json.load(f)

app = FastAPI()

class TelegramMessage(BaseModel):
    message: Dict[str, Any]

# --- OpenRouter LLM Integration ---
async def ask_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://yourdomain.com",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/mixtral-8x7b-instruct",
        "messages": [
            {"role": "system", "content": "You are ClimbGPT, a helpful climbing assistant for gyms, crags, and gear in Europe."},
            {"role": "user", "content": prompt}
        ]
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("❌ OpenRouter call failed:", e)
        return "Sorry, I had trouble processing that request."

# --- Knowledge Matching ---
def find_relevant_info(user_text: str) -> str:
    relevant = []
    lowered = user_text.lower()
    for entry in KNOWLEDGE_BASE:
        if any(word in lowered for word in entry["topic"].lower().split()) or \
           any(tag in lowered for tag in entry["tags"]):
            relevant.append(entry["content"])
    return "\n\n".join(relevant) if relevant else ""

def get_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        r = requests.get(url)
        r.raise_for_status()  # raises if HTTP 4xx or 5xx
        print("🌍 Weather API raw response:", r.json())  # DEBUG
        data = r.json()["current_weather"]
        return {
            "temp": data["temperature"],
            "windspeed": data["windspeed"]
        }
    except Exception as e:
        print("❌ Weather fetch failed:", e)
        return None


def get_crag_info_static(name: str) -> str:
    key = name.lower().strip().replace(" ", "")
    if key not in CRAG_INDEX:
        return f"❌ Crag '{name}' not found."

    crag = CRAG_INDEX[key]
    weather = get_weather(crag["lat"], crag["lon"])

    reply = (
        f"🧗‍♂️ *{crag['name']}*\n"
        f"{crag['style']}, Grades: {crag['grades']}\n\n"
    )

    if weather:
        reply += f"🌤️ *Weather now*: {weather['temp']}°C, wind {weather['windspeed']} km/h\n\n"
    else:
        reply += "🌤️ Weather unavailable\n\n"

    reply += (
        f"🔗 [Crag on TheCrag.com]({crag['url']})\n"
        f"📍 [Topo link]({crag['topo']})"
    )
    return reply

# --- Telegram Webhook Handler ---
@app.post("/webhook")
async def telegram_webhook(update: TelegramMessage):
    chat_id = update.message["chat"]["id"]
    user_text = update.message.get("text", "")
    lowered = user_text.lower()

    # Handle known crag requests
    for crag in CRAG_INDEX:
        if crag in lowered:
            crag_reply = get_crag_info_static(crag)
            await send_message(chat_id, crag_reply)
            return {"ok": True}

    # Use knowledge base + LLM
    base_knowledge = find_relevant_info(user_text)
    prompt = f"User asked: {user_text}\nRelevant info:\n{base_knowledge}\nRespond helpfully."
    llm_reply = await ask_llm(prompt)

    await send_message(chat_id, llm_reply)
    return {"ok": True}

# --- Send Telegram Message ---
async def send_message(chat_id: int, text: str):
    send_payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    async with httpx.AsyncClient() as client:
        await client.post(f"{BOT_URL}/sendMessage", json=send_payload)

# --- Health Check ---
@app.get("/")
def health():
    return {"status": "ClimbGPT Telegram bot is running!"}

if __name__ == "__main__":
    uvicorn.run("ClimbGPT:app", host="0.0.0.0", port=8000, reload=True)