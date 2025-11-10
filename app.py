# app.py
import os, time, json, requests, httpx
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------- Datenmodelle ----------
class Persona(BaseModel):
    name: str
    description: str
    industry: Optional[str] = None
    stage: Optional[str] = "Awareness"
    language: Optional[str] = "de"

class IdeaRequest(BaseModel):
    persona: Persona
    topic: str
    subreddits: Optional[List[str]] = None
    time_range: Optional[str] = "month"
    limit: Optional[int] = 10

# ---------- Reddit Suche ----------
def reddit_search(query: str, subreddits: Optional[List[str]] = None, time_range="month", limit=10):
    base = "https://www.reddit.com/search.json"
    headers = {"User-Agent": "PersonaResearcher/1.0"}
    q = query
    if subreddits:
        subs = " OR ".join([f"subreddit:{s}" for s in subreddits])
        q = f"({query}) AND ({subs})"
    params = {"q": q, "t": time_range, "limit": limit, "sort": "relevance"}
    r = requests.get(base, params=params, headers=headers, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Reddit search failed: {r.status_code}")
    data = r.json()
    results = []
    for child in data.get("data", {}).get("children", []):
        d = child["data"]
        results.append({
            "title": d.get("title"),
            "url": "https://www.reddit.com" + d.get("permalink", ""),
            "subreddit": d.get("subreddit"),
            "score": d.get("score", 0),
            "num_comments": d.get("num_comments", 0),
            "selftext": d.get("selftext", "")
        })
    return results

# ---------- OpenAI Aufruf ----------
def openai_chat(messages):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY fehlt.")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {"model": "gpt-4.1-mini", "messages": messages}
    resp = httpx.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=120.0)
    if resp.status_code >= 300:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()

# ---------- Synthese ----------
def synthesize_ideas(persona: Persona, topic: str, findings: List[Dict]):
    system = {
        "role": "system",
        "content": (
            "Du bist ein Research-Agent. Fasse Reddit-Ergebnisse zu Content-Ideen zusammen. "
            "Jede Idee: Hook, Format, Warum relevant für die Persona, Quellen. "
            "Schreibe in Deutsch. Keine Idee ohne Quelle!"
        ),
    }
    user = {
        "role": "user",
        "content": json.dumps({
            "persona": persona.dict(),
            "topic": topic,
            "findings": findings[:10]
        }, ensure_ascii=False)
    }
    res = openai_chat([system, user])
    return res["choices"][0]["message"]["content"]

# ---------- API ----------
app = FastAPI(title="Persona Research Agent")

@app.get("/")
def root():
    return {"status": "ok", "message": "Persona Research Agent läuft 🚀"}

@app.post("/ideas")
def ideas(req: IdeaRequest):
    results = reddit_search(req.topic, req.subreddits, req.time_range, req.limit)
    ideas = synthesize_ideas(req.persona, req.topic, results)
    return {"ideas": ideas, "sources": [r["url"] for r in results]}
