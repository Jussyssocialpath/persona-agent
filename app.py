import os, json
from typing import List
from flask import Flask, request, jsonify
import praw
from openai import OpenAI

oai = OpenAI()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "PersonaResearcher/1.0")

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_SECRET,
    user_agent=REDDIT_USER_AGENT,
)

app = Flask(__name__)

@app.get("/")
def health():
    return {"status": "ok", "service": "persona-agent"}

def _normalize_body(data: dict):
    persona = data.get("persona")
    if isinstance(persona, str):
        persona_obj = {"name": persona, "description": persona}
    else:
        persona_obj = persona or {}
    topic = data.get("topic") or data.get("query") or ""
    subs = data.get("subreddits")
    if not subs:
        sub = data.get("subreddit")
        subs = [sub] if isinstance(sub, str) and sub else []
    timeframe = data.get("time_range") or data.get("timeframe") or "month"
    limit = int(data.get("limit", 10))
    return persona_obj, topic, subs, timeframe, limit

def _fetch_reddit_posts(topic: str, subs: List[str], timeframe: str, limit: int):
    posts = []
    try:
        target = "+".join(subs) if subs else "all"
        sr = reddit.subreddit(target)
        if topic:
            results = sr.search(query=topic, time_filter=timeframe, sort="relevance", limit=limit)
        else:
            results = sr.top(timeframe, limit=limit)
        for s in results:
            posts.append({
                "title": s.title,
                "url": f"https://www.reddit.com{s.permalink}",
                "subreddit": str(s.subreddit),
                "score": int(getattr(s, "score", 0)),
                "num_comments": int(getattr(s, "num_comments", 0)),
            })
    except Exception as e:
        return [], str(e)
    return posts, None

@app.post("/ideas")
def ideas():
    data = request.get_json(force=True, silent=True) or {}
    persona, topic, subs, timeframe, limit = _normalize_body(data)
    posts, err = _fetch_reddit_posts(topic, subs, timeframe, limit)
    if err is not None:
        return jsonify({"error": f"Reddit API error: {err}"}), 403
    sys = (
        "Du bist ein Research- und Content-Strategie-Agent. "
        "Erzeuge präzise, umsetzbare Content-Ideen (Titel, Hook, Angle, Format) "
        "für Social Media und Blog, abgestimmt auf die Persona. "
        "Beziehe dich auf die Reddit-Insights. Antworte auf Deutsch."
    )
    usr = (
        f"Persona: {json.dumps(persona, ensure_ascii=False)}\n"
        f"Thema: {topic}\n"
        f"Reddit-Quellen (max {len(posts)}): {json.dumps(posts, ensure_ascii=False)}\n"
        "Erzeuge 5 Ideen. Zeige am Ende eine Liste der verwendeten Quellen-URLs."
    )
    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": usr}],
        temperature=0.7,
    )
    text = resp.choices[0].message.content
    return jsonify({
        "ideas": text,
        "sources": [p["url"] for p in posts]
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
