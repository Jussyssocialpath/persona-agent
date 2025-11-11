import os, json
from typing import List, Optional
from flask import Flask, request, jsonify
import praw

# ---------- OpenAI (neues SDK) ----------
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oai = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Reddit (OAuth, Script App) ----------
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "PersonaResearcher/1.0")

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_SECRET,
    user_agent=REDDIT_USER_AGENT,
)

app = Flask(__name__)

def normalize_body(data: dict):
    """Akzeptiert beide Varianten aus deinem openapi.yaml und dem Test-UI."""
    # persona kann Objekt oder String sein
    persona = data.get("persona")
    if isinstance(persona, str):
        persona_obj = {"name": persona, "description": persona}
    else:
        persona_obj = persona or {}

    topic = data.get("topic") or data.get("query") or ""
    # subreddits: Liste oder einzelner String
    subs = data.get("subreddits")
    if not subs:
        one = data.get("subreddit")
        subs = [one] if isinstance(one, str) and one else []
    # Zeitraum
    timeframe = data.get("time_range") or data.get("timeframe") or "month"
    limit = int(data.get("limit", 10))
    return persona_obj, topic, subs, timeframe, limit

def fetch_reddit_posts(topic: str, subs: List[str], timeframe: str, limit: int):
    """Sucht relevante Posts. Bei mehreren Subs: Multi-Subreddit via '+'. """
    posts = []
    try:
        target = "+".join(subs) if subs else "all"
        sr = reddit.subreddit(target)
        # Suche nach Topic, fallback auf top
        results = sr.search(query=topic, time_filter=timeframe, sort="relevance", limit=limit) if topic else sr.top(timeframe, limit=limit)
        for s in results:
            posts.append({
                "title": s.title,
                "url": f"https://www.reddit.com{s.permalink}",
                "subreddit": str(s.subreddit),
                "score": int(getattr(s, "score", 0)),
                "num_comments": int(getattr(s, "num_comments", 0)),
            })
    except Exception as e:
        # Klarer Fehler statt 500-Blackbox
        return [], str(e)
    return posts, None

@app.route("/ideas", methods=["POST"])
def ideas():
    data = request.get_json(force=True, silent=True) or {}
    persona, topic, subs, timeframe, limit = normalize_body(data)

    posts, err = fetch_reddit_posts(topic, subs, timeframe, limit)
    if err is not None:
        return jsonify({"error": f"Reddit API error: {err}"}), 403

    # Prompt für Ideen (deutsch), inkl. Quellenbezug
    sys = ("Du bist ein Research- und Content-Strategie-Agent. "
           "Erzeuge präzise, umsetzbare Content-Ideen für Social Media & Blog, "
           "abgestimmt auf die Persona. Gib kurze Titel + Hook + Angle + Format. "
           "Beziehe dich auf die gefundenen Reddit-Insights. Antworte auf Deutsch.")
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

# Render/Gunicorn: kein app.run hier nötig.
# Für lokales Debuggen:
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
