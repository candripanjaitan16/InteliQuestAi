import json
import numpy as np
import random
import os
import sqlite3
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

app = FastAPI()

app.mount("/templates/assets", StaticFiles(directory="templates/assets"), name="assets")

DB_PATH = "inteliquest.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

with open('otak.json', 'r') as f:
    data_json = json.load(f)

X = []
y = []

for intent in data_json['intents']:
    for pola in intent['pola']:
        X.append(pola.lower())
        y.append(intent['tag'])

vectorizer = CountVectorizer()
X_vektor = vectorizer.fit_transform(X)

model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    max_iter=5000,
    activation='relu',
    solver='adam',
    random_state=42
)
model.fit(X_vektor, y)


class ChatInput(BaseModel):
    pesan: str
    session_id: int = None


class NewSession(BaseModel):
    title: str = "Chat baru"


@app.get("/", response_class=HTMLResponse)
def index():
    path = os.path.join("templates", "index.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/session/new")
def new_session(data: NewSession):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute("INSERT INTO sessions (title, created_at) VALUES (?, ?)", (data.title, now))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return {"session_id": session_id, "title": data.title}


@app.get("/sessions")
def get_sessions():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM sessions ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return {"sessions": [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]}


@app.get("/session/{session_id}/messages")
def get_messages(session_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,)
    )
    rows = c.fetchall()
    conn.close()
    return {"messages": [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]}


@app.delete("/session/{session_id}")
def delete_session(session_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
    return {"ok": True}


@app.post("/chat")
def chat(input_data: ChatInput):
    pesan = input_data.pesan.lower()
    session_id = input_data.session_id
    now = datetime.now().isoformat()

    input_vektor = vectorizer.transform([pesan])
    probabilities = model.predict_proba(input_vektor)[0]
    max_prob = np.max(probabilities)

    if max_prob < 0.45:
        reply = "Maaf, saya memerlukan lebih banyak informasi untuk memahami maksud Anda."
    else:
        tag_prediksi = model.predict(input_vektor)[0]
        reply = "Maaf, tidak ada jawaban yang cocok."
        for intent in data_json['intents']:
            if intent['tag'] == tag_prediksi:
                reply = random.choice(intent['jawaban'])
                break

    if session_id:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        msg_count = c.fetchone()[0]
        if msg_count == 0:
            title = input_data.pesan[:40] + ("..." if len(input_data.pesan) > 40 else "")
            c.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
        c.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "user", input_data.pesan, now)
        )
        c.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "ai", reply, now)
        )
        conn.commit()
        conn.close()

    return {"reply": reply}