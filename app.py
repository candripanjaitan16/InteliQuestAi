import json
import numpy as np
import random
import os
import sqlite3
import re
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
    hidden_layer_sizes=(64, 32, 16),
    max_iter=5000,
    activation='relu',
    solver='adam',
    random_state=42,
    learning_rate='adaptive',
    alpha=0.0001
)
model.fit(X_vektor, y)


class ChatInput(BaseModel):
    pesan: str
    session_id: int | None = None


class NewSession(BaseModel):
    title: str = "Chat baru"


conversation_context = {}

def get_context(session_id):
    if session_id not in conversation_context:
        conversation_context[session_id] = []
    return conversation_context[session_id]

def add_to_context(session_id, role, content):
    if session_id not in conversation_context:
        conversation_context[session_id] = []
    conversation_context[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    if len(conversation_context[session_id]) > 10:
        conversation_context[session_id] = conversation_context[session_id][-10:]

def detect_sentiment(pesan):
    positive_words = ['senang', 'bahagia', 'happy', 'mantap', 'keren', 'hebat', 'good', 'great', 'awesome', 'cinta', 'suka', 'love', 'yes', 'hore', 'yeay']
    negative_words = ['sedih', 'galau', 'marah', 'kesal', 'benci', 'bad', 'sad', 'angry', 'hate', 'kecewa', 'stres', 'depresi', 'takut', 'cemas', 'khawatir']
    
    pesan_lower = pesan.lower()
    pos_count = sum(1 for word in positive_words if word in pesan_lower)
    neg_count = sum(1 for word in negative_words if word in pesan_lower)
    
    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    else:
        return 'neutral'

def generate_fallback_response(pesan, sentiment, context):
    fallback_responses = {
        'positive': [
            "Senang sekali mendengar semangatmu! 😊 Ada yang bisa saya bantu untuk mempertahankan mood positif ini?",
            "Wah, energimu terasa sampai ke sini! ✨ Mau cerita lebih lanjut atau ada yang ingin kamu tanyakan?",
            "Keren! Dengan semangat seperti ini, pasti banyak hal hebat yang bisa kamu capai. Ada yang ingin didiskusikan?"
        ],
        'negative': [
            "Saya mengerti kadang ada hal yang sulit. 🤗 Ingin cerita lebih lanjut? Saya di sini untuk mendengarkan.",
            "Tidak apa-apa merasa seperti ini. 💙 Kalau mau curhat atau butuh saran, saya siap membantu.",
            "Tenang, saya di sini untukmu. 🌟 Kadang berbicara tentang masalah bisa membuatnya terasa lebih ringan. Mau coba cerita?"
        ],
        'neutral': [
            "Hmm, menarik! Bisa kamu jelaskan lebih detail apa yang kamu maksud? 🤔",
            "Saya ingin memahami maksudmu dengan lebih baik. Ada cara lain untuk menjelaskannya? 💭",
            "Pertanyaan yang bagus! 🌟 Agar saya bisa memberikan jawaban yang tepat, bisakah kamu memberikan lebih banyak konteks?"
        ]
    }
    
    responses = fallback_responses.get(sentiment, fallback_responses['neutral'])
    return random.choice(responses)

def extract_keywords(pesan):
    stop_words = {'yang', 'dan', 'atau', 'tapi', 'dengan', 'untuk', 'dari', 'di', 'ke', 'dari', 'pada', 'dalam', 'ini', 'itu', 'apa', 'siapa', 'kapan', 'dimana', 'mengapa', 'bagaimana', 'aku', 'saya', 'kamu', 'anda', 'ga', 'nggak', 'tidak', 'kok', 'sih', 'lah', 'deh', 'ya', 'dong', 'nih', 'tu', 'kan'}
    words = re.findall(r'\b\w+\b', pesan.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    return keywords[:5]

def handle_general_question(pesan, keywords):
    general_knowledge = {
        'cuaca': ["Maaf, saya belum bisa mengakses informasi cuaca secara real-time. Tapi kamu bisa cek di aplikasi cuaca di ponselmu! ☀️🌧️",
                  "Untuk info cuaca terkini, coba buka website BMKG atau aplikasi cuaca favoritmu ya! 🌤️"],
        'waktu': [f"Sekarang adalah {datetime.now().strftime('%H:%M:%WIB')}. Semoga harimu menyenangkan! ⏰",
                  f"Waktu menunjukkan {datetime.now().strftime('%H:%M')}. Sudah saatnya untuk istirahat kalau lelah ya! 🕐"],
        'tanggal': [f"Hari ini adalah {datetime.now().strftime('%A, %d %B %Y')}. Ada rencana spesial hari ini? 📅",
                    f"Tanggal {datetime.now().strftime('%d/%m/%Y')}. Semoga hari ini membawa keberuntungan untukmu! 🗓️"],
        'siapa presiden': ["Presiden Indonesia saat ini adalah Prabowo Subianto. 🇮🇩",
                          "Indonesia dipimpin oleh Presiden Prabowo Subianto sejak 2024. 🎯"],
        'indonesia': ["Indonesia adalah negara kepulauan terbesar di dunia dengan lebih dari 17.000 pulau! 🏝️",
                      "Indonesia terkenal dengan keberagaman budaya, bahasa, dan alamnya yang indah. 🌺",
                      "Tahukah kamu? Indonesia memiliki hutan hujan tropis terbesar ketiga di dunia! 🌳"],
        'tips': ["Tips produktivitas: Pecah tugas besar menjadi langkah-langkah kecil yang lebih mudah dikelola! 📝",
                 "Coba teknik Pomodoro: 25 menit fokus, 5 menit istirahat. Ulangi 4 kali lalu istirahat lebih lama! ⏱️",
                 "Jangan lupa minum air putih cukup! Dehidrasi bisa bikin konsentrasi menurun. 💧"],
        'motivasi': ["Ingat: Setiap langkah kecil membawa kamu lebih dekat ke tujuanmu! 🚀",
                     "Kamu lebih kuat dari yang kamu kira. Terus berjuang! 💪",
                     "Hari ini adalah kesempatan baru untuk menjadi versi terbaik dari dirimu sendiri! ✨"],
        'belajar': ["Tips belajar efektif: Ajarkan kembali apa yang sudah kamu pelajari. Ini membantu memperkuat pemahaman! 📚",
                    "Istirahat itu penting! Otak butuh waktu untuk memproses informasi. Jangan belajar terlalu lama tanpa jeda. 🧠",
                    "Buat ringkasan atau mind map untuk memudahkanmu mengingat konsep-konsep penting! 🗺️"]
    }
    
    for key, responses in general_knowledge.items():
        if key in pesan.lower():
            return random.choice(responses)
    
    return None

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
    conversation_context[session_id] = []
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
    if session_id in conversation_context:
        del conversation_context[session_id]
    return {"ok": True}


@app.post("/chat")
def chat(input_data: ChatInput):
    pesan = input_data.pesan
    pesan_lower = pesan.lower()
    session_id = input_data.session_id
    now = datetime.now().isoformat()

    context = get_context(session_id) if session_id else []
    sentiment = detect_sentiment(pesan)
    keywords = extract_keywords(pesan)
    
    general_response = handle_general_question(pesan, keywords)
    if general_response:
        reply = general_response
        confidence = 0.8
    else:
        input_vektor = vectorizer.transform([pesan_lower])
        probabilities = model.predict_proba(input_vektor)[0]
        max_prob = np.max(probabilities)
        confidence = max_prob
        
        if max_prob < 0.35:
            reply = generate_fallback_response(pesan, sentiment, context)
        else:
            tag_prediksi = model.predict(input_vektor)[0]
            reply = "Maaf, tidak ada jawaban yang cocok."
            for intent in data_json['intents']:
                if intent['tag'] == tag_prediksi:
                    responses = intent['jawaban']
                    
                    if context and len(context) >= 2:
                        last_user_msg = None
                        for msg in reversed(context):
                            if msg['role'] == 'user':
                                last_user_msg = msg['content']
                                break
                        
                        if last_user_msg:
                            last_keywords = extract_keywords(last_user_msg)
                            common_keywords = set(keywords) & set(last_keywords)
                            if common_keywords and len(responses) > 1:
                                reply = random.choice(responses)
                                reply += f" Ngomong-ngomong, tentang {', '.join(common_keywords)} tadi, ada yang ingin ditanyakan lagi?"
                                break
                    
                    reply = random.choice(responses)
                    break
    
    if session_id:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
        msg_count = c.fetchone()[0]
        if msg_count == 0:
            title = pesan[:40] + ("..." if len(pesan) > 40 else "")
            c.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
        c.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "user", pesan, now)
        )
        c.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, "ai", reply, now)
        )
        conn.commit()
        conn.close()
        
        add_to_context(session_id, "user", pesan)
        add_to_context(session_id, "ai", reply)

    return {"reply": reply, "sentiment": sentiment, "confidence": float(confidence)}