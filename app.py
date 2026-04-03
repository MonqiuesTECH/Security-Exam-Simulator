import streamlit as st
import os
import time
import random
import urllib.parse
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. PAGE CONFIGURATION
load_dotenv()
st.set_page_config(page_title="Security+ SY0-701 Simulator", page_icon="🛡️", layout="wide")

# ==========================================
# DATABASE LOGIC
# ==========================================
DB_FILE = "study_logs.json"

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_db(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def ensure_user_exists(user):
    db = load_db()
    if user not in db:
        db[user] = {"time_spent_sec": 0, "logs": [], "weak_topics": [], "current_score": "0 / 0"}
        save_db(db)
    return db

def log_event(user, event_type, notes, topic=None):
    db = ensure_user_exists(user)
    timestamp = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    db[user]["logs"].insert(0, {"timestamp": timestamp, "event": event_type, "notes": notes})
    if topic and topic not in db[user]["weak_topics"]:
        db[user]["weak_topics"].append(topic)
    save_db(db)

def ping_time_tracker(user):
    if "last_ping" in st.session_state:
        elapsed = time.time() - st.session_state.last_ping
        if elapsed < 3600: 
            db = ensure_user_exists(user)
            db[user]["time_spent_sec"] += elapsed
            save_db(db)
    st.session_state.last_ping = time.time()

def update_live_score(user, correct, total):
    db = load_db()
    if user in db:
        db[user]["current_score"] = f"{correct} / {total}"
        save_db(db)

# ==========================================
# LOGIN SYSTEM
# ==========================================
def check_password():
    def password_entered():
        user = st.session_state["username"]
        if user in st.secrets["passwords"] and st.session_state["password"] == st.secrets["passwords"][user]:
            st.session_state["password_correct"] = True
            st.session_state["current_user"] = user
            del st.session_state["password"]  
            if user != "admin":
                st.session_state.last_ping = time.time()
                log_event(user, "Logged In", "Session started.")
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 Security+ Simulator Login")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 Security+ Simulator Login")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.error("Invalid credentials.")
        return False
    return True

# ==========================================
# RESOURCE LOADING & AI LOGIC
# ==========================================
@st.cache_resource
def load_resources():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        return vectorstore, llm
    except: return None, None

def get_adaptive_question(llm, raw_text, diff):
    t = PromptTemplate.from_template("Extract ONE Security+ question. Level: {d}\nText: {r}\nFormat: QUESTION, A, B, C, D, CORRECT.")
    return (t | llm | StrOutputParser()).invoke({"r": raw_text[:1200], "d": diff})

def get_tutor_feedback(llm, q, ua, cl, ir, diff):
    t = PromptTemplate.from_template("You are a Security+ tutor. Question: {q}, User: {ua}, Correct: {cl}, Result: {ir}, Level: {d}. Explain simply.")
    return (t | llm | StrOutputParser()).invoke({"q": q, "ua": ua, "cl": cl, "ir": ir, "d": diff})

# ==========================================
# PBQ DATABASE
# ==========================================
PBQ_DB = {
    1: {"topic": "Network Ports", "title": "Port Matching", "desc": "Assign ports.", "type": "match", "keys": ["SSH", "HTTPS", "RDP", "DNS"], "options": ["22", "53", "443", "3389"]},
    7: {"topic": "Firewall Rules", "title": "Firewall Config", "desc": "Block HTTP from 192.168.1.50.", "type": "firewall"}
    # ... (Other PBQs follow same structure)
}

# ==========================================
# ADMIN DASHBOARD (UPDATED PERSISTENT LAYOUT)
# ==========================================
def run_admin_dashboard():
    st.title("👨‍🏫 Instructor Dashboard")
    
    # 1. Define authorized guest list
    authorized_guests = ["guest1", "guest2"]
    st.subheader(f"Active Profiles Monitored: {len(authorized_guests)}")
    
    db = load_db()
    tab1, tab2 = st.tabs(["📊 Performance Overview", "📋 Session Logs"])
    
    with tab1:
        cols = st.columns(len(authorized_guests))
        for idx, student in enumerate(authorized_guests):
            # Ensure student has a data shell even if they haven't logged in
            data = db.get(student, {"time_spent_sec": 0, "current_score": "0 / 0", "weak_topics": [], "logs": []})
            total_hrs = data["time_spent_sec"] / 3600
            
            with cols[idx]:
                st.markdown(f"### 🧑‍🎓 `{student}`")
                st.metric("Total Study Time", f"{total_hrs:.2f} Hours")
                st.metric("Live Quiz Score", data["current_score"])
                
                st.write("**Topics Needing Work:**")
                if data["weak_topics"]:
                    for topic in data["weak_topics"]: st.error(f"⚠️ {topic}")
                else: st.success("No weak topics logged.")
                
                st.markdown("---")
                if st.button(f"🗑️ Wipe {student}'s Data", key=f"del_{student}"):
                    if student in db: del db[student]
                    save_db(db)
                    st.rerun()
    
    with tab2:
        for student in authorized_guests:
            st.write(f"**{student}'s Recent Events:**")
            logs = db.get(student, {}).get("logs", [])
            if logs:
                for log in logs[:10]:
                    st.text(f"[{log['timestamp']}] {log['event']}: {log['notes']}")
            else: st.info("No activity recorded for this user yet.")

# ==========================================
# STUDENT SIMULATOR
# ==========================================
def run_student_simulator(vs, llm):
    user = st.session_state["current_user"]
    ping_time_tracker(user)

    # State init
    if 'display_idx' not in st.session_state:
        st.session_state.update({'db_idx': 0, 'display_idx': 1, 'correct_count': 0, 'wrong_count': 0, 'streak': 0, 'difficulty': 'NORMAL', 'phase': 'answering', 'current_q': None, 'pbq_feedback': ""})
        docs = vs.similarity_search("Security+", k=100)
        random.shuffle(docs)
        st.session_state.all_docs = docs

    with st.sidebar:
        st.header("🧭 Navigation")
        st.radio("Mode:", ["Simulator", "PBQ Lab"], key="app_mode")
        st.success(f"Score: {st.session_state.correct_count} / {st.session_state.display_idx - 1}")
        if st.button("🔄 Restart Quiz"):
            update_live_score(user, 0, 0)
            log_event(user, "Restarted", "Manual reset.")
            for k in ['db_idx', 'display_idx', 'correct_count', 'wrong_count', 'streak', 'current_q']: st.session_state.pop(k, None)
            st.rerun()

    if st.session_state.app_mode == "Simulator":
        # Adaptive logic
        if st.session_state.streak >= 3: st.session_state.difficulty = "HARD"
        elif st.session_state.streak <= -3: st.session_state.difficulty = "EASY"
        
        if not st.session_state.current_q:
            raw = st.session_state.all_docs[st.session_state.db_idx].page_content
            res = get_adaptive_question(llm, raw, st.session_state.difficulty)
            # (Parsing logic... simplified for brevity)
            # Assume successful parse into st.session_state.current_q
            
        st.subheader(f"Question {st.session_state.display_idx} ({st.session_state.difficulty})")
        st.info(st.session_state.current_q['text'] if st.session_state.current_q else "Finding question...")
        # ... (Simulator UI logic)

# ==========================================
# MAIN
# ==========================================
if check_password():
    if st.session_state["current_user"] == "admin":
        run_admin_dashboard()
    else:
        vs, llm = load_resources()
        if vs: run_student_simulator(vs, llm)
    
    with st.sidebar:
        if st.button("🚪 Log Out"):
            st.session_state.clear()
            st.rerun()
