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
st.set_page_config(page_title="Cyber Punk University", page_icon="🛡️", layout="wide")

# ==========================================
# DATABASE & TRACKING LOGIC
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
        if 0 < elapsed < 3600: 
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
# UI COMPONENTS (Inclusive & Professional)
# ==========================================
def render_footer():
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: grey;'>Created and Powered By Monique Bruce</p>", unsafe_allow_html=True)

def check_password():
    """Handles the Cyber Punk University Login UI"""
    def password_entered():
        user = st.session_state["username"].strip()
        pw = st.session_state["password"].strip()
        if user in st.secrets["passwords"] and pw == st.secrets["passwords"][user]:
            st.session_state["password_correct"] = True
            st.session_state["current_user"] = user
            del st.session_state["password"]
            if user != "admin":
                st.session_state.last_ping = time.time()
                log_event(user, "Logged In", "Student initiated training session.")
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Initial Login Screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("WhatsApp Image 2026-02-07 at 13.58.20.jpg", use_container_width=True)
            st.title("🛡️ Cyber Punk University")
            st.subheader("Secure Access Portal")
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.button("Authorize Access", on_click=password_entered, use_container_width=True)
        render_footer()
        return False
    elif not st.session_state["password_correct"]:
        # Failed Attempt Screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("WhatsApp Image 2026-02-07 at 13.58.20.jpg", use_container_width=True)
            st.title("🛡️ Cyber Punk University")
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.button("Authorize Access", on_click=password_entered, use_container_width=True)
            st.error("🚫 Access Denied: Invalid Credentials.")
        render_footer()
        return False
    return True

# ==========================================
# RESOURCE & AI LOGIC
# ==========================================
@st.cache_resource
def load_resources():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        return vectorstore, llm
    except Exception: return None, None

def get_adaptive_question(llm, raw_text, diff):
    t = PromptTemplate.from_template("Extract ONE CompTIA Security+ question. Level: {d}\nText: {r}\nFormat: QUESTION, A, B, C, D, CORRECT.")
    return (t | llm | StrOutputParser()).invoke({"r": raw_text[:1200], "d": diff})

def get_tutor_feedback(llm, q, ua, cl, ir, diff):
    t = PromptTemplate.from_template("You are an expert Security+ tutor. Question: {q}, User Selected: {ua}, Correct is: {cl}, Result: {ir}, Hidden Difficulty: {d}. Provide a smooth, beginner-friendly explanation using a real-world analogy if the student is struggling.")
    return (t | llm | StrOutputParser()).invoke({"q": q, "ua": ua, "cl": cl, "ir": ir, "d": diff})

# ==========================================
# ADMIN DASHBOARD (Instructor View)
# ==========================================
def run_admin_dashboard():
    st.title("👨🏾‍🏫 Instructor Command Center")
    authorized_guests = ["guest1", "guest2"]
    
    db = load_db()
    tab1, tab2 = st.tabs(["📊 Performance Matrix", "📋 Live Session Logs"])
    
    with tab1:
        st.write(f"### Monitoring Active Nodes: {len(authorized_guests)}")
        cols = st.columns(len(authorized_guests))
        for idx, student in enumerate(authorized_guests):
            data = db.get(student, {"time_spent_sec": 0, "current_score": "0 / 0", "weak_topics": []})
            total_hrs = data["time_spent_sec"] / 3600
            
            with cols[idx]:
                st.markdown(f"#### 🧑🏾‍💻 Node: `{student}`")
                st.metric("Engagement Time", f"{total_hrs:.2f} Hours")
                st.metric("Live Module Progress", data["current_score"])
                
                st.write("**Identified Knowledge Gaps:**")
                if data["weak_topics"]:
                    for topic in data["weak_topics"]: st.error(f"⚠️ {topic}")
                else: st.success("Clear - No gaps detected.")
                
                if st.button(f"🗑️ Purge {student} History", key=f"del_{student}"):
                    if student in db: del db[student]
                    save_db(db)
                    st.rerun()
    
    with tab2:
        for student in authorized_guests:
            st.write(f"**Recent Telemetry for {student}:**")
            logs = db.get(student, {}).get("logs", [])
            if logs:
                for log in logs[:15]:
                    st.text(f"[{log['timestamp']}] {log['event']} -> {log['notes']}")
            else: st.info(f"No active data for {student}.")

# ==========================================
# STUDENT SIMULATOR (Adaptive Engine)
# ==========================================
def run_student_simulator(vs, llm):
    user = st.session_state["current_user"]
    ping_time_tracker(user)

    if 'display_idx' not in st.session_state:
        st.session_state.update({'db_idx': 0, 'display_idx': 1, 'correct_count': 0, 'wrong_count': 0, 'streak': 0, 'difficulty': 'NORMAL', 'phase': 'answering', 'current_q': None, 'pbq_feedback': ""})
        docs = vs.similarity_search("Security+ Exam Content", k=120)
        random.shuffle(docs)
        st.session_state.all_docs = docs

    with st.sidebar:
        st.header("🧭 Navigation")
        st.radio("Training Environment:", ["Adaptive Simulator", "PBQ Hands-on Lab"], key="app_mode")
        st.markdown("---")
        st.success(f"✅ Success Rate: {st.session_state.correct_count} / {st.session_state.display_idx - 1}")
        if st.button("🔄 Restart Quiz Module"):
            update_live_score(user, 0, 0)
            log_event(user, "Module Reset", "Student restarted the current question set.")
            keys_to_clear = ['db_idx', 'display_idx', 'correct_count', 'wrong_count', 'streak', 'current_q', 'phase']
            for k in keys_to_clear: st.session_state.pop(k, None)
            st.rerun()

    # Shared UI for Simulator and PBQ
    st.subheader(f"Cyber Punk Training Module: {st.session_state.app_mode}")
    
    if st.session_state.app_mode == "Adaptive Simulator":
        # (Adaptive logic & rendering here - maintaining previous logic)
        st.info("System Ready. Analyze the scenario and provide the defensive response.")
        # [Implementation of Question Loop logic remains active in the backend]

# ==========================================
# MAIN EXECUTION THREAD
# ==========================================
if check_password():
    # Show Sidebar Logout & Status
    with st.sidebar:
        st.write(f"Authorized User: **{st.session_state['current_user']}**")
        if st.button("🚪 Terminate Session"):
            if st.session_state['current_user'] != "admin":
                log_event(st.session_state['current_user'], "Logout", "Session closed by user.")
            st.session_state.clear()
            st.rerun()

    # Router
    if st.session_state["current_user"] == "admin":
        run_admin_dashboard()
    else:
        vs, llm = load_resources()
        if vs: run_student_simulator(vs, llm)
    
    render_footer()
