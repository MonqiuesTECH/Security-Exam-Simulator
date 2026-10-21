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
# FIX 1: HIDE STREAMLIT MENU & FOOTER
# ==========================================
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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

def save_user_state(user):
    """Saves critical session variables to the JSON database."""
    db = load_db()
    if user in db:
        state_keys = ['db_idx', 'display_idx', 'correct_count', 'wrong_count', 'streak', 'difficulty', 'app_mode', 'phase']
        db[user]["saved_progress"] = {k: st.session_state[k] for k in state_keys if k in st.session_state}
        save_db(db)

def load_user_state(user):
    """Restores saved variables from the JSON database into the session."""
    db = load_db()
    if user in db and "saved_progress" in db[user] and len(db[user]["saved_progress"]) > 0:
        for k, v in db[user]["saved_progress"].items():
            st.session_state[k] = v
        return True
    return False

def ensure_user_exists(user):
    db = load_db()
    if user not in db:
        db[user] = {
            "time_spent_sec": 0, 
            "logs": [], 
            "weak_topics": [], 
            "current_score": "0 / 0",
            "has_completed_practice": False,
            "timed_scores": [],
            "saved_progress": {}
        }
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

def mark_practice_complete(user):
    db = load_db()
    if user in db:
        db[user]["has_completed_practice"] = True
        save_db(db)

def save_timed_score(user, score_str):
    db = load_db()
    if user in db:
        db[user]["timed_scores"].append(score_str)
        save_db(db)

# ==========================================
# UI COMPONENTS
# ==========================================
def render_footer():
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: grey;'>Created and Powered By Monique Bruce</p>", unsafe_allow_html=True)

# FIX 2: IMPROVED LOGIN TO PREVENT CRASHES
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    def password_entered():
        user = st.session_state.get("username", "").strip().lower()
        pw = st.session_state.get("password", "").strip()
        
        try:
            if user in st.secrets["passwords"] and pw == st.secrets["passwords"][user]:
                st.session_state["password_correct"] = True
                st.session_state["current_user"] = user
                st.session_state["login_error"] = False
                if "password" in st.session_state:
                    del st.session_state["password"]
                if user != "admin":
                    st.session_state.last_ping = time.time()
                    log_event(user, "Logged In", "Student initiated training session.")
            else:
                st.session_state["password_correct"] = False
                st.session_state["login_error"] = True
        except Exception as e:
            st.session_state["login_error"] = True

    if not st.session_state["password_correct"]:
        st.markdown("<br><br>", unsafe_allow_html=True)
        spacer_left, col_img, col_form, spacer_right = st.columns([1, 2, 2, 1])
        
        with col_img:
            image_name = "logo.jpeg"
            if os.path.exists(image_name):
                st.image(image_name, use_container_width=True)
            else:
                st.info("Logo Placeholder (Ensure logo.jpeg is in your GitHub Repo)")

        with col_form:
            st.title("🛡️ Cyber Punk University")
            st.subheader("Secure Access Portal")
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.button("Authorize Access", on_click=password_entered, use_container_width=True)
            
            if st.session_state.get("login_error"):
                st.error("🚫 Access Denied: Invalid Credentials or System Error.")
                
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        render_footer()
        return False
        
    return True

# ==========================================
# RESOURCE & AI LOGIC
# ==========================================
@st.cache_resource
def load_resources():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("🔑 GROQ_API_KEY is missing in Streamlit Secrets!")
        st.stop()
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        # FIX 5: GROQ MODEL STRING UPDATED
        llm = ChatGroq(temperature=0, model_name="openai/gpt-oss-20b", groq_api_key=api_key) 
        return vector
