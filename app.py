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
    - If EASY: Extract questions about basic definitions, ports, or core concepts.
    - If NORMAL: Extract standard scenario-based questions.
    - If HARD: Extract complex scenarios, log analysis, or multi-step troubleshooting.
    
    If the text is a disclaimer, table of contents, or does not match the difficulty, return: SKIP.
    
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
    You are an expert, empathetic CompTIA Security+ (SY0-701) Tutor. 
    Question: {question}
    Student Choice: {user_ans}
    Correct Answer: {correct_letter}
    Result: {result}
    Student's Current Hidden Level: {difficulty}
    
    Task: Write a smooth, highly readable explanation tailored to the student's level.
    1. If the level is EASY (student is struggling): Use a simple, real-world analogy (like a house, a bouncer, a lock) to explain the core concept. Avoid dense jargon. Focus on building confidence.
    2. If the level is HARD (student is excelling): Go deeper into the technical "why" and provide an advanced pro-tip.
    3. If NORMAL: Provide a clear, standard CompTIA explanation.
    4. Explicitly state why the correct answer is the BEST choice.
    5. If the student was incorrect, gently explain the flaw in their selection.
    6. Use formatting (bolding key terms) to make it easy to read.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "question": question, 
        "user_ans": user_ans, 
        "correct_letter": correct_letter, 
        "result": result_text,
        "difficulty": difficulty
    })

# 5. AI LOGIC: TOPIC EXTRACTOR FOR VIDEO REHAB
def get_video_topic(llm, question):
    template = """
    Analyze this Security+ question and extract the ONE core concept or technology it is asking about in 2 to 4 words.
    Example: If it's about ports, output "Network Ports". If it's about Active Directory, output "802.1X Authentication".
    
    Question: {question}
    
    Output ONLY the short topic name, nothing else.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

# 6. MAIN APPLICATION
def main():
    st.title("🛡️ Security+ SY0-701 Pro Simulator")
    vectorstore, llm = load_resources()
    if not vectorstore: return

    # --- BULLETPROOF STATE INITIALIZATION ---
    if 'display_idx' not in st.session_state:
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

    # --- ADAPTIVE DIFFICULTY LOGIC ---
    if st.session_state.streak >= 3:
        st.session_state.difficulty = "HARD"
    elif st.session_state.streak <= -2: # Switches to EASY after 2 wrong to try and save them
        st.session_state.difficulty = "EASY"
    else:
        st.session_state.difficulty = "NORMAL"

    # --- SIDEBAR SCOREBOARD & QUICK LINKS ---
    with st.sidebar:
        st.header("📊 Exam Progress")
        st.write(f"**Question:** {st.session_state.display_idx} / 90")
        
        st.markdown("---")
        st.success(f"✅ Correct: {st.session_state.correct_count}")
        st.error(f"❌ Wrong: {st.session_state.wrong_count}")
        
        st.markdown("---")
        st.header("📺 Quick Review Topics")
        st.write("Watch Professor Messer lessons anytime:")
        # Core SY0-701 Domains
        topics = [
            "1.0 General Security Concepts",
            "2.0 Threats, Vulnerabilities, & Mitigations",
            "3.0 Security Architecture",
            "4.0 Security Operations",
            "5.0 Security Program Management"
        ]
        for topic in topics:
            search_query = urllib.parse.quote(f"Professor Messer SY0-701 {topic}")
            st.markdown(f"- [{topic}](https://www.youtube.com/results?search_query={search_query})")
        
        st.markdown("---")
        if st.button("🔄 Restart Exam"):
            st.session_state.clear()
            st.rerun()

    if st.session_state.display_idx > 90 or st.session_state.db_idx >= len(st.session_state.all_docs):
        st.balloons()
        st.success(f"🎉 Exam Finished! Final Score: {st.session_state.correct_count} / 90")
        return

    # --- QUESTION FETCHING LOGIC ---
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
                        
                        raw_correct = formatted.split("CORRECT:")[1].strip()
                        correct_letter = raw_correct[0].upper()
                        
                        if correct_letter not in ["A", "B", "C", "D"]:
                            raise ValueError("Invalid letter")
                        
                        st.session_state.current_q = {
                            "text": q_text,
                            "options": [f"A: {a_opt}", f"B: {b_opt}", f"C: {c_opt}", f"D: {d_opt}"],
                            "correct_letter": correct_letter
                        }
                        break 
                    except Exception:
                        st.session_state.db_idx += 1 
                else:
                    st.session_state.db_idx += 1 

    if st.session_state.current_q is None:
        st.error("Ran out of study material in the database.")
        return

    # --- UI RENDERING ---
    cq = st.session_state.current_q
    
    # PHASE 1: ANSWERING
    if st.session_state.phase == "answering":
        st.subheader(f"Question {st.session_state.display_idx}")
        st.info(cq["text"])
        selected_option = st.radio("Select your answer:", cq["options"], index=None)

        if st.button("Submit Answer"):
            if selected_option is None:
                st.warning("⚠️ Please select an answer before submitting.")
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
                    st.session_state.feedback = get_tutor_feedback(
                        llm, cq["text"], selected_option, cq["correct_letter"], is_right, st.session_state.difficulty
                    )
                
                st.session_state.phase = "reviewing"
                st.rerun()

    # PHASE 2: REVIEWING
    elif st.session_state.phase == "reviewing":
        st.subheader(f"Question {st.session_state.display_idx}")
        st.info(cq["text"])
        st.radio("Your answer:", cq["options"], index=cq["options"].index(st.session_state.user_choice), disabled=True)
        
        st.markdown("---")
        if st.session_state.is_right:
            st.success(f"✅ Correct! The answer is {cq['correct_letter']}.")
        else:
            st.error(f"❌ Incorrect. The correct answer was {cq['correct_letter']}.")
        
        st.warning("🤖 **AI Tutor Explanation:**")
        st.write(st.session_state.feedback)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Next Question ➡️"):
            # CHECK FOR VIDEO TIMEOUT (3 Wrong in a row)
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

    # PHASE 3: VIDEO REHAB TIMEOUT
    elif st.session_state.phase == "video_rehab":
        st.error("🛑 Study Timeout Triggered: You've missed 3 questions in a row.")
        st.write("When we guess, we stop learning. Let's review this concept before moving forward.")
        
        st.markdown("---")
        st.subheader(f"📚 Recommended Review Topic: **{st.session_state.rehab_topic}**")
        
        # Create a dynamic search link to Professor Messer
        search_query = urllib.parse.quote(f"Professor Messer SY0-701 {st.session_state.rehab_topic}")
        youtube_link = f"https://www.youtube.com/results?search_query={search_query}"
        
        st.info(f"The industry standard for this certification is Professor Messer. Please click the link below to find his lesson on **{st.session_state.rehab_topic}**.")
        st.markdown(f"### 📺 [Click Here to Search YouTube for '{st.session_state.rehab_topic}' Lessons]({youtube_link})")
        
        st.markdown("---")
        st.write("*(Honor System: I cannot see your browser to verify you watched the video. You are studying for your own future!)*")
        
        # Text input to proceed
        user_done = st.text_input("Type **done** when you have finished reviewing the material:")
        
        if user_done.strip().lower() == "done":
            st.success("Great job reviewing! Let's get back into the exam.")
            if st.button("Resume Exam ➡️"):
                st.session_state.streak = 0         # Reset streak so they aren't trapped
                st.session_state.db_idx += 1        
                st.session_state.display_idx += 1   
                st.session_state.current_q = None   
                st.session_state.phase = "answering"
                st.rerun()

if __name__ == "__main__":
    main()
