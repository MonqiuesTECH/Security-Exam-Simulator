import streamlit as st
import os
import time
import random
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. SETUP
load_dotenv()
st.set_page_config(page_title="Security+ SY0-701 Adaptive Tutor", page_icon="🛡️", layout="wide")

# 2. RESOURCE LOADING
@st.cache_resource
def load_resources():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("🔑 GROQ_API_KEY missing in Secrets!")
        st.stop()
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        return vectorstore, llm
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None

# 3. AI LOGIC: ADAPTIVE QUESTION PARSER
def get_adaptive_question(llm, raw_text, difficulty):
    template = """
    Extract ONE CompTIA Security+ question EXACTLY as written.
    
    DIFFICULTY LEVEL: {difficulty}
    - If EASY: Focus on basic definitions and core concepts.
    - If HARD: Focus on complex scenarios, logs, and multi-step troubleshooting.
    
    If the text doesn't match this difficulty or isn't a question, return: SKIP.
    
    RAW TEXT: {raw_text}

    FORMAT:
    QUESTION: [text]
    A: [text]
    B: [text]
    C: [text]
    D: [text]
    CORRECT: [Letter Only]
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"raw_text": raw_text[:1500], "difficulty": difficulty})

# 4. AI LOGIC: TUTOR FEEDBACK
def get_tutor_feedback(llm, question, user_ans, correct_letter, is_right):
    result_text = "Correct" if is_right else "Incorrect"
    template = """
    You are a friendly Security+ Tutor. 
    Question: {question}
    Student Choice: {user_ans}
    Correct Answer: {correct_letter}
    Result: {result}
    
    Explain the concept using a simple analogy. Focus on building the student's confidence.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "user_ans": user_ans, "correct_letter": correct_letter, "result": result_text})

# 5. MAIN APP
def main():
    st.title("🛡️ Adaptive Security+ Tutor")
    vectorstore, llm = load_resources()
    if not vectorstore: return

    # --- INITIALIZE STATE ---
    if 'display_idx' not in st.session_state:
        st.session_state.db_idx = 0         
        st.session_state.display_idx = 1    
        st.session_state.correct_count = 0  
        st.session_state.wrong_count = 0    
        st.session_state.streak = 0         # Positive = Win streak, Negative = Loss streak
        st.session_state.difficulty = "NORMAL"
        
        # Initial doc pull
        docs = vectorstore.similarity_search("Security+ practice question", k=200)
        random.shuffle(docs)
        st.session_state.all_docs = docs
        
        st.session_state.current_q = None   
        st.session_state.phase = "answering" 
        st.session_state.feedback = ""
        st.session_state.user_choice = None
        st.session_state.is_right = False

    # --- SIDEBAR ADAPTIVE MONITOR ---
    with st.sidebar:
        st.header("📊 Performance Monitor")
        st.write(f"**Current Difficulty:** {st.session_state.difficulty}")
        st.write(f"**Current Streak:** {st.session_state.streak}")
        st.markdown("---")
        st.success(f"✅ Correct: {st.session_state.correct_count}")
        st.error(f"❌ Wrong: {st.session_state.wrong_count}")
        
        if st.button("🔄 Restart Exam"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    # --- ADAPTIVE DIFFICULTY LOGIC ---
    if st.session_state.streak >= 3:
        st.session_state.difficulty = "HARD"
    elif st.session_state.streak <= -5:
        st.session_state.difficulty = "EASY"
    else:
        st.session_state.difficulty = "NORMAL"

    # --- QUESTION FETCHING ---
    if st.session_state.current_q is None:
        with st.spinner(f"Finding a {st.session_state.difficulty} question..."):
            while st.session_state.db_idx < len(st.session_state.all_docs):
                time.sleep(1)
                raw_content = st.session_state.all_docs[st.session_state.db_idx].page_content
                formatted = get_adaptive_question(llm, raw_content, st.session_state.difficulty)
                
                if "SKIP" not in formatted and "QUESTION:" in formatted:
                    try:
                        q_parts = formatted.split("QUESTION:")[1]
                        q_text = q_parts.split("A:")[0].strip()
                        a_opt = q_parts.split("A:")[1].split("B:")[0].strip()
                        b_opt = q_parts.split("B:")[1].split("C:")[0].strip()
                        c_opt = q_parts.split("C:")[1].split("D:")[0].strip()
                        d_opt = q_parts.split("D:")[1].split("CORRECT:")[0].strip()
                        correct_letter = q_parts.split("CORRECT:")[1].strip()[0].upper()
                        
                        st.session_state.current_q = {
                            "text": q_text,
                            "options": [f"A: {a_opt}", f"B: {b_opt}", f"C: {c_opt}", f"D: {d_opt}"],
                            "correct_letter": correct_letter
                        }
                        break
                    except:
                        st.session_state.db_idx += 1
                else:
                    st.session_state.db_idx += 1

    # --- UI ---
    if st.session_state.current_q:
        cq = st.session_state.current_q
        st.subheader(f"Question {st.session_state.display_idx} ({st.session_state.difficulty})")
        st.info(cq["text"])

        if st.session_state.phase == "answering":
            user_pick = st.radio("Select answer:", cq["options"], index=None)
            if st.button("Submit Answer") and user_pick:
                user_letter = user_pick.split(":")[0]
                is_right = (user_letter == cq["correct_letter"])
                
                # UPDATE STREAK
                if is_right:
                    st.session_state.correct_count += 1
                    st.session_state.streak = max(1, st.session_state.streak + 1) if st.session_state.streak >= 0 else 1
                else:
                    st.session_state.wrong_count += 1
                    st.session_state.streak = min(-1, st.session_state.streak - 1) if st.session_state.streak <= 0 else -1
                
                st.session_state.user_choice = user_pick
                st.session_state.is_right = is_right
                st.session_state.feedback = get_tutor_feedback(llm, cq["text"], user_pick, cq["correct_letter"], is_right)
                st.session_state.phase = "reviewing"
                st.rerun()

        elif st.session_state.phase == "reviewing":
            st.radio("Your answer:", cq["options"], index=cq["options"].index(st.session_state.user_choice), disabled=True)
            st.markdown("---")
            if st.session_state.is_right: st.success("Correct!")
            else: st.error(f"Incorrect. Answer was {cq['correct_letter']}")
            st.warning("🤖 AI Tutor Explanation:")
            st.write(st.session_state.feedback)
            
            if st.button("Next Question ➡️"):
                st.session_state.db_idx += 1
                st.session_state.display_idx += 1
                st.session_state.current_q = None
                st.session_state.phase = "answering"
                st.rerun()

if __name__ == "__main__":
    main()
