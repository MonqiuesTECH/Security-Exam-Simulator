import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. SETUP
load_dotenv()
st.set_page_config(page_title="Security+ SY0-701 Pro Sim", page_icon="🛡️", layout="wide")

# 2. RESOURCE INITIALIZATION
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
        st.error(f"Load Error: {e}")
        return None, None

# 3. AI LOGIC: STRICT EXTRACTION
def get_clean_question(llm, raw_text):
    template = """
    Extract the Security+ question EXACTLY as written. 
    If this is a disclaimer or NOT a question, return: SKIP.
    RAW TEXT: {raw_text}
    FORMAT:
    QUESTION: [text]
    A: [text]
    B: [text]
    C: [text]
    D: [text]
    CORRECT: [Letter]
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"raw_text": raw_text[:1200]})

# 4. AI LOGIC: TUTOR FEEDBACK
def get_tutor_feedback(llm, question, user_ans, correct_letter, is_right):
    template = """
    You are an expert Security+ Tutor. 
    Question: {question}
    Student Choice: {user_ans}
    Correct Answer: {correct_letter}
    Result: {"Correct" if is_right else "Incorrect"}
    
    Explain why the correct answer is the BEST choice and why the other options (especially the student's choice if wrong) are technically incorrect for the SY0-701 exam.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "user_ans": user_ans, "correct_letter": correct_letter, "is_right": is_right})

# 5. MAIN APP
def main():
    st.title("🛡️ Security+ SY0-701 Pro Simulator")
    vectorstore, llm = load_resources()
    if not vectorstore: return

    # INITIALIZE ALL STATE VARIABLES AT ONCE
    if 'display_idx' not in st.session_state:
        st.session_state.all_docs = vectorstore.similarity_search("Security+ practice question", k=150)
        st.session_state.db_idx = 0         # The actual index in the database
        st.session_state.display_idx = 1    # What you see on screen (1-90)
        st.session_state.correct_count = 0
        st.session_state.wrong_count = 0
        st.session_state.current_formatted = None
        st.session_state.feedback = None
        st.session_state.submitted = False

    # SIDEBAR SCOREBOARD
    with st.sidebar:
        st.header("📊 Scoreboard")
        st.write(f"Question: {st.session_state.display_idx} / 90")
        st.success(f"✅ Correct: {st.session_state.correct_count}")
        st.error(f"❌ Wrong: {st.session_state.wrong_count}")
        if st.button("🔄 Restart Exam"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    # STOP AT 90 QUESTIONS
    if st.session_state.display_idx > 90 or st.session_state.db_idx >= len(st.session_state.all_docs):
        st.balloons()
        st.success(f"Exam Finished! Final Score: {st.session_state.correct_count} / 90")
        return

    # PREPARE QUESTION (Only if we don't have one)
    if st.session_state.current_formatted is None:
        raw_content = st.session_state.all_docs[st.session_state.db_idx].page_content
        with st.spinner("AI finding valid question..."):
            time.sleep(1.2)
            formatted = get_clean_question(llm, raw_content)
            if "SKIP" in formatted or "QUESTION:" not in formatted:
                st.session_state.db_idx += 1
                st.rerun()
            st.session_state.current_formatted = formatted

    # PARSE DATA
    try:
        f = st.session_state.current_formatted
        q_text = f.split("QUESTION:")[1].split("A:")[0].strip()
        a_opt = f.split("A:")[1].split("B:")[0].strip()
        b_opt = f.split("B:")[1].split("C:")[0].strip()
        c_opt = f.split("C:")[1].split("D:")[0].strip()
        d_opt = f.split("D:")[1].split("CORRECT:")[0].strip()
        correct_letter = f.split("CORRECT:")[1].strip()
    except Exception:
        st.session_state.db_idx += 1
        st.session_state.current_formatted = None
        st.rerun()

    # DISPLAY UI
    st.subheader(f"Question {st.session_state.display_idx}")
    st.info(q_text)
    
    opts = {f"A: {a_opt}": "A", f"B: {b_opt}": "B", f"C: {c_opt}": "C", f"D: {d_opt}": "D"}
    
    # Selection logic
    user_choice = st.radio("Select choice:", list(opts.keys()), index=None, disabled=st.session_state.submitted)

    # SUBMIT PHASE
    if not st.session_state.submitted:
        if st.button("Submit Answer") and user_choice:
            user_letter = opts[user_choice]
            is_right = (user_letter == correct_letter)
            
            # Update score immediately
            if is_right: st.session_state.correct_count += 1
            else: st.session_state.wrong_count += 1
            
            # Fetch feedback
            with st.spinner("AI Tutor generating explanation..."):
                st.session_state.feedback = get_tutor_feedback(llm, q_text, user_choice, correct_letter, is_right)
            
            st.session_state.submitted = True
            st.rerun()
    
    # REVIEW PHASE (Only shows after Submit)
    if st.session_state.submitted:
        st.markdown("---")
        # Visual cue for result
        if opts[user_choice] == correct_letter:
            st.success(f"Correct! The answer was {correct_letter}")
        else:
            st.error(f"Incorrect. The answer was {correct_letter}")
        
        st.warning("🤖 **AI Tutor Explanation:**")
        st.write(st.session_state.feedback)
        
        # Next button is the ONLY way to advance
        if st.button("Next Question ➡️"):
            st.session_state.db_idx += 1
            st.session_state.display_idx += 1
            st.session_state.current_formatted = None
            st.session_state.feedback = None
            st.session_state.submitted = False
            st.rerun()

if __name__ == "__main__":
    main()
