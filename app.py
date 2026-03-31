import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. PAGE SETUP
load_dotenv()
st.set_page_config(page_title="Security+ SY0-701 Pro Sim", page_icon="🛡️", layout="wide")

# 2. RESOURCE INITIALIZATION
@st.cache_resource
def load_resources():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("🔑 GROQ_API_KEY missing in Streamlit Secrets!")
        st.stop()
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        return vectorstore, llm
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None

# 3. AI LOGIC: STRICT EXTRACTION
def get_clean_question(llm, raw_text):
    template = """
    Extract the Security+ question EXACTLY as written in the text. 
    If this is a disclaimer, copyright notice, or NOT a question, return: SKIP.
    
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
    return chain.invoke({"raw_text": raw_text[:1200]})

# 4. AI LOGIC: TUTOR FEEDBACK
def get_tutor_feedback(llm, question, user_ans, correct_letter, is_right):
    template = """
    You are a Security+ Tutor. 
    Question: {question}
    Student Choice: {user_ans}
    Correct Answer: {correct_letter}
    Result: {"Correct" if is_right else "Incorrect"}

    Explain why the correct answer is the right choice and why the other options (including the student's if wrong) are incorrect for the SY0-701 exam.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "user_ans": user_ans, "correct_letter": correct_letter, "is_right": is_right})

# 5. MAIN APPLICATION
def main():
    st.title("🛡️ Security+ SY0-701 Pro Simulator")
    vectorstore, llm = load_resources()
    if not vectorstore: return

    # CRITICAL: Initialize ALL state variables at once to prevent AttributeError
    if 'display_idx' not in st.session_state:
        # Pull 150 chunks to ensure we can find 90 valid questions after skipping junk
        st.session_state.all_docs = vectorstore.similarity_search("Security+ practice question", k=150)
        st.session_state.q_idx = 0         # Database index
        st.session_state.display_idx = 1   # User-facing question number
        st.session_state.correct_count = 0
        st.session_state.wrong_count = 0
        st.session_state.current_formatted = None
        st.session_state.feedback = None
        st.session_state.submitted = False

    # SIDEBAR SCOREBOARD
    with st.sidebar:
        st.header("📊 Exam Progress")
        st.write(f"Question: {st.session_state.display_idx} / 90")
        st.success(f"✅ Correct: {st.session_state.correct_count}")
        st.error(f"❌ Wrong: {st.session_state.wrong_count}")
        
        if st.button("🔄 Restart Exam"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # MAIN EXAM LOGIC
    if st.session_state.display_idx <= 90 and st.session_state.q_idx < len(st.session_state.all_docs):
        # 1. Fetch and Parse Question
        if st.session_state.current_formatted is None:
            raw_content = st.session_state.all_docs[st.session_state.q_idx].page_content
            with st.spinner("AI Tutor is analyzing study material..."):
                time.sleep(1.2) # Rate limit protection
                formatted = get_clean_question(llm, raw_content)
                
                # If AI says it's a disclaimer, move to next DB chunk but keep display_idx same
                if "SKIP" in formatted or "QUESTION:" not in formatted:
                    st.session_state.q_idx += 1
                    st.rerun()
                st.session_state.current_formatted = formatted

        # 2. Display Question
        try:
            f = st.session_state.current_formatted
            q_text = f.split("QUESTION:")[1].split("A:")[0].strip()
            a_opt = f.split("A:")[1].split("B:")[0].strip()
            b_opt = f.split("B:")[1].split("C:")[0].strip()
            c_opt = f.split("C:")[1].split("D:")[0].strip()
            d_opt = f.split("D:")[1].split("CORRECT:")[0].strip()
            correct_letter = f.split("CORRECT:")[1].strip()

            st.subheader(f"Question {st.session_state.display_idx}")
            st.info(q_text)
            
            opts = {f"A: {a_opt}": "A", f"B: {b_opt}": "B", f"C: {c_opt}": "C", f"D: {d_opt}": "D"}
            
            # Radio is locked after submission
            user_choice = st.radio("Select the correct option:", list(opts.keys()), index=None, disabled=st.session_state.submitted)

            # 3. Submit Answer
            if not st.session_state.submitted:
                if st.button("Submit Answer") and user_choice:
                    user_letter = opts[user_choice]
                    is_right = (user_letter == correct_letter)
                    
                    if is_right:
                        st.session_state.correct_count += 1
                        st.success(f"✅ Correct! The answer is {correct_letter}.")
                    else:
                        st.session_state.wrong_count += 1
                        st.error(f"❌ Incorrect. The correct answer was {correct_letter}.")
                    
                    # Generate AI Explanation
                    with st.spinner("AI Tutor is generating explanation..."):
                        st.session_state.feedback = get_tutor_feedback(llm, q_text, user_choice, correct_letter, is_right)
                    
                    st.session_state.submitted = True
                    st.rerun()
            
            # 4. Feedback & Next Step
            if st.session_state.submitted:
                if st.session_state.feedback:
                    st.markdown("---")
                    st.warning("🤖 **AI Tutor Explanation:**")
                    st.write(st.session_state.feedback)
                
                if st.button("Next Question ➡️"):
                    st.session_state.q_idx += 1
                    st.session_state.display_idx += 1
                    st.session_state.current_formatted = None
                    st.session_state.feedback = None
                    st.session_state.submitted = False
                    st.rerun()
                    
        except Exception:
            # Skip if the text block is messy or unparsable
            st.session_state.q_idx += 1
            st.session_state.current_formatted = None
            st.rerun()
    else:
        st.balloons()
        st.success(f"Exam Complete! Final Score: {st.session_state.correct_count} / 90")

if __name__ == "__main__":
    main()
