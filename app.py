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
    If this is a disclaimer or NOT a question, return: SKIP.
    
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
    You are an expert Security+ Tutor (SY0-701). 
    Question: {question}
    Student Selected: {user_ans}
    Correct Answer: {correct_letter}
    Result: {"Correct" if is_right else "Incorrect"}

    Task: Provide a high-quality explanation. 
    1. Explain why the correct answer is the standard CompTIA choice.
    2. Explain why the other options (including the student's if wrong) are technically incorrect.
    3. Add a short 'Study Tip' for this concept.
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
        st.session_state.all_docs = vectorstore.similarity_search("Security+ practice question", k=150)
        st.session_state.q_idx = 0         
        st.session_state.display_idx = 1   
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
        if st.session_state.current_formatted is None:
            raw_content = st.session_state.all_docs[st.session_state.q_idx].page_content
            with st.spinner("AI Tutor is analyzing study material..."):
                time.sleep(1.2) 
                formatted = get_clean_question(llm, raw_content)
                if "SKIP" in formatted or "QUESTION:" not in formatted:
                    st.session_state.q_idx += 1
                    st.rerun()
                st.session_state.current_formatted = formatted

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
            user_choice = st.radio("Select choice:", list(opts.keys()), index=None, disabled=st.session_state.submitted)

            # SUBMIT ACTION
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
                    
                    with st.spinner("AI Tutor is generating explanation..."):
                        st.session_state.feedback = get_tutor_feedback(llm, q_text, user_choice, correct_letter, is_right)
                    
                    st.session_state.submitted = True
                    st.rerun()
            
            # FEEDBACK & NEXT ACTION
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
            st.session_state.q_idx += 1
            st.session_state.current_formatted = None
            st.rerun()
    else:
        st.balloons()
        st.success(f"Exam Finished! Final Score: {st.session_state.correct_count} / 90")

if __name__ == "__main__":
    main()
