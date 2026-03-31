import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. SETUP & PAGE CONFIG
load_dotenv()
st.set_page_config(page_title="Security+ SY0-701 AI Mastery", page_icon="🛡️", layout="wide")

# 2. SECURE RESOURCE INITIALIZATION
@st.cache_resource
def load_exam_resources():
    # Priority: Streamlit Secrets, then local .env
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.error("🔑 GROQ_API_KEY is missing! Please add it to your Streamlit Secrets.")
        st.stop()

    try:
        # Load Embeddings (sentence-transformers)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load FAISS Index (ensure 'faiss_index' folder is in your repo)
        vectorstore = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # UPDATED: Using the current 2026 replacement for decommissioned models
        llm = ChatGroq(
            temperature=0.1, 
            model_name="llama-3.3-70b-versatile", 
            groq_api_key=api_key
        )
        return vectorstore, llm
    except Exception as e:
        st.error(f"Critical Load Error: {e}")
        return None, None

# 3. AI EXTRACTION & CLEANING (Optimized for Free Tier)
def get_clean_question(llm, raw_text):
    # Free Tier Safety: Truncate to avoid 'Request too large' errors
    text_sample = raw_text[:1200]
    
    template = """
    You are a CompTIA Security+ SY0-701 Instructor. 
    Analyze the text below. 
    
    RULES:
    1. If the text is a legal disclaimer, table of contents, or copyright notice, return: SKIP
    2. If it is a question, extract the question and full text for options A, B, C, and D.
    3. Identify the correct answer letter.

    RAW TEXT: {raw_text}

    RESPONSE FORMAT:
    QUESTION: [text]
    A: [text]
    B: [text]
    C: [text]
    D: [text]
    CORRECT: [Letter Only]
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"raw_text": text_sample})

# 4. MAIN APP INTERFACE
def main():
    st.title("🛡️ Security+ SY0-701 AI Exam Simulator")
    st.markdown("---")

    vectorstore, llm = load_exam_resources()
    if not vectorstore: return

    # Session State Setup
    if 'q_idx' not in st.session_state:
        # Pull 60 random samples from your study guide
        st.session_state.all_docs = vectorstore.similarity_search("Security+ practice exam question", k=60)
        st.session_state.q_idx = 0
        st.session_state.score = 0
        st.session_state.current_formatted = None

    if st.session_state.q_idx < len(st.session_state.all_docs):
        # AI Formatting Layer
        if st.session_state.current_formatted is None:
            raw_content = st.session_state.all_docs[st.session_state.q_idx].page_content
            
            with st.spinner("AI Tutor is analyzing (Free Tier Mode)..."):
                try:
                    # Small delay to respect Free Tier Rate Limits
                    time.sleep(1.5)
                    formatted = get_clean_question(llm, raw_content)
                    
                    if "SKIP" in formatted:
                        st.session_state.q_idx += 1
                        st.rerun()
                    st.session_state.current_formatted = formatted
                except Exception as e:
                    st.error(f"AI Service Error: {e}")
                    st.info("Check if your Groq Free Tier daily limit has been reached.")
                    st.stop()

        # UI Logic
        try:
            f = st.session_state.current_formatted
            q_text = f.split("QUESTION:")[1].split("A:")[0].strip()
            a_opt = f.split("A:")[1].split("B:")[0].strip()
            b_opt = f.split("B:")[1].split("C:")[0].strip()
            c_opt = f.split("C:")[1].split("D:")[0].strip()
            d_opt = f.split("D:")[1].split("CORRECT:")[0].strip()
            correct_letter = f.split("CORRECT:")[1].strip()

            st.subheader(f"Question {st.session_state.q_idx + 1}")
            st.info(q_text)
            
            options = {f"A: {a_opt}": "A", f"B: {b_opt}": "B", f"C: {c_opt}": "C", f"D: {d_opt}": "D"}
            user_choice = st.radio("Choose the correct option:", list(options.keys()), index=None)

            if st.button("Submit Answer") and user_choice:
                user_letter = options[user_choice]
                if user_letter == correct_letter:
                    st.success(f"✅ Correct! The answer is {correct_letter}.")
                    st.session_state.score += 1
                else:
                    st.error(f"❌ Incorrect. The answer was {correct_letter}.")

                if st.button("Next Question"):
                    st.session_state.q_idx += 1
                    st.session_state.current_formatted = None
                    st.rerun()
        except:
            # Skip if the AI returns a format the app can't parse
            st.session_state.q_idx += 1
            st.session_state.current_formatted = None
            st.rerun()
    else:
        st.success(f"Session Finished! Final Score: {st.session_state.score}/{len(st.session_state.all_docs)}")

if __name__ == "__main__":
    main()
