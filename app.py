import streamlit as st
import os
import random
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. SETUP & CONFIG
load_dotenv()
st.set_page_config(page_title="Security+ SY0-701 AI Tutor", page_icon="🛡️", layout="wide")

# 2. RESOURCE LOADING
@st.cache_resource
def init_resources():
    api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGroq(temperature=0.1, model_name="llama-3.1-70b-versatile", groq_api_key=api_key)
        return vectorstore, llm
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None

# 3. AI CLEANING LOGIC (This fixes the 'Disclaimer' and 'Empty Options' issue)
def format_question(llm, raw_text):
    template = """
    You are an expert at parsing CompTIA Security+ study materials. 
    Take the following raw text and extract the exam question and the multiple-choice options.

    RULES:
    1. If the text is a legal disclaimer, table of contents, or NOT a question, return exactly: SKIP
    2. If it is a question, format it clearly with A, B, C, and D labels.
    3. Identify the correct answer based on the text.

    RAW TEXT: {raw_text}

    RESPONSE FORMAT:
    QUESTION: [The question text]
    A: [Option A]
    B: [Option B]
    C: [Option C]
    D: [Option D]
    CORRECT: [Just the Letter]
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"raw_text": raw_text})

# 4. MAIN APP
def main():
    st.title("🛡️ Security+ SY0-701 Mastery")
    
    vectorstore, llm = init_resources()
    if not vectorstore: return

    # Session State
    if 'q_idx' not in st.session_state:
        # Pull 100 chunks to ensure we have enough after skipping disclaimers
        docs = vectorstore.similarity_search("Security+ practice question", k=100)
        st.session_state.questions = docs
        st.session_state.q_idx = 0
        st.session_state.score = 0
        st.session_state.current_formatted = None

    # Process Current Question
    if st.session_state.q_idx < len(st.session_state.questions):
        if st.session_state.current_formatted is None:
            raw_text = st.session_state.questions[st.session_state.q_idx].page_content
            with st.spinner("AI is preparing your next question..."):
                formatted = format_question(llm, raw_text)
                
            if "SKIP" in formatted:
                st.session_state.q_idx += 1
                st.rerun()
            else:
                st.session_state.current_formatted = formatted

        # Parsing the AI output
        try:
            data = st.session_state.current_formatted
            q_text = data.split("QUESTION:")[1].split("A:")[0].strip()
            opt_a = data.split("A:")[1].split("B:")[0].strip()
            opt_b = data.split("B:")[1].split("C:")[0].strip()
            opt_c = data.split("C:")[1].split("D:")[0].strip()
            opt_d = data.split("D:")[1].split("CORRECT:")[0].strip()
            correct_ans = data.split("CORRECT:")[1].strip()
        except:
            st.session_state.q_idx += 1
            st.session_state.current_formatted = None
            st.rerun()

        # UI DISPLAY
        st.subheader(f"Question {st.session_state.q_idx + 1}")
        st.info(q_text)
        
        # Display options as full text
        options_map = {f"A: {opt_a}": "A", f"B: {opt_b}": "B", f"C: {opt_c}": "C", f"D: {opt_d}": "D"}
        choice = st.radio("Select the correct answer:", list(options_map.keys()), index=None)

        if st.button("Submit Answer") and choice:
            user_letter = options_map[choice]
            if user_letter == correct_ans:
                st.success(f"Correct! {user_letter} is the right answer.")
                st.session_state.score += 1
            else:
                st.error(f"Incorrect. The correct answer was {correct_ans}.")
            
            if st.button("Next Question"):
                st.session_state.q_idx += 1
                st.session_state.current_formatted = None
                st.rerun()
    else:
        st.success(f"Exam Complete! Final Score: {st.session_state.score}")

if __name__ == "__main__":
    main()
