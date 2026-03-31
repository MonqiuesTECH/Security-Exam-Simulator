import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. SETUP & CONFIG
load_dotenv()
st.set_page_config(page_title="Security+ SY0-701 AI Mastery", page_icon="🛡️", layout="wide")

# 2. RESOURCE LOADING
@st.cache_resource
def init_resources():
    # Use Streamlit Secrets in production, .env in local dev
    api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.error("GROQ_API_KEY not found. Please add it to your Streamlit Secrets.")
        st.stop()

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGroq(temperature=0.1, model_name="llama-3.1-70b-versatile", groq_api_key=api_key)
        return vectorstore, llm
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None

# 3. AI EXTRACTION LOGIC (Skips disclaimers & fixes options)
def format_question_logic(llm, raw_text):
    template = """
    You are an expert CompTIA Security+ instructor. 
    Analyze the provided raw text from a study guide.

    INSTRUCTIONS:
    1. If the text is a legal disclaimer, copyright notice, or title page, return exactly: SKIP
    2. If it is a question, extract the question, four options (A, B, C, D), and identify the correct answer.

    RAW TEXT: {raw_text}

    RESPONSE FORMAT:
    QUESTION: [text]
    A: [text]
    B: [text]
    C: [text]
    D: [text]
    CORRECT: [Just the Letter]
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"raw_text": raw_text})

# 4. MAIN APPLICATION
def main():
    st.title("🛡️ Security+ SY0-701 AI Exam Simulator")
    
    vectorstore, llm = init_resources()
    if not vectorstore: return

    # Session State Management
    if 'q_idx' not in st.session_state:
        # Search for practice-related content
        docs = vectorstore.similarity_search("Security+ practice exam question", k=100)
        st.session_state.questions = docs
        st.session_state.q_idx = 0
        st.session_state.score = 0
        st.session_state.current_data = None

    # App Logic
    if st.session_state.q_idx < len(st.session_state.questions):
        # Format the question if it hasn't been done for this index
        if st.session_state.current_data is None:
            raw_content = st.session_state.questions[st.session_state.q_idx].page_content
            with st.spinner("AI is analyzing exam material..."):
                formatted = format_question_logic(llm, raw_content)
            
            if "SKIP" in formatted:
                st.session_state.q_idx += 1
                st.rerun()
            else:
                st.session_state.current_data = formatted

        # Parsing the AI-generated data
        try:
            data = st.session_state.current_data
            question_body = data.split("QUESTION:")[1].split("A:")[0].strip()
            ans_a = data.split("A:")[1].split("B:")[0].strip()
            ans_b = data.split("B:")[1].split("C:")[0].strip()
            ans_c = data.split("C:")[1].split("D:")[0].strip()
            ans_d = data.split("D:")[1].split("CORRECT:")[0].strip()
            final_correct = data.split("CORRECT:")[1].strip()
        except Exception:
            # Skip if parsing fails
            st.session_state.q_idx += 1
            st.session_state.current_data = None
            st.rerun()

        st.subheader(f"Question {st.session_state.q_idx + 1}")
        st.info(question_body)
        
        # Display buttons with the actual text of the answers
        mapping = {
            f"A: {ans_a}": "A",
            f"B: {ans_b}": "B",
            f"C: {ans_c}": "C",
            f"D: {ans_d}": "D"
        }
        
        user_choice = st.radio("Choose the correct option:", list(mapping.keys()), index=None)

        if st.button("Submit Answer") and user_choice:
            user_letter = mapping[user_choice]
            if user_letter == final_correct:
                st.success(f"Correct! {user_letter} is right.")
                st.session_state.score += 1
            else:
                st.error(f"Incorrect. The correct answer was {final_correct}.")
            
            if st.button("Next Question"):
                st.session_state.q_idx += 1
                st.session_state.current_data = None
                st.rerun()
    else:
        st.balloons()
        st.success(f"Session Complete! Final Score: {st.session_state.score}/{len(st.session_state.questions)}")

if __name__ == "__main__":
    main()
