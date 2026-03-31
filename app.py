import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialization
load_dotenv()
st.set_page_config(page_title="Security+ SY0-701 Mastery", page_icon="🛡️", layout="wide")

def get_api_key():
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    return os.getenv("GROQ_API_KEY")

@st.cache_resource
def load_resources():
    api_key = get_api_key()
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        llm = ChatGroq(temperature=0.1, model_name="llama-3.1-70b-versatile", groq_api_key=api_key)
        return vectorstore, llm
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

def format_question(llm, raw_text):
    """AI logic to filter disclaimers and extract question details."""
    template = """
    Analyze the following text from a Security+ guide.
    1. If it's a disclaimer or legal text, return: SKIP
    2. If it's a question, extract the question and options A, B, C, D.
    3. Identify the correct answer.

    RAW TEXT: {raw_text}

    RESPONSE FORMAT:
    QUESTION: [text]
    A: [text]
    B: [text]
    C: [text]
    D: [text]
    CORRECT: [Letter]
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"raw_text": raw_text})

def main():
    st.title("🛡️ Security+ SY0-701 AI Exam Simulator")
    vectorstore, llm = load_resources()
    if not vectorstore: return

    if 'q_idx' not in st.session_state:
        st.session_state.questions = vectorstore.similarity_search("Security+ practice question", k=100)
        st.session_state.q_idx = 0
        st.session_state.score = 0
        st.session_state.current_formatted = None

    if st.session_state.q_idx < len(st.session_state.questions):
        if st.session_state.current_formatted is None:
            raw_text = st.session_state.questions[st.session_state.q_idx].page_content
            formatted = format_question(llm, raw_text)
            if "SKIP" in formatted:
                st.session_state.q_idx += 1
                st.rerun()
            st.session_state.current_formatted = formatted

        try:
            data = st.session_state.current_formatted
            q_text = data.split("QUESTION:")[1].split("A:")[0].strip()
            # Parsing logic for options...
            correct_ans = data.split("CORRECT:")[1].strip()
            
            st.subheader(f"Question {st.session_state.q_idx + 1}")
            st.info(q_text)
            choice = st.radio("Select answer:", ["A", "B", "C", "D"], index=None)

            if st.button("Submit"):
                if choice == correct_ans:
                    st.success("Correct!")
                    st.session_state.score += 1
                else:
                    st.error(f"Incorrect. Correct answer: {correct_ans}")
                
                if st.button("Next"):
                    st.session_state.q_idx += 1
                    st.session_state.current_formatted = None
                    st.rerun()
        except:
            st.session_state.q_idx += 1
            st.session_state.current_formatted = None
            st.rerun()
    else:
        st.success(f"Final Score: {st.session_state.score}")

if __name__ == "__main__":
    main()
