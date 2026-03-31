import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. INITIALIZATION & SECURITY
load_dotenv()
st.set_page_config(page_title="Security+ SY0-701 Mastery", page_icon="🛡️", layout="wide")

# Secure Key Retrieval
def get_api_key():
    # Priority 1: Streamlit Secrets (Production)
    # Priority 2: .env file (Local Dev)
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    return os.getenv("GROQ_API_KEY")

# 2. ASSET LOADING (CACHED)
@st.cache_resource
def load_resources():
    api_key = get_api_key()
    if not api_key:
        st.error("🔑 API Key Missing! Add it to Streamlit Secrets or your .env file.")
        st.stop()

    try:
        # Fixed: Using the updated langchain-huggingface package
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load your FAISS index (Ensure this folder is in your GitHub repo)
        vectorstore = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        llm = ChatGroq(
            temperature=0.2, # Low temp for factual accuracy
            model_name="llama-3.1-70b-versatile",
            groq_api_key=api_key
        )
        return vectorstore, llm
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        return None, None

# 3. AI TUTOR LOGIC (The "Why am I wrong?" Engine)
def explain_mistake(llm, question, user_ans, correct_ans):
    template = """
    You are a CompTIA Security+ Expert. A student just got a question wrong.
    
    QUESTION: {question}
    THEIR WRONG ANSWER: {user_ans}
    THE ACTUAL CORRECT ANSWER: {correct_ans}
    
    INSTRUCTIONS:
    1. Explain WHY the correct answer is technically accurate per SY0-701 standards.
    2. Gently explain the flaw in the student's logic for choosing {user_ans}.
    3. Provide a 'Security+ Memory Hack' to help them never miss this concept again.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "user_ans": user_ans, "correct_ans": correct_ans})

# 4. MAIN APP INTERFACE
def main():
    st.title("🛡️ Security+ SY0-701 AI Exam Simulator")
    st.markdown("### *Guaranteed Mastery through AI Feedback*")

    vectorstore, llm = load_resources()
    if not vectorstore: return

    # Session State Management
    if 'q_index' not in st.session_state:
        # Load 50 questions from the vectorstore to start
        docs = vectorstore.similarity_search("CompTIA Security+ Exam Question", k=50)
        st.session_state.questions = docs
        st.session_state.q_index = 0
        st.session_state.score = 0
        st.session_state.feedback = None
        st.session_state.answered = False

    # Progress Sidebar
    with st.sidebar:
        st.header("Exam Analytics")
        total = len(st.session_state.questions)
        curr = st.session_state.q_index
        st.metric("Mastery Score", f"{(st.session_state.score / (curr if curr > 0 else 1)) * 100:.1f}%")
        st.progress(curr / total)
        if st.button("Reset & Refresh Questions"):
            st.session_state.clear()
            st.rerun()

    # Question UI
    if st.session_state.q_index < len(st.session_state.questions):
        doc = st.session_state.questions[st.session_state.q_index]
        content = doc.page_content
        
        st.info(f"**Question {st.session_state.q_index + 1}:**\n\n{content}")

        # Note: You'll need to parse your 'content' to find the actual answer key
        # For this logic, let's assume your text contains "Answer: X"
        correct_answer = "C" # Placeholder: Update with your logic to extract answer from doc.metadata or content

        options = ["A", "B", "C", "D"]
        choice = st.radio("Choose your answer:", options, index=None, key=f"q_{st.session_state.q_index}")

        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Submit Answer", use_container_width=True) and choice:
                st.session_state.answered = True
                if choice == correct_answer:
                    st.success("✅ Spot on! You understood that concept perfectly.")
                    st.session_state.score += 1
                else:
                    st.error(f"❌ Not quite. The correct answer was {correct_answer}.")
                    with st.spinner("Consulting the AI Tutor..."):
                        st.session_state.feedback = explain_mistake(llm, content, choice, correct_answer)

        with col2:
            if st.session_state.answered:
                if st.button("Next Question ➡️", use_container_width=True):
                    st.session_state.q_index += 1
                    st.session_state.answered = False
                    st.session_state.feedback = None
                    st.rerun()

        # Display AI Tutor Feedback
        if st.session_state.feedback:
            st.warning("### 🤖 AI Tutor Explanation")
            st.write(st.session_state.feedback)

    else:
        st.balloons()
        st.success(f"Exam Complete! Final Mastery Score: {st.session_state.score}/{len(st.session_state.questions)}")

if __name__ == "__main__":
    main()
