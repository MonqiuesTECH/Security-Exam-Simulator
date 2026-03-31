import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. PAGE SETUP
load_dotenv()
st.set_page_config(page_title="Security+ SY0-701 AI Mastery", page_icon="🛡️", layout="wide")

# 2. SECURE RESOURCE INITIALIZATION
@st.cache_resource
def init_app_resources():
    # Priority 1: Streamlit Secrets | Priority 2: Local .env
    api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.error("GROQ_API_KEY is missing. Add it to Streamlit Secrets to launch.")
        st.stop()

    try:
        # Load the Embeddings (Requires sentence-transformers & torch)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load FAISS Vector Store
        vectorstore = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Initialize Groq LLM
        llm = ChatGroq(
            temperature=0.1, 
            model_name="llama-3.1-70b-versatile", 
            groq_api_key=api_key
        )
        
        return vectorstore, llm
    except Exception as e:
        st.error(f"Resource Load Error: {e}")
        return None, None

# 3. AI LOGIC: CLEANING & FORMATTING
def get_formatted_question(llm, raw_text):
    template = """
    You are a CompTIA Security+ Expert. Analyze the text below from a study guide.
    
    CRITICAL RULES:
    1. If the text is a 'Legal Disclaimer', 'Front Matter', or 'Copyright' page, return: SKIP
    2. If it is a question, rewrite it clearly.
    3. Extract the full text for options A, B, C, and D.
    4. Identify the correct answer letter.

    RAW TEXT: {raw_text}

    RESPONSE FORMAT:
    QUESTION: [Question Text]
    A: [Option A Text]
    B: [Option B Text]
    C: [Option C Text]
    D: [Option D Text]
    CORRECT: [Just the Letter]
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"raw_text": raw_text})

# 4. AI LOGIC: TUTOR FEEDBACK
def get_tutor_explanation(llm, question, user_choice, correct_choice):
    template = """
    A student missed a Security+ question.
    Question: {question}
    Student Choice: {user_choice}
    Correct Choice: {correct_choice}
    
    Task: Explain why the correct choice is right and specifically why the student's logic was wrong.
    End with a 'Pro-Tip' for the SY0-701 exam.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": question, 
        "user_choice": user_choice, 
        "correct_choice": correct_choice
    })

# 5. MAIN UI LOGIC
def main():
    st.title("🛡️ Security+ SY0-701 AI Exam Simulator")
    st.markdown("---")

    vectorstore, llm = init_app_resources()
    if not vectorstore: return

    # Session State Setup
    if 'q_idx' not in st.session_state:
        # Pull 100 chunks to ensure we have enough content after skipping disclaimers
        st.session_state.questions = vectorstore.similarity_search("Security+ exam practice", k=100)
        st.session_state.q_idx = 0
        st.session_state.score = 0
        st.session_state.current_formatted = None
        st.session_state.feedback = None

    # App Progression
    if st.session_state.q_idx < len(st.session_state.questions):
        # AI Formatting Step
        if st.session_state.current_formatted is None:
            raw_content = st.session_state.questions[st.session_state.q_idx].page_content
            with st.spinner("AI Tutor is preparing the next question..."):
                formatted = get_formatted_question(llm, raw_content)
                
            if "SKIP" in formatted:
                st.session_state.q_idx += 1
                st.rerun()
            st.session_state.current_formatted = formatted

        # Data Parsing
        try:
            f = st.session_state.current_formatted
            q_text = f.split("QUESTION:")[1].split("A:")[0].strip()
            a_text = f.split("A:")[1].split("B:")[0].strip()
            b_text = f.split("B:")[1].split("C:")[0].strip()
            c_text = f.split("C:")[1].split("D:")[0].strip()
            d_text = f.split("D:")[1].split("CORRECT:")[0].strip()
            correct_letter = f.split("CORRECT:")[1].strip()

            st.subheader(f"Question {st.session_state.q_idx + 1}")
            st.info(q_text)
            
            # Map full text to letters
            options_map = {
                f"A: {a_text}": "A",
                f"B: {b_text}": "B",
                f"C: {c_text}": "C",
                f"D: {d_text}": "D"
            }
            
            user_selection = st.radio("Select the correct answer:", list(options_map.keys()), index=None)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Submit Answer") and user_selection:
                    letter = options_map[user_selection]
                    if letter == correct_letter:
                        st.success(f"✅ Correct! The answer is {letter}.")
                        st.session_state.score += 1
                    else:
                        st.error(f"❌ Incorrect. The correct answer was {correct_letter}.")
                        st.session_state.feedback = get_tutor_explanation(llm, q_text, letter, correct_letter)

            # AI Feedback display
            if st.session_state.feedback:
                st.warning("🤖 **AI Tutor Explanation:**")
                st.write(st.session_state.feedback)

            with col2:
                if st.button("Next Question ➡️"):
                    st.session_state.q_idx += 1
                    st.session_state.current_formatted = None
                    st.session_state.feedback = None
                    st.rerun()

        except Exception:
            # Skip if parsing fails
            st.session_state.q_idx += 1
            st.session_state.current_formatted = None
            st.rerun()
    else:
        st.balloons()
        st.success(f"Exam Complete! Final Score: {st.session_state.score}/{len(st.session_state.questions)}")

if __name__ == "__main__":
    main()
