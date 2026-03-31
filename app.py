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
st.set_page_config(page_title="Security+ SY0-701 Free Tier", page_icon="🛡️", layout="wide")

# 2. RESOURCE LOADING
@st.cache_resource
def init_resources():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("🔑 API Key Missing in Secrets!")
        st.stop()

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Using the Free Tier's most stable model
        llm = ChatGroq(
            temperature=0.1, 
            model_name="llama3-70b-8192", 
            groq_api_key=api_key
        )
        return vectorstore, llm
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return None, None

# 3. AI CLEANING (Reduced for Free Tier Limits)
def format_question(llm, raw_text):
    # Only send the first 1500 characters to stay under Free Tier limits
    truncated_text = raw_text[:1500]
    
    template = """
    Extract the exam question from this text. 
    If it is a disclaimer/legal text, return: SKIP.
    
    TEXT: {raw_text}

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
    return chain.invoke({"raw_text": truncated_text})

# 4. MAIN APP
def main():
    st.title("🛡️ Security+ SY0-701 AI Exam Simulator")

    vectorstore, llm = init_resources()
    if not vectorstore: return

    if 'q_idx' not in st.session_state:
        # Search for questions - limit to 50 for Free Tier performance
        st.session_state.questions = vectorstore.similarity_search("practice question", k=50)
        st.session_state.q_idx = 0
        st.session_state.score = 0
        st.session_state.current_formatted = None

    if st.session_state.q_idx < len(st.session_state.questions):
        if st.session_state.current_formatted is None:
            raw_content = st.session_state.questions[st.session_state.q_idx].page_content
            
            with st.spinner("AI Tutor is analyzing (Free Tier Mode)..."):
                try:
                    # Small sleep to prevent 'Rate Limit' errors
                    time.sleep(1) 
                    formatted = format_question(llm, raw_content)
                    
                    if "SKIP" in formatted:
                        st.session_state.q_idx += 1
                        st.rerun()
                    st.session_state.current_formatted = formatted
                except Exception as e:
                    if "429" in str(e):
                        st.error("Groq Free Tier Rate Limit hit. Please wait 60 seconds.")
                    else:
                        st.error(f"Error: {e}")
                    st.stop()

        # Display Logic
        try:
            f = st.session_state.current_formatted
            q_text = f.split("QUESTION:")[1].split("A:")[0].strip()
            a_t = f.split("A:")[1].split("B:")[0].strip()
            b_t = f.split("B:")[1].split("C:")[0].strip()
            c_t = f.split("C:")[1].split("D:")[0].strip()
            d_t = f.split("D:")[1].split("CORRECT:")[0].strip()
            correct_letter = f.split("CORRECT:")[1].strip()

            st.subheader(f"Question {st.session_state.q_idx + 1}")
            st.info(q_text)
            
            opts = {f"A: {a_t}": "A", f"B: {b_t}": "B", f"C: {c_t}": "C", f"D: {d_t}": "D"}
            choice = st.radio("Choose answer:", list(opts.keys()), index=None)

            if st.button("Submit"):
                if opts[choice] == correct_letter:
                    st.success(f"Correct! The answer is {correct_letter}.")
                    st.session_state.score += 1
                else:
                    st.error(f"Wrong. The correct answer was {correct_letter}.")
                
                if st.button("Next Question"):
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
