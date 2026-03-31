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
st.set_page_config(page_title="Security+ SY0-701 AI Mastery", page_icon="🛡️", layout="wide")

# 2. SECURE RESOURCE INITIALIZATION
@st.cache_resource
def load_exam_resources():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("🔑 GROQ_API_KEY is missing! Please check your Streamlit Secrets.")
        st.stop()

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Free Tier Model
        llm = ChatGroq(
            temperature=0.1, 
            model_name="llama-3.3-70b-versatile", 
            groq_api_key=api_key
        )
        return vectorstore, llm
    except Exception as e:
        st.error(f"Critical Load Error: {e}")
        return None, None

# 3. AI LOGIC: QUESTION EXTRACTION
def get_clean_question(llm, raw_text):
    text_sample = raw_text[:1200]
    template = """
    You are a CompTIA Security+ Instructor. Extract the exam question from this text.
    If it's a disclaimer or copyright page, return: SKIP.

    TEXT: {raw_text}

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

# 4. AI LOGIC: TUTOR FEEDBACK (New Feature)
def get_tutor_feedback(llm, question, user_choice_text, correct_letter, is_correct):
    template = """
    You are an expert Security+ (SY0-701) Tutor.
    
    Question: {question}
    Student Answer: {user_choice_text}
    Result: {"Correct" if is_correct else "Incorrect"}
    Correct Letter: {correct_letter}
    
    Task: 
    1. Explain WHY the correct answer is technically the best choice according to CompTIA standards.
    2. If the student was wrong, explain the specific flaw in their choice.
    3. Provide a 'Pro-Tip' or 'Memory Trick' for this specific concept.
    
    Keep the explanation concise but highly educational.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": question, 
        "user_choice_text": user_choice_text, 
        "correct_letter": correct_letter,
        "is_correct": is_correct
    })

# 5. MAIN APP INTERFACE
def main():
    st.title("🛡️ Security+ SY0-701 AI Exam Simulator")
    st.markdown("---")

    vectorstore, llm = load_exam_resources()
    if not vectorstore: return

    # Session State Initialization
    if 'all_docs' not in st.session_state:
        st.session_state.all_docs = vectorstore.similarity_search("practice question", k=60)
        st.session_state.q_idx = 0
        st.session_state.score = 0
        st.session_state.current_formatted = None
        st.session_state.feedback = None # Stores the AI explanation

    if st.session_state.q_idx < len(st.session_state.all_docs):
        if st.session_state.current_formatted is None:
            raw_content = st.session_state.all_docs[st.session_state.q_idx].page_content
            with st.spinner("AI Tutor is analyzing study material..."):
                try:
                    time.sleep(1.5) 
                    formatted = get_clean_question(llm, raw_content)
                    if "SKIP" in formatted:
                        st.session_state.q_idx += 1
                        st.rerun()
                    st.session_state.current_formatted = formatted
                except Exception as e:
                    st.error(f"AI Service Error: {e}")
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
            
            options_dict = {f"A: {a_opt}": "A", f"B: {b_opt}": "B", f"C: {c_opt}": "C", f"D: {d_opt}": "D"}
            user_choice = st.radio("Select the correct option:", list(options_dict.keys()), index=None)

            if st.button("Submit Answer") and user_choice:
                user_letter = options_dict[user_choice]
                is_right = (user_letter == correct_letter)
                
                if is_right:
                    st.success(f"✅ Correct! The answer is {correct_letter}.")
                    st.session_state.score += 1
                else:
                    st.error(f"❌ Incorrect. The answer was {correct_letter}.")
                
                # Fetch AI explanation
                with st.spinner("AI Tutor is generating your explanation..."):
                    st.session_state.feedback = get_tutor_feedback(llm, q_text, user_choice, correct_letter, is_right)

            # Display the Explanation if it exists
            if st.session_state.feedback:
                st.markdown("---")
                st.warning("🤖 **AI Tutor Explanation:**")
                st.write(st.session_state.feedback)
                
                if st.button("Next Question ➡️"):
                    st.session_state.q_idx += 1
                    st.session_state.current_formatted = None
                    st.session_state.feedback = None
                    st.rerun()
        except:
            st.session_state.q_idx += 1
            st.session_state.current_formatted = None
            st.rerun()
    else:
        st.balloons()
        st.success(f"Final Score: {st.session_state.score}/{len(st.session_state.all_docs)}")

if __name__ == "__main__":
    main()
