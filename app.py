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
st.set_page_config(page_title="Security+ SY0-701 Pro Simulator", page_icon="🛡️", layout="wide")

# 2. RESOURCE INITIALIZATION
@st.cache_resource
def load_resources():
    # Priority: Streamlit Secrets, then local .env
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("🔑 GROQ_API_KEY missing in Secrets!")
        st.stop()

    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Ensure the 'faiss_index' folder is in your GitHub repo root
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        # Using Llama 3.3 for better reasoning on explanations
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        return vectorstore, llm
    except Exception as e:
        st.error(f"Load Error: {e}")
        return None, None

# 3. AI LOGIC: STRICT EXTRACTION
def get_clean_question(llm, raw_text):
    template = """
    You are a literal data extraction tool. Extract the Security+ question EXACTLY.
    If it is not a question or is a disclaimer, return: SKIP.

    TEXT: {raw_text}

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
    return chain.invoke({"raw_text": raw_text[:1200]})

# 4. AI LOGIC: TUTOR FEEDBACK
def get_tutor_feedback(llm, question, user_ans, correct_letter, is_right):
    template = """
    You are a Security+ (SY0-701) Tutor. 
    Question: {question}
    Student Choice: {user_ans}
    Correct Answer: {correct_letter}
    Result: {"Correct" if is_right else "Incorrect"}

    Task: Explain why the correct answer is the right choice and why the other options (especially the student's choice if wrong) are incorrect according to CompTIA standards.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "user_ans": user_ans, "correct_letter": correct_letter, "is_right": is_right})

# 5. MAIN APP
def main():
    st.title("🛡️ Security+ SY0-701 Pro Simulator")
    
    vectorstore, llm = load_resources()
    if not vectorstore: return

    # SESSION STATE INITIALIZATION
    if 'q_idx' not in st.session_state:
        # Pull exactly 90 docs for the exam
        st.session_state.all_docs = vectorstore.similarity_search("Security+ practice question", k=90)
        st.session_state.q_idx = 0
        st.session_state.correct_count = 0
        st.session_state.wrong_count = 0
        st.session_state.current_formatted = None
        st.session_state.feedback = None
        st.session_state.submitted = False

    # SIDEBAR SCOREBOARD & RESTART
    with st.sidebar:
        st.header("📊 Your Progress")
        st.write(f"Question: {st.session_state.q_idx + 1} / 90")
        st.success(f"✅ Correct: {st.session_state.correct_count}")
        st.error(f"❌ Wrong: {st.session_state.wrong_count}")
        
        if st.button("🔄 Restart Exam"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # QUESTION LOGIC
    if st.session_state.q_idx < len(st.session_state.all_docs):
        if st.session_state.current_formatted is None:
            raw_content = st.session_state.all_docs[st.session_state.q_idx].page_content
            with st.spinner("AI Tutor is analyzing study material..."):
                time.sleep(1.2) # Rate limit protection for Free Tier
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

            st.subheader(f"Question {st.session_state.q_idx + 1}")
            st.info(q_text)
            
            opts = {f"A: {a_opt}": "A", f"B: {b_opt}": "B", f"C: {c_opt}": "C", f"D: {d_opt}": "D"}
            
            # Radio is disabled once submitted to prevent changing answers
            user_choice = st.radio("Select choice:", list(opts.keys()), index=None, disabled=st.session_state.submitted)

            # BUTTONS
            if not st.session_state.submitted:
                if st.button("Submit Answer") and user_choice:
                    user_letter = opts[user_choice]
                    is_right = (user_letter == correct_letter)
                    
                    if is_right:
                        st.session_state.correct_count += 1
                        st.success("✅ Correct!")
                    else:
                        st.session_state.wrong_count += 1
                        st.error(f"❌ Incorrect. The correct answer was {correct_letter}.")
                    
                    # Get Explanation
                    with st.spinner("AI Tutor is generating your explanation..."):
                        st.session_state.feedback = get_tutor_feedback(llm, q_text, user_choice, correct_letter, is_right)
                    st.session_state.submitted = True
                    st.rerun()
            
            # Show Feedback and Next Button
            if st.session_state.submitted:
                st.markdown("---")
                st.warning("🤖 **AI Tutor Explanation:**")
                st.write(st.session_state.feedback)
                
                if st.button("Next Question ➡️"):
                    st.session_state.q_idx += 1
                    st.session_state.current_formatted = None
                    st.session_state.feedback = None
                    st.session_state.submitted = False
                    st.rerun()
        except:
            # Move to next doc if parsing fails
            st.session_state.q_idx += 1
            st.session_state.current_formatted = None
            st.rerun()
    else:
        st.balloons()
        st.success(f"Exam Session Complete! Final Score: {st.session_state.correct_count} / 90")

if __name__ == "__main__":
    main()
