import streamlit as st
import os
import time
import random
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. PAGE CONFIGURATION
load_dotenv()
st.set_page_config(page_title="Security+ SY0-701 Pro Sim", page_icon="🛡️", layout="wide")

# 2. SECURE RESOURCE LOADING
@st.cache_resource
def load_resources():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("🔑 GROQ_API_KEY is missing in Streamlit Secrets!")
        st.stop()
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        # Using Llama 3.3 for high-quality, readable explanations
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        return vectorstore, llm
    except Exception as e:
        st.error(f"Critical Initialization Error: {e}")
        return None, None

# 3. AI LOGIC: STRICT QUESTION PARSER
def get_clean_question(llm, raw_text):
    template = """
    Extract ONE CompTIA Security+ question EXACTLY as written. 
    If this text is a disclaimer, table of contents, or NOT a question, return: SKIP.
    
    RAW TEXT: {raw_text}

    FORMAT:
    QUESTION: [text]
    A: [text]
    B: [text]
    C: [text]
    D: [text]
    CORRECT: [A, B, C, or D]
    
    CRITICAL INSTRUCTION: Stop generating text immediately after providing the single CORRECT letter. Do not extract the next question.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"raw_text": raw_text[:1200]})

# 4. AI LOGIC: NON-TECHNICAL EXPLANATION GENERATOR
def get_tutor_feedback(llm, question, user_ans, correct_letter, is_right):
    result_text = "Correct" if is_right else "Incorrect"
    
    template = """
    You are a friendly, expert CompTIA Security+ (SY0-701) Tutor. 
    Question: {question}
    Student Choice: {user_ans}
    Correct Answer: {correct_letter}
    Result: {result}
    
    Task: Write a smooth, highly readable explanation tailored for a beginner.
    1. Explain the core concept using a simple, real-world analogy so a non-technical person can understand.
    2. Clearly state why the correct answer is the BEST choice.
    3. If the student was incorrect, gently explain the flaw in their selection without using overly dense jargon.
    4. Use formatting (like bolding key terms) to make it easy to read.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "question": question, 
        "user_ans": user_ans, 
        "correct_letter": correct_letter, 
        "result": result_text
    })

# 5. MAIN APPLICATION
def main():
    st.title("🛡️ Security+ SY0-701 Pro Simulator")
    vectorstore, llm = load_resources()
    if not vectorstore: return

    # --- STATE INITIALIZATION WITH SHUFFLE ---
    if 'display_idx' not in st.session_state:
        # 1. Fetch the documents
        docs = vectorstore.similarity_search("Security+ practice question", k=150)
        
        # 2. SHUFFLE the documents so the order is different every time!
        random.shuffle(docs)
        
        st.session_state.all_docs = docs
        st.session_state.db_idx = 0         
        st.session_state.display_idx = 1    
        st.session_state.correct_count = 0  
        st.session_state.wrong_count = 0    
        
        # State Machine Variables
        st.session_state.current_q = None   
        st.session_state.phase = "answering" 
        st.session_state.feedback = ""
        st.session_state.user_choice = None
        st.session_state.is_right = False

    # --- SIDEBAR SCOREBOARD ---
    with st.sidebar:
        st.header("📊 Exam Progress")
        st.write(f"**Question:** {st.session_state.display_idx} / 90")
        st.success(f"✅ Correct: {st.session_state.correct_count}")
        st.error(f"❌ Wrong: {st.session_state.wrong_count}")
        
        st.markdown("---")
        if st.button("🔄 Restart Exam"):
            # This wipes the memory, causing it to pull a new list and shuffle again
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # --- EXAM COMPLETION CHECK ---
    if st.session_state.display_idx > 90 or st.session_state.db_idx >= len(st.session_state.all_docs):
        st.balloons()
        st.success(f"🎉 Exam Finished! Final Score: {st.session_state.correct_count} / 90")
        return

    # --- QUESTION FETCHING LOGIC ---
    if st.session_state.current_q is None:
        with st.spinner("AI Tutor is finding your next question..."):
            while st.session_state.db_idx < len(st.session_state.all_docs):
                time.sleep(1) # Protect free tier limits
                raw_content = st.session_state.all_docs[st.session_state.db_idx].page_content
                formatted = get_clean_question(llm, raw_content)
                
                if "SKIP" not in formatted and "QUESTION:" in formatted:
                    try:
                        q_text = formatted.split("QUESTION:")[1].split("A:")[0].strip()
                        a_opt = formatted.split("A:")[1].split("B:")[0].strip()
                        b_opt = formatted.split("B:")[1].split("C:")[0].strip()
                        c_opt = formatted.split("C:")[1].split("D:")[0].strip()
                        d_opt = formatted.split("D:")[1].split("CORRECT:")[0].strip()
                        
                        # Isolate ONLY the first letter of the correct answer string
                        raw_correct = formatted.split("CORRECT:")[1].strip()
                        correct_letter = raw_correct[0].upper()
                        
                        if correct_letter not in ["A", "B", "C", "D"]:
                            raise ValueError("Invalid letter extracted")
                        
                        st.session_state.current_q = {
                            "text": q_text,
                            "options": [f"A: {a_opt}", f"B: {b_opt}", f"C: {c_opt}", f"D: {d_opt}"],
                            "correct_letter": correct_letter
                        }
                        break # Exit loop successfully
                    except Exception:
                        st.session_state.db_idx += 1 # Move to next doc if parsing fails
                else:
                    st.session_state.db_idx += 1 # Move to next doc if it's a disclaimer

    # If database is exhausted
    if st.session_state.current_q is None:
        st.error("Ran out of study material in the database.")
        return

    # --- UI RENDERING ---
    cq = st.session_state.current_q
    
    st.subheader(f"Question {st.session_state.display_idx}")
    st.info(cq["text"])

    # PHASE 1: ANSWERING
    if st.session_state.phase == "answering":
        selected_option = st.radio("Select your answer:", cq["options"], index=None)

        if st.button("Submit Answer"):
            if selected_option is None:
                st.warning("⚠️ Please select an answer before submitting.")
            else:
                user_letter = selected_option.split(":")[0].strip()
                is_right = (user_letter == cq["correct_letter"])
                
                if is_right:
                    st.session_state.correct_count += 1
                else:
                    st.session_state.wrong_count += 1
                
                st.session_state.user_choice = selected_option
                st.session_state.is_right = is_right
                
                with st.spinner("AI Tutor is writing your explanation..."):
                    st.session_state.feedback = get_tutor_feedback(
                        llm, cq["text"], selected_option, cq["correct_letter"], is_right
                    )
                
                # Switch to Review phase
                st.session_state.phase = "reviewing"
                st.rerun()

    # PHASE 2: REVIEWING
    elif st.session_state.phase == "reviewing":
        # Show what user selected, disabled
        st.radio("Your answer:", cq["options"], index=cq["options"].index(st.session_state.user_choice), disabled=True)
        
        st.markdown("---")
        # Visual Banner
        if st.session_state.is_right:
            st.success(f"✅ Correct! The answer is {cq['correct_letter']}.")
        else:
            st.error(f"❌ Incorrect. The correct answer was {cq['correct_letter']}.")
        
        st.warning("🤖 **AI Tutor Explanation:**")
        st.write(st.session_state.feedback)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # THE ONLY WAY OUT
        if st.button("Next Question ➡️"):
            st.session_state.db_idx += 1        
            st.session_state.display_idx += 1   
            st.session_state.current_q = None   
            st.session_state.phase = "answering"
            st.rerun()

if __name__ == "__main__":
    main()
