import streamlit as st
import random
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Load environment variables for local testing
load_dotenv()

st.set_page_config(page_title="Security+ SY0-701 Simulator", page_icon="🛡️", layout="centered")

# Initialize session state variables for tracking quiz progress
if "score" not in st.session_state:
    st.session_state.score = 0
if "questions_attempted" not in st.session_state:
    st.session_state.questions_attempted = 0
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "user_choice" not in st.session_state:
    st.session_state.user_choice = None

@st.cache_resource
def load_vectorstore():
    """Loads the pre-computed FAISS vector index."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def generate_question(vectorstore):
    """Retrieves context from the PDFs and generates a multiple-choice question."""
    terms = ["malware", "firewall", "cryptography", "social engineering", "cloud security", 
             "IAM", "risk management", "incident response", "PKI", "vulnerability", "zero trust"]
    
    # Select random context from vectorstore
    docs = vectorstore.similarity_search(random.choice(terms), k=1)
    context = docs[0].page_content

    # Initialize LLM using Streamlit Secrets
    llm = ChatGroq(
        temperature=0.7,
        groq_api_key=st.secrets.get("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"), 
        model_name="llama-3.1-8b-instant"
    )
    
    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="""You are a CompTIA Security+ SY0-701 expert exam writer. 
Based strictly on the following context from the official study material, generate a challenging multiple-choice question.
It must have exactly 4 options. Only one option should be correct.
Provide the correct answer letter and a detailed explanation of why the answer is correct and why the others are wrong.

Context: {context}

Return ONLY valid JSON with the following keys:
"question": the question text
"options": list of 4 strings (e.g., ["A. ...", "B. ...", "C. ...", "D. ..."])
"answer": the correct letter (e.g., "A")
"explanation": the detailed explanation

JSON:"""
    )
    
    try:
        response = llm.invoke(prompt_template.format(context=context))
        content = response.content.strip()
        
        # Clean up potential markdown formatting from LLM response
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        return json.loads(content)
    except Exception as e:
        st.error(f"Failed to generate question: {e}")
        return None

# Sidebar Statistics
st.sidebar.title("📊 Your Stats")
st.sidebar.write(f"**Score:** {st.session_state.score} / {st.session_state.questions_attempted}")
if st.session_state.questions_attempted > 0:
    accuracy = (st.session_state.score / st.session_state.questions_attempted) * 100
    st.sidebar.write(f"**Accuracy:** {accuracy:.1f}%")

if st.sidebar.button("Reset Quiz"):
    st.session_state.score = 0
    st.session_state.questions_attempted = 0
    st.session_state.current_question = None
    st.rerun()

# Main UI
st.title("🛡️ CompTIA Security+ SY0-701 Simulator")
st.write("Test your knowledge. Questions are dynamically generated from study materials.")

vectorstore = load_vectorstore()

if st.session_state.current_question is None:
    with st.spinner("Analyzing study materials & generating your next question..."):
        st.session_state.current_question = generate_question(vectorstore)
        st.session_state.feedback = None
        st.session_state.user_choice = None
        st.rerun()

q_data = st.session_state.current_question

if q_data:
    st.markdown(f"### {q_data['question']}")
    options = q_data['options']
    
    if st.session_state.feedback is None:
        user_choice = st.radio("Select your answer:", options, index=None)
        
        if st.button("Submit Answer"):
            if user_choice:
                st.session_state.user_choice = user_choice
                user_letter = user_choice.split(".")[0].strip()
                correct_letter = q_data['answer'].strip()
                
                st.session_state.questions_attempted += 1
                
                if user_letter == correct_letter:
                    st.session_state.score += 1
                    st.session_state.feedback = ("correct", q_data['explanation'])
                else:
                    st.session_state.feedback = ("incorrect", f"**Correct Answer: {correct_letter}**\n\n{q_data['explanation']}")
                st.rerun()
            else:
                st.warning("Please select an answer before submitting.")
    else:
        # Display results
        st.radio("Your answer:", options, index=options.index(st.session_state.user_choice), disabled=True)
        
        status, explanation = st.session_state.feedback
        if status == "correct":
            st.success("✅ Correct!")
        else:
            st.error("❌ Incorrect.")
        
        st.info(explanation)
            
        if st.button("Next Question"):
            st.session_state.current_question = None
            st.rerun()
