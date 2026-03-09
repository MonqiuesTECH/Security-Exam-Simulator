import streamlit as st
import os
import json
import random
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Load environment variables
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
    embeddings = OpenAIEmbeddings()
    # allow_dangerous_deserialization is required for local FAISS loading in newer LangChain versions
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def generate_question(vectorstore):
    """Retrieves context from the PDFs and generates a multiple-choice question."""
    # List of broad SY0-701 domains to ensure varied questions
    terms = ["malware", "firewall", "cryptography", "social engineering", "cloud security", 
             "IAM", "risk management", "incident response", "PKI", "vulnerability", "zero trust"]
    
    # Retrieve a random chunk of context
    docs = vectorstore.similarity_search(random.choice(terms), k=1)
    context = docs[0].page_content

    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
    prompt = PromptTemplate(
        input_variables=["context"],
        template="""You are a CompTIA Security+ SY0-701 expert exam writer. 
Based strictly on the following context from the official study material, generate a challenging multiple-choice question.
It must have exactly 4 options. Only one option should be correct.
Provide the correct answer letter and a detailed explanation of why the answer is correct and why the others are wrong to help the student study.

Context: {context}

Return ONLY valid JSON with the following keys:
"question": the question text
"options": list of 4 strings (e.g., ["A. ...", "B. ...", "C. ...", "D. ..."])
"answer": the correct letter (e.g., "A")
"explanation": the detailed explanation

JSON:"""
    )
    
    response = llm.invoke(prompt.format(context=context))
    
    # Clean up the response in case the LLM wrapped it in markdown code blocks
    content = response.content.strip()
    if content.startswith("```json"):
        content = content[7:-3]
    elif content.startswith("```"):
        content = content[3:-3]
        
    try:
        q_data = json.loads(content)
        return q_data
    except Exception as e:
        st.error("Failed to parse the question. Retrying...")
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
st.write("Test your knowledge. Questions are dynamically generated from your study materials.")

vectorstore = load_vectorstore()

# Generate a new question if we don't have one queued up
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
    
    # State 1: User needs to answer the question
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
                
    # State 2: Answer submitted, show feedback and explanation
    else:
        st.radio("Your answer:", options, index=options.index(st.session_state.user_choice), disabled=True)
        
        status, explanation = st.session_state.feedback
        if status == "correct":
            st.success("✅ Correct!")
            st.info(explanation)
        else:
            st.error("❌ Incorrect.")
            st.info(explanation)
            
        if st.button("Next Question"):
            st.session_state.current_question = None
            st.rerun()
