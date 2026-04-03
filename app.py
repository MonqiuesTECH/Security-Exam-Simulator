import streamlit as st
import os
import time
import random
import urllib.parse
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. PAGE CONFIGURATION
load_dotenv()
st.set_page_config(page_title="Security+ SY0-701 Adaptive Sim", page_icon="🛡️", layout="wide")

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
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
        return vectorstore, llm
    except Exception as e:
        st.error(f"Critical Initialization Error: {e}")
        return None, None

# 3. AI LOGIC: INVISIBLE ADAPTIVE PARSER
def get_adaptive_question(llm, raw_text, difficulty):
    template = """
    Extract ONE CompTIA Security+ question EXACTLY as written. 
    DIFFICULTY TARGET: {difficulty}
    If the text is a disclaimer or doesn't match, return: SKIP.
    RAW TEXT: {raw_text}
    FORMAT:
    QUESTION: [text]
    A: [text]
    B: [text]
    C: [text]
    D: [text]
    CORRECT: [A, B, C, or D]
    CRITICAL INSTRUCTION: Stop generating text immediately after providing the single CORRECT letter.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"raw_text": raw_text[:1200], "difficulty": difficulty})

# 4. AI LOGIC: ADAPTIVE TEACHING TUTOR
def get_tutor_feedback(llm, question, user_ans, correct_letter, is_right, difficulty):
    result_text = "Correct" if is_right else "Incorrect"
    template = """
    You are an expert CompTIA Security+ Tutor. 
    Question: {question}
    Student Choice: {user_ans}
    Correct Answer: {correct_letter}
    Result: {result}
    Student's Current Hidden Level: {difficulty}
    
    Task: Write a smooth explanation.
    1. EASY Level: Use a simple, real-world analogy. Focus on building confidence.
    2. HARD Level: Go deeper into the technical "why" and provide an advanced pro-tip.
    3. Explicitly state why the correct answer is the BEST choice and the flaw in their selection.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "user_ans": user_ans, "correct_letter": correct_letter, "result": result_text, "difficulty": difficulty})

# 5. AI LOGIC: TOPIC EXTRACTOR & PBQ GRADER
def get_video_topic(llm, question):
    template = "Analyze this Security+ question and extract the ONE core concept in 2 to 4 words. Question: {question}\nOutput ONLY the short topic name."
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

def grade_pbq(llm, pbq_title, pbq_scenario, user_answers):
    template = """
    You are a friendly, encouraging CompTIA PBQ Grader.
    PBQ Title: {title}
    Scenario: {scenario}
    Student's Submitted Answers: {answers}
    
    Task:
    1. Tell the student if they got the configuration completely right, partially right, or wrong.
    2. Go through their answers and explain the "Why" using simple terms.
    3. Provide the definitive correct configuration at the end.
    
    CRITICAL INSTRUCTION:
    If the student got EVERY single part 100% correct, you MUST put the exact phrase "[PASSED]" at the very end of your response. 
    If they got ANYTHING wrong (even partially), you MUST put the exact phrase "[FAILED]" at the very end of your response.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"title": pbq_title, "scenario": pbq_scenario, "answers": str(user_answers)})

# 6. PBQ DATABASE (Guided with Word Banks & Video Topics)
PBQ_DB = {
    1: {"topic": "Network Ports", "title": "Port Configuration", "desc": "Match the correct default port number to the network service.", "type": "match", "keys": ["SSH", "HTTPS", "RDP", "DNS"], "options": ["22", "23", "53", "80", "443", "3389"]},
    2: {"topic": "Multifactor Authentication", "title": "Authentication Factors", "desc": "Categorize the MFA security factors.", "type": "match", "keys": ["Password", "Smartcard", "Retina Scan", "PIN"], "options": ["Something you know", "Something you have", "Something you are", "Somewhere you are"]},
    3: {"topic": "Malware Types", "title": "Malware Identification", "desc": "Match the malicious behavior to the
