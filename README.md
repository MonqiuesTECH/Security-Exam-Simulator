# 🛡️ Security+ SY0-701 Pro Simulator

An AI-powered, interactive exam simulator designed to help candidates study for the CompTIA Security+ (SY0-701) certification. Built with Python, Streamlit, and LangChain, this application uses Retrieval-Augmented Generation (RAG) and Llama 3.3 to act as a real-time, personalized tutor.

## ✨ Features
* **AI Tutor Explanations:** Doesn't just tell you if you are right or wrong; generates real-time, highly readable explanations using Llama 3.3 to explain *why* an answer is correct and why the others are flawed.
* **Strict Phase Rendering:** Custom Streamlit state-machine logic ensures the UI strictly locks between "Answering" and "Reviewing" phases, preventing accidental skips and ensuring explanations persist on screen.
* **Dynamic Shuffling:** Automatically scrambles a pool of 150+ database queries on startup so you never memorize the question order.
* **Free-Tier Optimized:** Includes built-in API rate-limit protections (`time.sleep` buffers) and strict token-truncation to run flawlessly on Groq's free tier.
* **Live Scoreboard:** Tracks progress, correct answers, and incorrect answers in real-time across a strict 90-question exam format.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **AI/LLM:** Groq API (Llama-3.3-70b-versatile)
* **Orchestration:** LangChain
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (Local)

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/security-exam-simulator.git](https://github.com/yourusername/security-exam-simulator.git)
   cd security-exam-simulator
