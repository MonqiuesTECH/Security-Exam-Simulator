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

   Gemini said
I see exactly what happened! This is a classic coding "gotcha."

If you look closely at your GitHub file list, the image you uploaded is named logo.jpeg (with an "e"), but in the code, we told it to look for logo.jpg (without the "e").

Computers are incredibly literal, so os.path.exists() saw .jpg and said, "Nope, doesn't exist here!" and safely skipped it.

How to fix the image:
In your app.py file, go to Line 90 (inside the render_login_header function) and change the image_name to exactly match your GitHub file:

Change this:

Python
def render_login_header():
    """Safely loads the image if it exists, prevents crashes if it doesn't."""
    image_name = "WhatsApp Image 2026-02-07 at 13.58.20.jpg" # Or whatever you had here
To this:

Python
def render_login_header():
    """Safely loads the image if it exists, prevents crashes if it doesn't."""
    image_name = "logo.jpeg"  # Notice the .jpeg extension!
Save that change to your app.py in GitHub, and your image will instantly appear above the login box!

Your Login Credentials
Here are the usernames and passwords we set up for your system. (Make sure these exact values are saved in your .streamlit/secrets.toml file):

👨🏾‍🏫 The Instructor / Admin

Username: admin

Password: SecPlusMaster2026!

Access: Full dashboard, sees all student hours/scores, can delete user data.

🧑🏾‍🎓 Student Profile 1

Username: guest1

Password: StudyBuddy01

Access: Takes the exam, practices PBQs, time is tracked silently.

🧑🏾‍🎓 Student Profile 2

Username: guest2

Password: StudyBuddy02

Access: Same as Guest 1, tracked entirely separately.
