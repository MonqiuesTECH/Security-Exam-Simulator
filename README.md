Here is the complete, ready-to-use `README.md` file. I have clearly outlined Steps 1 through 5 so you and your team can simply copy, paste, and follow along without any confusion. 

Just copy everything inside the block below and paste it directly into your GitHub `README.md` file!

```markdown
# 🛡️ Cyber Punk University

**Created and Powered By Monique Bruce**

Cyber Punk University is an AI-driven Learning Management System (LMS) and Adaptive Exam Simulator designed specifically for the CompTIA Security+ (SY0-701) certification. Built with Python, Streamlit, and LangChain, it offers an intelligent, adaptive learning environment for students and a comprehensive analytics dashboard for instructors.

---

## ✨ Core Features

### 🧑🏾‍🎓 For Students (Guest Profiles)
* **Adaptive AI Simulator:** The system dynamically scales difficulty (EASY, NORMAL, HARD) based on your current answer streak.
* **Performance-Based Questions (PBQs):** Interactive, hands-on labs (drag-and-drop matching, log analysis, firewall configuration) graded instantly by an AI Tutor.
* **Auto-Save & Silent Tracking:** Study time, live scores, and weak topics are tracked silently in the background. No manual "saving" required.
* **Study Timeouts:** If a student misses 3 questions in a row, the AI triggers a "Study Timeout," identifies the knowledge gap, and provides targeted review material (Professor Messer video links).
* **Cross-Platform UI:** Fully responsive design that works flawlessly on desktop, Android, and iOS (includes specific logic to prevent iPhone auto-capitalization login bugs).

### 👨🏾‍🏫 For Instructors (Admin Profile)
* **Live Telemetry:** Monitor student engagement time, live quiz scores, and recent activity logs in real-time.
* **Knowledge Gap Analysis:** The AI automatically flags weak topics for each student based on failed PBQs or multiple-choice misses.
* **Data Management:** Instantly purge a student's history and reset their metrics directly from the UI.

---

## 🛠️ Tech Stack & Architecture
* **Frontend:** Streamlit
* **LLM Engine:** Groq (`llama-3.3-70b-versatile`)
* **Vector Database:** FAISS (Local storage for exam questions)
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Data Storage:** Local JSON (`study_logs.json`) for seamless LMS state tracking.

---

## 🚀 Quick Start Guide (Steps 1 - 5)

Follow these 5 exact steps to get the application running locally or deploy it to the cloud.

### Step 1: Clone the Repository
Pull the code down to your local machine:
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/security-exam-simulator.git](https://github.com/YOUR_GITHUB_USERNAME/security-exam-simulator.git)
cd security-exam-simulator
```

### Step 2: Set Up a Virtual Environment
Create an isolated environment and install the required dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Configure Your Secrets 
Streamlit needs a secrets file to store passwords and API keys securely. 
1. Create a folder named `.streamlit` in the root directory.
2. Inside that folder, create a file named `secrets.toml`.
3. Paste the following configuration into `secrets.toml`:

```toml
# .streamlit/secrets.toml

# AI API Key (Required for the tutor to work)
GROQ_API_KEY = "your_groq_api_key_here"

# System Credentials
[passwords]
admin = "SecPlusMaster2026!"
guest1 = "StudyBuddy01"
guest2 = "StudyBuddy02"
```
*(Note: Ensure `.streamlit/secrets.toml` is added to your `.gitignore` file so you do not accidentally upload your passwords to GitHub!)*

### Step 4: Ensure Required Assets Are Uploaded
For the application to render perfectly, ensure these two assets are in your root folder:
1. `logo.jpeg` (The main image displayed on the login screen).
2. `faiss_index/` (The folder containing your vectorized Security+ exam questions).

### Step 5: Run the Application
Launch the simulator locally:
```bash
streamlit run app.py
```

---

## 🤝 How to Contribute & Build on This

If you are a developer joining the team, here is how you can extend the platform:

1. **Adding New PBQs:**
   Open `app.py` and locate the `PBQ_DB` dictionary. You can easily add new PBQ scenarios by following the existing JSON structure. The AI Grader will automatically adapt to new questions without requiring prompt rewrites.
2. **Updating the Vector Store:**
   If you want to add new multiple-choice questions, you must update the `faiss_index` using your ingestion script. The main app queries this local vector store to feed raw text to the Groq LLM.
3. **Database Upgrades:**
   Currently, the app uses a lightweight `study_logs.json` file for LMS tracking. For large-scale production (100+ concurrent users), consider replacing the `load_db()` and `save_db()` functions in `app.py` with API calls to a robust database like Firebase, Supabase, or PostgreSQL.

---
*Created and Powered By Monique Bruce*
```
