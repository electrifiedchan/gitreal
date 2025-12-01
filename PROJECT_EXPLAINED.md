# 🕵️‍♂️ GitReal: The Code Detective
> *A simple guide to understanding this project, written for everyone!*

## 👋 What is this?
Imagine you tell your teacher, "I built a robot!" but you actually just bought a toy robot from the store. **GitReal** is like a super-smart detective that checks if you *really* built the robot.

In the programming world:
1.  People write **Resumes** (saying "I can code this!").
2.  People have **Code** on GitHub (the actual work).
3.  **GitReal** looks at both and says: *"Hmm, did you really write this? Or are you faking it?"*

---

## 🏗️ How It Works (The 3 Steps)

### Step 1: The Resume (The Story)
First, the user uploads their resume (a PDF file). The computer reads it to find out what projects they claim to have built.

### Step 2: The Code (The Proof)
Then, the user gives a link to their **GitHub** (where programmers store their code). GitReal sends a little "scout" to read the actual code files.

### Step 3: The Roast (The Verdict)
This is the fun part! An **AI Brain** (using Google Gemini) reads the resume AND the code. It compares them.
-   If you said you are an "Expert" but your code is messy -> **It roasts you!** 🔥
-   If your code is amazing -> **It congratulates you!** 🎉

---

## 🧩 The Building Blocks (How we built it)

This project has two main parts, like a restaurant:
1.  **The Frontend (The Dining Room):** What the user sees.
2.  **The Backend (The Kitchen):** Where the cooking happens.

### 1. The Frontend (The Face) 🖥️
*   **Where is it?** `frontend/` folder
*   **Language:** TypeScript / Next.js
*   **Style:** "The Matrix" (Green text, black background, falling code).

It looks like this:
```tsx
// This is how we make the text look like The Matrix!
<div className="matrix-text">
  <h1>Wake up, Neo...</h1>
  <p>Upload your resume to see the truth.</p>
</div>
```

### 2. The Backend (The Brain) 🧠
*   **Where is it?** `backend/` folder
*   **Language:** Python
*   **Main Tools:** FastAPI (web server) and Google Gemini (AI).

Here is the "Brain" code simplified:

```python
# backend/brain.py (Simplified)

def judge_programmer(resume, code):
    # We ask the AI to compare them
    prompt = f"""
    Look at this resume: {resume}
    Look at this code: {code}
    
    Does the code match the resume?
    If not, make a joke about it!
    """
    
    verdict = ai.ask(prompt)
    return verdict
```

---

## 🚀 How to Run It (Be the Boss)

Want to start the detective agency? Here is how you run the code on your computer.

### Terminal 1: The Kitchen (Backend)
This starts the Python server that does the thinking.
```bash
cd backend
# Create a virtual environment (a safe box for our tools)
python -m venv venv
# Turn it on
venv\Scripts\activate
# Install tools
pip install -r requirements.txt
# Start the server!
python main.py
```

### Terminal 2: The Dining Room (Frontend)
This starts the website so you can see it.
```bash
cd frontend
# Install the furniture (dependencies)
pnpm install
# Open the restaurant!
pnpm dev
```

Now open your browser to `http://localhost:3000` and watch the magic happen! 🐇

---

## 🎤 Bonus: It Can Hear You!
We also added a **Voice Mode**. You can talk to the AI (who acts like Morpheus from The Matrix), and it talks back!

```python
# backend/main.py

@app.post("/listen")
def listen_to_voice(audio_file):
    # Turn sound into text
    text = deepgram.transcribe(audio_file)
    return text
```

---

> **Summary:** GitReal is a fun, strict teacher that checks your homework using AI. It uses **Python** for thinking and **Next.js** for showing the cool Matrix interface.
