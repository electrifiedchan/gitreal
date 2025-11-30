# GitReal - AI-Powered Resume Verification System

<div align="center">

![GitReal Banner](https://img.shields.io/badge/GitReal-Matrix%20Themed-00FF41?style=for-the-badge&logo=matrix&logoColor=white)

**"Welcome to the Real World"**

🔴 *Take the Red Pill* - Get brutally roasted on your resume vs actual code
🔵 *Take the Blue Pill* - Rewrite your resume with AI assistance

[![Next.js](https://img.shields.io/badge/Next.js-15-black?style=flat-square&logo=next.js)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Python-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?style=flat-square&logo=google)](https://deepmind.google/technologies/gemini/)
[![Deepgram](https://img.shields.io/badge/Deepgram-Voice%20AI-13EF93?style=flat-square)](https://deepgram.com/)

</div>

---

## 🎯 Overview

GitReal is an AI-powered resume verification tool that cross-references your resume claims with your actual GitHub repositories. Built for a hackathon, it provides:

- **🔥 Brutal Resume Analysis** - AI-powered critique comparing resume vs. actual code
- **🎤 Voice Interview Mode** - Real-time voice conversations powered by Deepgram
- **📝 ATS Resume Compiler** - Generate optimized resumes with code evidence
- **🎯 Project-Specific Analysis** - Choose which project to analyze

## ✨ Features

### 🔴 Red Pill - "Roast Me" Mode
- Upload your resume (PDF)
- **Gemini OCR extracts all projects** from your resume
- **Choose which project** to get roasted on
- AI fetches and analyzes your actual GitHub code
- Three-part verdict:
  - **Real World Critique** - Architectural and code quality issues
  - **False Claims** - Discrepancies between resume and code
  - **Resume Additions** - Skills you should highlight

### 🔵 Blue Pill - "Rewrite Me" Mode
- Same project selection flow
- AI-assisted resume improvement
- Interactive chat to refine content
- Generate ATS-optimized bullet points

### 🎤 Voice Interview Mode
- **Real-time voice conversations** with AI interviewer
- Powered by **Deepgram** (STT + TTS)
- **Gemini 2.5 Pro** as the AI brain
- Push-to-talk interface
- Tests your actual understanding of your code

### 📄 ATS Resume Compiler
- Generates ATS-optimized content
- Injects code evidence into bullet points
- Project-specific optimization
- Professional, action-oriented tone

## 🚀 Tech Stack

### Frontend
- **Next.js 15** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Matrix-themed styling
- **Axios** - HTTP client
- **Lucide React** - Icons
- **MediaRecorder API** - Voice recording

### Backend
- **FastAPI** - Python web framework
- **Google Gemini 2.5 Flash** - AI model for analysis
- **Deepgram** - Speech-to-Text & Text-to-Speech
- **PyPDF** - PDF parsing
- **GitHub API** - Repository analysis
- **Python 3.11+**

### AI Stack
- **Gemini 2.5 Flash** - Resume analysis, OCR, chat
- **Deepgram Nova-2** - Speech recognition
- **Deepgram Aura** - Voice synthesis

## 📦 Installation

### Prerequisites
- Node.js 18+ and npm
- Python 3.11+
- Google Gemini API key
- GitHub Personal Access Token (optional, for private repos)

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file:**
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   DEEPGRAM_API_KEY=your_deepgram_api_key_here
   GITHUB_TOKEN=your_github_token_here  # Optional
   ```

5. **Run the server:**
   ```bash
   uvicorn main:app --reload
   ```
   Server runs on `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Run development server:**
   ```bash
   npm run dev
   ```
   App runs on `http://localhost:3000`

## 🎮 Usage

### Step 1: Upload Resume
- Upload your resume (PDF format)
- Gemini OCR extracts all projects automatically

### Step 2: Select Project
- See list of projects found in your resume
- Choose which one to analyze
- Or enter GitHub URL manually

### Step 3: Choose Your Reality

**🔴 Red Pill - "Roast Me"**
- Get brutal honest feedback on selected project
- See verdict with code evidence
- Voice interview option available

**🔵 Blue Pill - "Rewrite Me"**
- AI-assisted resume improvement
- Project-specific suggestions
- Generate ATS-optimized content

### Step 4: Voice Interview (Optional)
- Click "Voice Interview" button
- Hold microphone to speak
- AI responds with voice
- Real-time conversation about your code

## 🏗️ Project Structure

```
gitreal/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── brain.py             # AI logic (Gemini)
│   ├── ingest_github.py     # GitHub scraper
│   ├── ingest_pdf.py        # PDF parser
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # Environment variables
│
└── frontend/
    ├── app/
    │   ├── page.tsx         # Main application
    │   ├── layout.tsx       # Root layout
    │   └── globals.css      # Global styles
    ├── public/
    │   └── morpheus.png     # Matrix theme assets
    ├── package.json         # Node dependencies
    └── next.config.js       # Next.js config
```

## 🔌 API Endpoints

### `POST /extract_projects`
Extracts projects from resume using Gemini OCR
- **Input:** Resume PDF
- **Output:** List of projects with GitHub URLs

### `POST /analyze`
Analyzes resume against GitHub code
- **Input:** Resume PDF, GitHub URL
- **Output:** Analysis JSON with critique, false claims, suggestions

### `POST /chat`
Interactive chat with AI
- **Input:** Message, conversation history
- **Output:** AI response

### `POST /listen`
Speech-to-Text via Deepgram
- **Input:** Audio file (webm)
- **Output:** Transcribed text

### `POST /speak`
Text-to-Speech via Deepgram
- **Input:** Text
- **Output:** Audio stream (MP3)

### `POST /interview_start_voice`
Starts voice interview session
- **Input:** None (uses cached analysis)
- **Output:** Initial interview question

### `POST /interview_respond_voice`
Responds in voice interview
- **Input:** User's answer text
- **Output:** AI follow-up question

## 🎨 Design Features

- **Matrix Theme** - Green terminal aesthetic
- **CRT Effects** - Scanlines and flicker
- **Glitch Animations** - Text effects on Morpheus screen
- **Loading Animations** - Matrix rain during processing
- **Responsive Design** - Works on desktop and mobile

## 🔐 Security Notes

- Never commit `.env` files
- Keep API keys secure
- GitHub tokens should have minimal permissions
- Resume data is not persisted (in-memory only)

## 🐛 Troubleshooting

**Backend won't start:**
- Check Python version (3.11+)
- Verify virtual environment is activated
- Ensure all dependencies are installed
- Check `.env` file exists with valid API key

**Frontend won't start:**
- Check Node.js version (18+)
- Run `npm install` again
- Clear `.next` folder and rebuild
- Check port 3000 is not in use

**Analysis fails:**
- Verify GitHub URL is public or token is valid
- Check resume PDF is not corrupted
- Ensure Gemini API key is valid and has quota

## 📝 Environment Variables

### Backend `.env`
```env
GEMINI_API_KEY=your_gemini_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
GITHUB_TOKEN=your_github_token  # Optional, for private repos
```

## 🖼️ Screenshots

| Landing | Project Selection | Roast Results |
|---------|------------------|---------------|
| Matrix-themed upload | Choose your project | Brutal analysis |

## 🤝 Contributing

This is a hackathon project, but suggestions are welcome!

## 📄 License

MIT License - Feel free to use and modify

## 🙏 Acknowledgments

- Inspired by **The Matrix**
- AI powered by **Google Gemini 2.5**
- Voice powered by **Deepgram**
- Built with **Next.js** and **FastAPI**

---

<div align="center">

**"Remember, all I'm offering is the truth. Nothing more."**

Made with 💚 and brutal honesty by [@electrifiedchan](https://github.com/electrifiedchan)

</div>
