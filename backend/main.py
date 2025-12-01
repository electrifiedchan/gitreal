import os
import shutil
import re
import base64
import io
import time
import logging
from collections import OrderedDict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, field_validator
from typing import List, Optional
from dotenv import load_dotenv

# Deepgram for Voice
from deepgram import DeepgramClient

import ingest_github
import ingest_pdf
import brain

load_dotenv()

# --- LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize Deepgram
try:
    dg_key = os.getenv("DEEPGRAM_API_KEY")
    if dg_key and dg_key != "YOUR_DEEPGRAM_KEY_HERE":
        deepgram = DeepgramClient(api_key=dg_key)
        logger.info("✅ Deepgram Voice System Online")
    else:
        deepgram = None
        logger.warning("⚠️ Deepgram not configured: No API key")
except Exception as e:
    deepgram = None
    logger.warning(f"⚠️ Deepgram not configured: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB = {}


# --- LRU CACHE WITH TTL ---
class LRUCache:
    """LRU Cache with size limit and TTL to prevent memory leaks"""

    def __init__(self, max_size: int = 50, ttl_seconds: int = 3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl_seconds  # Time-to-live in seconds
        self.timestamps = {}

    def get(self, key: str):
        """Get item from cache, returns None if expired or not found"""
        if key not in self.cache:
            return None

        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl:
            self.delete(key)
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value):
        """Set item in cache with eviction if needed"""
        # If key exists, update it
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
            self.timestamps[key] = time.time()
            return

        # Evict oldest if at capacity
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            self.delete(oldest_key)
            logger.debug(f"Cache evicted: {oldest_key}")

        # Add new item
        self.cache[key] = value
        self.timestamps[key] = time.time()

    def delete(self, key: str):
        """Remove item from cache"""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]

    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.timestamps.clear()

    def __contains__(self, key: str):
        """Check if key exists and is not expired"""
        return self.get(key) is not None

    def __len__(self):
        return len(self.cache)


# Use LRU cache instead of plain dict (max 50 repos, 1 hour TTL)
REPO_CACHE = LRUCache(max_size=50, ttl_seconds=3600)

class ChatRequest(BaseModel):
    message: str
    history: List[dict] = []

class RepoRequest(BaseModel):
    github_url: str

    @field_validator('github_url')
    @classmethod
    def validate_github_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('GitHub URL is required')
        v = v.strip()
        # Must be a GitHub URL
        if 'github.com' not in v.lower():
            raise ValueError('Must be a valid GitHub URL (e.g., https://github.com/user/repo)')
        # Basic pattern check
        pattern = r'(https?://)?(www\.)?github\.com/[\w\-\.]+/[\w\-\.]+'
        if not re.match(pattern, v, re.IGNORECASE):
            raise ValueError('Invalid GitHub URL format')
        return v


# Input validation helpers
ALLOWED_FILE_EXTENSIONS = {'.pdf'}
MAX_FILE_SIZE_MB = 10

def validate_file_upload(file: UploadFile) -> tuple[bool, str]:
    """Validate uploaded file"""
    if not file or not file.filename:
        return False, "No file provided"

    # Check extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_FILE_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_FILE_EXTENSIONS)}"

    return True, ""


def extract_github_details(url):
    """
    Extracts owner, repo, AND branch from URL.
    """
    if not url:
        return None, None, None
        
    # Clean up
    clean = url.replace("https://", "").replace("http://", "").replace("github.com/", "")
    parts = clean.split("/")
    
    if len(parts) < 2:
        return None, None, None
        
    owner = parts[0]
    repo = parts[1]
    branch = None
    
    # Check for /tree/BRANCH_NAME
    if "tree" in parts:
        try:
            tree_index = parts.index("tree")
            if len(parts) > tree_index + 1:
                branch = parts[tree_index + 1]
        except:
            pass
            
    return owner, repo, branch

@app.get("/")
def health_check():
    return {"status": "GitReal System Online", "mode": "Matrix", "voice": "Deepgram" if deepgram else "Browser"}

# ============ DEEPGRAM VOICE ENDPOINTS ============

@app.post("/listen")
async def listen_to_audio(file: UploadFile = File(...)):
    """
    Deepgram Speech-to-Text - Transcribes user voice to text
    """
    if not deepgram:
        return {"text": "", "error": "Deepgram not configured"}

    try:
        buffer_data = await file.read()
        # Use httpx to call Deepgram STT API directly (most reliable)
        import httpx
        url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true"
        headers = {
            "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}",
            "Content-Type": "audio/webm"
        }
        response = httpx.post(url, headers=headers, content=buffer_data, timeout=30.0)

        if response.status_code == 200:
            result = response.json()
            transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
            print(f"🎤 Transcribed: {transcript[:50] if transcript else 'empty'}...")
            return {"text": transcript}
        else:
            print(f"❌ Deepgram API Error: {response.status_code} - {response.text}")
            return {"text": "", "error": f"Deepgram API error: {response.status_code}"}
    except Exception as e:
        print(f"❌ Deepgram Listen Error: {e}")
        return {"text": "", "error": str(e)}

@app.post("/speak")
async def text_to_speech(text: str = Form(...)):
    """
    Deepgram Text-to-Speech - Converts AI response to audio
    """
    if not deepgram:
        return {"error": "Deepgram not configured"}

    try:
        # Use httpx to call Deepgram TTS API directly (most reliable)
        import httpx
        url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
        headers = {
            "Authorization": f"Token {os.getenv('DEEPGRAM_API_KEY')}",
            "Content-Type": "application/json"
        }
        response = httpx.post(url, headers=headers, json={"text": text}, timeout=30.0)

        if response.status_code == 200:
            audio_data = response.content
            print(f"🔊 TTS generated: {len(audio_data)} bytes")
            return StreamingResponse(
                io.BytesIO(audio_data),
                media_type="audio/mp3",
                headers={"Content-Disposition": "inline; filename=speech.mp3"}
            )
        else:
            print(f"❌ Deepgram API Error: {response.status_code} - {response.text}")
            return {"error": f"Deepgram API error: {response.status_code}"}
    except Exception as e:
        print(f"❌ Deepgram Speak Error: {e}")
        return {"error": str(e)}

# ============ CORE ENDPOINTS ============

@app.post("/validate_resume")
async def validate_resume(file: UploadFile = File(...)):
    """
    🛡️ THE GATEKEEPER: Validates if uploaded PDF is a resume BEFORE any processing.
    Called immediately on file upload (at the gate).
    """
    # Validate file extension
    is_valid, error_msg = validate_file_upload(file)
    if not is_valid:
        return {"valid": False, "reason": error_msg}

    logger.info(f"🛡️ Gatekeeper checking: {file.filename}")
    temp_filename = f"temp_validate_{file.filename}"

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        resume_text = ingest_pdf.parse_pdf(temp_filename)

        # AI-powered validation with model fallback
        is_resume, rejection_reason = brain.validate_is_resume(resume_text)

        if is_resume:
            logger.info(f"✅ Gatekeeper approved: {file.filename}")
            return {"valid": True, "reason": ""}
        else:
            logger.warning(f"❌ Gatekeeper rejected: {rejection_reason}")
            return {"valid": False, "reason": rejection_reason}

    except Exception as e:
        logger.error(f"❌ Gatekeeper error: {e}")
        return {"valid": False, "reason": f"Error processing file: {str(e)}"}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


@app.post("/extract_projects")
async def extract_projects(file: UploadFile = File(...)):
    """
    Step 1: Upload resume, extract project names and GitHub URLs using Gemini OCR
    Returns list of projects for user to choose from
    """
    # Validate file upload
    is_valid, error_msg = validate_file_upload(file)
    if not is_valid:
        logger.warning(f"Invalid file upload: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

    logger.info(f"📥 Extracting projects from resume: {file.filename}")
    temp_filename = f"temp_{file.filename}"

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        resume_text = ingest_pdf.parse_pdf(temp_filename)

        # 🛡️ THE GATEKEEPER: Validate this is actually a resume/CV
        is_valid, rejection_reason = brain.validate_is_resume(resume_text)
        if not is_valid:
            logger.warning(f"❌ Document rejected: {rejection_reason}")
            raise HTTPException(status_code=400, detail=rejection_reason)

        # Use Gemini to extract projects
        projects = brain.extract_projects_from_resume(resume_text)

        # Store resume for later use
        DB['pending_resume'] = resume_text

        return {
            "status": "success",
            "projects": projects,
            "resume_preview": resume_text[:500] + "..."
        }
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        print(f"❌ Error extracting projects: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


@app.post("/analyze")
async def analyze_portfolio(
    file: UploadFile = File(...),
    github_url: Optional[str] = Form(None),
    project_name: Optional[str] = Form(None)
):
    # Validate file upload
    is_valid, error_msg = validate_file_upload(file)
    if not is_valid:
        logger.warning(f"Invalid file upload: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)

    # Validate GitHub URL if provided
    if github_url and github_url.strip() and github_url != "null":
        github_url = github_url.strip()
        if 'github.com' not in github_url.lower():
            raise HTTPException(status_code=400, detail="Invalid GitHub URL format")

    logger.info(f"📥 Received Analysis Request.")
    logger.info(f"   📁 Selected Project: {project_name or 'None specified'}")
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        resume_text = ingest_pdf.parse_pdf(temp_filename)

        # 🛡️ THE GATEKEEPER: Validate this is actually a resume/CV
        is_valid, rejection_reason = brain.validate_is_resume(resume_text)
        if not is_valid:
            logger.warning(f"❌ Document rejected: {rejection_reason}")
            raise HTTPException(status_code=400, detail=rejection_reason)

        # IMPORTANT: Only use the URL explicitly provided by user's project selection
        # Do NOT auto-scan resume for GitHub URLs - user chose a specific project
        target_url = github_url if github_url and github_url != "null" and github_url.strip() else None

        if target_url:
            print(f"   🎯 Using selected project URL: {target_url}")
        else:
            print(f"   ⚠️ No GitHub URL provided - analyzing resume claims only (PHANTOMWARE CHECK)")

        code_context = ""
        if target_url:
            owner, repo, branch = extract_github_details(target_url)
            if owner and repo:
                cache_key = f"{owner}/{repo}/{branch}"
                cached = REPO_CACHE.get(cache_key)
                if cached:
                    logger.info(f"   ⚡ Cache Hit: {cache_key}")
                    code_context = cached
                else:
                    logger.info(f"   💻 Target: {owner}/{repo} (Branch: {branch or 'Auto'})")
                    code_context = ingest_github.fetch_repo_content(owner, repo, branch)
                    # Cache if valid
                    if code_context and len(code_context) > 100:
                        REPO_CACHE.set(cache_key, code_context)
            else:
                code_context = "Error: Invalid URL extracted."
        else:
            # No GitHub provided = PHANTOMWARE CHECK MODE
            # AI will flag all project claims as "unverified" since there's no code to prove them
            code_context = "⚠️ NO CODE PROVIDED. This project has NO GitHub link. All claims are UNVERIFIED and should be flagged as potential PHANTOMWARE."

        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        # Pass project_name to focus the analysis on ONLY that project
        analysis_json = brain.analyze_resume_vs_code(resume_text, code_context, project_name)
        
        # Parse JSON to construct chat message
        import json
        try:
            data = json.loads(analysis_json)
            critique = "\n".join([f"- {x}" for x in data.get("project_critique", [])])
            claims = "\n".join([f"- {x}" for x in data.get("false_claims", [])])
            suggestions = "\n".join([f"- {x}" for x in data.get("resume_suggestions", [])])
            
            chat_msg = f"""**REAL WORLD CRITIQUE:**
{critique}

**FALSE CLAIMS / VERIFICATION:**
{claims}

**RESUME ADDITIONS:**
{suggestions}"""
        except:
            chat_msg = "Analysis Complete. Check Dashboard for details."

        DB['current_user'] = {
            "resume": resume_text,
            "code": code_context[:50000],
            "analysis": analysis_json
        }

        return {
            "status": "success",
            "data": analysis_json,
            "initial_chat": chat_msg
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/add_repo")
async def add_repo_context(request: RepoRequest):
    print(f"📥 Adding Repo: {request.github_url}")
    try:
        owner, repo, branch = extract_github_details(request.github_url)
        
        if not owner or not repo:
             raise HTTPException(status_code=400, detail="Invalid GitHub URL")
        
        cache_key = f"{owner}/{repo}/{branch}"
        cached = REPO_CACHE.get(cache_key)
        if cached:
            logger.info(f"   ⚡ Cache Hit: {cache_key}")
            code_context = cached
        else:
            logger.info(f"   💻 Fetching: {owner}/{repo} (Branch: {branch or 'Default'})")
            # Pass the extracted branch to the scraper
            code_context = ingest_github.fetch_repo_content(owner, repo, branch)
            if code_context and len(code_context) >= 100:
                REPO_CACHE.set(cache_key, code_context)

        if not code_context or len(code_context) < 100:
            return {"status": "error", "bullets": "⚠️ ACCESS DENIED: Repo is empty, Private, or Branch not found."}

        bullets = brain.generate_star_bullets(code_context)

        if 'current_user' in DB:
            DB['current_user']['code'] += f"\n\n--- NEW REPO: {repo} ---\n{code_context[:20000]}"

        return {"status": "success", "bullets": bullets}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/interview_start")
async def start_interview():
    user_data = DB.get('current_user')
    if not user_data:
        return {"status": "error", "message": "No data found."}
    
    # Generate the "Opening Shot"
    question = brain.generate_interview_challenge(user_data['code'], user_data['analysis'])
    
    return {"status": "success", "question": question}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_data = DB.get('current_user')
    if not user_data:
        return {"response": "⚠️ SYSTEM ERROR: No data found."}

    context_summary = f"""
    --- RESUME ---
    {user_data['resume'][:1000]}...
    --- CODE EVIDENCE ---
    {user_data['code']}
    """

    gemini_history = [] 
    for msg in request.history:
        role = "user" if msg['type'] == 'user' else "model"
        gemini_history.append({"role": role, "parts": [msg['text']]})

    response_text = brain.get_chat_response(gemini_history, request.message, context_summary)
    
    return {"response": response_text}

@app.post("/generate_resume")
async def generate_resume_endpoint():
    user_data = DB.get('current_user')
    if not user_data:
        return {"response": "⚠️ ERROR: No data found."}

    # Call the new brain function
    new_resume = brain.generate_ats_resume(user_data['resume'], user_data['code'])

    return {"status": "success", "resume": new_resume}

# ============ VOICE INTERVIEW ENDPOINTS ============

class VoiceInterviewRequest(BaseModel):
    message: str
    history: List[dict] = []

@app.post("/voice_interview")
async def voice_interview_endpoint(request: VoiceInterviewRequest):
    """
    Voice interview endpoint - receives text (from browser STT),
    returns text response + audio (TTS from Gemini)
    """
    user_data = DB.get('current_user')
    if not user_data:
        return {"status": "error", "message": "No data found."}

    context_summary = f"""
    --- RESUME ---
    {user_data['resume'][:1000]}...
    --- CODE EVIDENCE ---
    {user_data['code'][:30000]}
    --- ANALYSIS ---
    {user_data['analysis'][:5000]}
    """

    # Get interview response from Gemini
    response_text = brain.get_interview_response(
        request.history,
        request.message,
        context_summary
    )

    # Generate audio using Gemini TTS
    audio_base64 = None
    try:
        audio_data = brain.generate_speech(response_text)
        if audio_data:
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        print(f"TTS Error: {e}")

    return {
        "status": "success",
        "response": response_text,
        "audio": audio_base64
    }

@app.post("/interview_start_voice")
async def start_voice_interview():
    """Start voice interview - returns opening question with audio"""
    user_data = DB.get('current_user')
    if not user_data:
        return {"status": "error", "message": "No data found."}

    # Initialize voice chat session
    brain.init_voice_chat(user_data['resume'], user_data['code'])

    # Generate the opening question
    question = brain.generate_interview_challenge(user_data['code'], user_data['analysis'])

    # Generate audio for the question
    audio_base64 = None
    try:
        audio_data = brain.generate_speech(question)
        if audio_data:
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    except Exception as e:
        print(f"TTS Error: {e}")

    return {
        "status": "success",
        "question": question,
        "audio": audio_base64
    }


class VoiceTextRequest(BaseModel):
    text: str

@app.post("/voice_chat")
async def voice_chat_endpoint(request: VoiceTextRequest):
    """
    Text-based voice chat - receives transcribed text, returns AI response.
    Frontend handles speech-to-text (browser) and text-to-speech (browser).
    This is the SIMPLE & RELIABLE approach for hackathon demo.
    """
    user_data = DB.get('current_user')
    if not user_data:
        return {"status": "error", "response": "No data found. Upload resume first."}

    try:
        print(f"🎤 Received voice text: {request.text[:50]}...")

        # Get AI response
        response_text = brain.process_voice_text(request.text)
        print(f"🤖 AI Response: {response_text[:50]}...")

        return {
            "status": "success",
            "response": response_text
        }

    except Exception as e:
        print(f"❌ Voice Chat Error: {e}")
        return {"status": "error", "response": f"Error: {str(e)}"}