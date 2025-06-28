import os
import json
import uuid
from datetime import datetime
import logging
import asyncio
from pathlib import Path
import tempfile
import atexit
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Form, UploadFile, File, Depends
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.background import BackgroundTasks
from google.oauth2 import id_token
from tts_manager import EdgeTTSManager 
from google.auth.transport import requests as google_requests

import asyncio
from contextlib import asynccontextmanager

# Add this global variable
cleanup_task: Optional[asyncio.Task] = None

# Import Mai's core components
from llm_handler import EnhancedLLMHandler as LLMHandler
from voice_interface import VoiceInterface

try:
    from memory_manager import MemoryManager
except ImportError:
    logging.warning("MemoryManager not found. Using a placeholder class. Ensure memory_manager.py exists and is correctly implemented.")
    class MemoryManager:
        def __init__(self, *args, **kwargs):
            logging.info("Placeholder MemoryManager initialized.")
            pass
        def retrieve_memories(self, query: str, user_id: str, max_memories: int) -> List[str]:
            logging.warning("MemoryManager.retrieve_memories is a placeholder.")
            return []
        def store_conversation(self, user_message: str, ai_response: str, user_id: str, timestamp: str) -> int:
            logging.warning("MemoryManager.store_conversation is a placeholder.")
            return 0
        def clear_user_memories(self, user_id: str) -> int:
            logging.warning("MemoryManager.clear_user_memories is a placeholder.")
            return 0
        def get_memory_stats(self) -> Dict:
            logging.warning("MemoryManager.get_memory_stats is a placeholder.")
            return {"total_memories": 0, "memories_by_user": {}}
        def get_recent_memories(self, user_id: str, limit: int) -> List[str]:
            logging.warning("MemoryManager.get_recent_memories is a placeholder.")
            return []
        
# Add this import
try:
    from sentiment_analysis import analyze_sentiment
except ImportError:
    logging.warning("sentiment_analysis.py not found. Sentiment analysis will be unavailable.")
    # Optional: create a placeholder analyze_sentiment function if you want to avoid crashes
    async def analyze_sentiment(text: str):
        return "neutral", 1.0 # Default fallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables (ensure you have a .env file with FLASK_SECRET_KEY, TOGETHER_API_KEY)
from dotenv import load_dotenv
load_dotenv()

# Global instances of Mai's core components
llm_handler: Optional[LLMHandler] = None
memory_manager: Optional[MemoryManager] = None
voice_interface: Optional[VoiceInterface] = None

TOGETHER_API_KEY_DIRECT = os.getenv("TOGETHER_API_KEY") #for lama from together.ai
OPENAI_API_KEY_DIRECT = os.getenv("OPENAI_API_KEY") #for gpt 4o from openai
AUDIO_OUTPUT_DIR = Path("audio_output")
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
if not GOOGLE_CLIENT_ID:
    logger.critical("CRITICAL ERROR: GOOGLE_CLIENT_ID is not set in environment or .env file.")
    # You might want to raise an exception or handle this more gracefully based on your deployment
    # For now, let's just log a warning.

# You'll need a way to store email-to-UUID mappings persistently.
# For simplicity, let's use a dictionary in memory for demonstration.
# In a real application, this would be a database (PostgreSQL, MongoDB, SQLite, etc.)
user_email_to_id_mapping: Dict[str, str] = {} # This will not persist across restarts!

# For Rishi's specific ID
RISHI_EMAIL = "restlessrishi@gmail.com"
FIXED_SINGLE_USER_ID: Optional[str] = "6653d9f1-b272-434f-8f2b-b0a96c35a1d2"

# Populate Rishi's ID initially
user_email_to_id_mapping[RISHI_EMAIL] = FIXED_SINGLE_USER_ID #change the global variable to ur own user_id 

# Define MAX_CHAT_HISTORY_LENGTH for session history trimming
MAX_CHAT_HISTORY_LENGTH = 20 # Keep last 20 messages (10 turns) in session

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronously initialize Mai's core components at startup
    and gracefully shut down resources at application exit.
    """
    global llm_handler, memory_manager, voice_interface, cleanup_task
    
    logger.info("Starting Mai components initialization (via lifespan)...")
    try:
        # Get API key
        final_api_key = TOGETHER_API_KEY_DIRECT
        if not final_api_key:
            logger.critical("CRITICAL ERROR: TOGETHER_API_KEY is not set in environment or .env file.")
            raise ValueError("TOGETHER_API_KEY is missing.")

        # 1. Initialize LLM Handler
        logger.info("Initializing LLM handler...")
        llm_handler = LLMHandler(api_key=final_api_key) 
        logger.info("LLM handler initialized successfully.")
        
        logger.info("Testing LLM handler connection...")
        if await llm_handler.test_connection():
            logger.info("LLM handler connection test successful.")
        else:
            logger.error("LLM handler connection test failed. This might cause issues.")

        # 2. Initialize ENHANCED Memory Manager with LLM integration
        logger.info("Initializing enhanced memory manager with enhanced LLM integration...")
        memory_manager = MemoryManager(
            collection_name="mai_memories", 
            persist_directory="./mai_memory",
            llm_handler=llm_handler  # Pass LLM handler for intelligent episode summaries
        ) 
        logger.info("Enhanced memory manager initialized successfully.")

        # 3. Initialize Voice Interface (unchanged)
        logger.info("Initializing voice interface...")
        if llm_handler is None:
            raise RuntimeError("LLMHandler was not initialized, cannot initialize VoiceInterface.")
        
        tts_provider = EdgeTTSManager(voice_name="en-US-AnaNeural")
        voice_interface = VoiceInterface(
            llm_handler=llm_handler,
            tts_manager=tts_provider
        )
        logger.info(f"VoiceInterface initialized successfully with {type(tts_provider).__name__}.")

    except ValueError as ve:
        logger.critical(f"CRITICAL ERROR: API key is missing or invalid - {ve}")
        raise RuntimeError(f"API key missing or invalid: {ve}") from ve
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Unexpected error during component initialization: {e}", exc_info=True)
        raise RuntimeError(f"An unexpected error occurred during Mai component setup: {e}") from e
    
    # Start periodic cleanup task
    cleanup_task = asyncio.create_task(periodic_session_cleanup())
    logger.info("Started periodic session cleanup task")

    logger.info("All Mai components initialized.")
    yield  # This yields control to the application

    # --- Code after yield runs on shutdown ---
    logger.info("FastAPI application is shutting down (via lifespan)...")

    # Cancel cleanup task on shutdown
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Stopped periodic session cleanup task")
    
    # NEW: Cleanup active sessions before shutdown
    if memory_manager:
        logger.info("Cleaning up active sessions and creating final episodes...")
        try:
            memory_manager.cleanup_expired_sessions()
            logger.info("Session cleanup completed.")
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")
    
    if llm_handler and hasattr(llm_handler, 'aclose'):
        logger.info("Shutting down LLMHandler's HTTP client gracefully...")
        try:
            await llm_handler.aclose()
            logger.info("LLMHandler HTTP client closed successfully.")
        except Exception as e:
            logger.error(f"Error closing LLMHandler HTTP client during shutdown: {e}", exc_info=True)
    logger.info("FastAPI application shutdown complete (via lifespan).")

# --- FastAPI App Setup ---
app = FastAPI(
    title="Mai - Emotionally Intelligent AI Assistant",
    description="An AI assistant powered by Together.ai, with voice and memory capabilities.",
    version="0.1.0",
    lifespan=lifespan
)

# Get secret key from environment variable
SESSION_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'mai-secret-key-change-in-production')
if SESSION_SECRET_KEY == 'mai-secret-key-change-in-production':
    logger.warning("FLASK_SECRET_KEY environment variable not set. Using default. CHANGE THIS IN PRODUCTION!")

# Add Session Middleware
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY)

# Add CORS Middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates setup
templates = Jinja2Templates(directory="templates")

# Static files setup for audio responses
from fastapi.staticfiles import StaticFiles
app.mount("/audio_output", StaticFiles(directory=AUDIO_OUTPUT_DIR), name="audio_output")

# --- Helper Functions ---

async def periodic_session_cleanup():
    """Periodically clean up expired sessions."""
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            if memory_manager:
                memory_manager.cleanup_expired_sessions()
                logger.debug("Periodic session cleanup completed")
        except Exception as e:
            logger.error(f"Error in periodic session cleanup: {e}")

async def get_user_id(request: Request) -> str:
    """
    Retrieves or creates a user ID for the current session, prioritizing Google login.
    """
    # 1. Check for authenticated Google user ID in session first
    if 'google_user_id' in request.session and request.session['google_user_id']:
        logger.debug(f"Using Google authenticated user ID from session: {request.session['google_user_id']}")
        return request.session['google_user_id']

    # 2. If no Google ID, check for a general session user ID
    if 'user_id' not in request.session:
        # Fallback to generating a new UUID if no user ID (Google or general) is found.
        # This primarily serves unauthenticated users or initial visits before Google login.
        request.session['user_id'] = str(uuid.uuid4())
        logger.info(f"Created new non-authenticated user session ID: {request.session['user_id']}")
    
    logger.debug(f"Using general session user ID: {request.session['user_id']}")
    return request.session['user_id']


async def get_or_create_user_id_from_email(email: str, user_name: Optional[str] = None) -> str:
    """
    Gets the UUID for a given email, or creates a new one if it doesn't exist,
    delegating persistence to MemoryManager.
    """
    if memory_manager is None:
        logger.error("MemoryManager not initialized, cannot get or create user ID from email.")
        raise RuntimeError("Memory services are not ready.")

    # This will use the logic within MemoryManager to retrieve/create the ID
    # and persist the email-to-ID and ID-to-name mapping.
    user_id = memory_manager.get_or_create_user_id(email=email, default_name=user_name)

    # Ensure Rishi's fixed ID is honored if his email logs in
    if email == RISHI_EMAIL and user_id != FIXED_SINGLE_USER_ID:
        logger.warning(f"Rishi's email '{RISHI_EMAIL}' logged in with a non-fixed ID '{user_id}'. "
                       f"Attempting to map to fixed ID '{FIXED_SINGLE_USER_ID}'. "
                       f"Consider pre-populating this in memory_manager's user map file.")
        # This is a complex edge case for existing systems. For a new system, 
        # ensure FIXED_SINGLE_USER_ID is the first entry in the persisted map for Rishi's email.
        # Forcing it here might overwrite legitimate new IDs if not careful.
        # A cleaner approach is to initialize memory_manager's internal map with Rishi's ID at startup
        # if that's the desired behavior.
        # For now, let's just make sure the correct ID is returned for Rishi if he's logging in.
        return FIXED_SINGLE_USER_ID # Force Rishi's ID if his email is provided.

    return user_id

async def get_chat_history(request: Request) -> List[Dict]:
    """Retrieves the chat history for the current session."""
    if 'chat_history' not in request.session:
        request.session['chat_history'] = []
    return request.session['chat_history']

async def add_to_chat_history(request: Request, role: str, content: str):
    """Adds a message to the session's chat history and trims it."""
    chat_history = await get_chat_history(request)
    chat_history.append({
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat()
    })
    # Trim history to keep only the most recent messages for conversational context
    if len(chat_history) > MAX_CHAT_HISTORY_LENGTH:
        request.session['chat_history'] = chat_history[-MAX_CHAT_HISTORY_LENGTH:]
    else:
        request.session['chat_history'] = chat_history

def clean_mai_response(text: str) -> str:
    """
    Cleans Mai's response for storage in MemoryManager.
    This function should remove any specific prefixes, suffixes, 
    or internal formatting from the LLM's raw output that
    should not be part of the actual memory.
    """
    # Example: If your LLM sometimes outputs "AI: " or "Mai says: "
    text = text.replace("AI: ", "").replace("Mai says: ", "").strip()
    # Add any other cleaning rules as needed based on your LLM's output
    return text


# --- FastAPI Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Renders the main chat interface."""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/voice", response_class=HTMLResponse)
async def audio_interface_route(request: Request):
    """Renders the audio chat interface."""
    return templates.TemplateResponse("audio_chat.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(
    request: Request,
    message: dict
):
    """Handles text-based chat interactions."""
    if llm_handler is None or memory_manager is None:
        logger.error("Mai components not initialized in /chat route.")
        raise HTTPException(status_code=503, detail='Mai is not ready. Please try again or check server logs.')

    try:
        user_message = message.get('message', '').strip()
        if not user_message:
            logger.warning("Received chat request with no message data.")
            raise HTTPException(status_code=400, detail='No message provided')

        user_id = await get_user_id(request)
        chat_history = await get_chat_history(request)

        logger.info(f"User {user_id}: Processing message: '{user_message[:70]}{'...' if len(user_message) > 70 else ''}'")

        # --- User Name Association Logic ---
        user_name_for_storage: Optional[str] = None
        # Check if a user name was stored from Google login
        if 'user_name' in request.session:
            user_name_for_storage = request.session['user_name']
            logger.debug(f"Assigned user_name '{user_name_for_storage}' from session to user_id: {user_id}")
        elif 'user_email' in request.session:
            # If name isn't directly available, use email prefix or default
            email_prefix = request.session['user_email'].split('@')[0]
            user_name_for_storage = email_prefix.capitalize()
            logger.debug(f"Assigned user_name '{user_name_for_storage}' derived from email for user_id: {user_id}")
        else:
            # Fallback if no authenticated name/email, or for the fixed Rishi user
            # If the user_id is Rishi's fixed ID, set the name
            if user_id == FIXED_SINGLE_USER_ID:
                user_name_for_storage = "Rishi"
                logger.debug(f"Assigned user_name 'Rishi' to fixed user_id: {user_id}")
            else:
                user_name_for_storage = "Guest" # Or "User" or some other default
                logger.debug(f"Assigned user_name '{user_name_for_storage}' as fallback for user_id: {user_id}")
        # --- End User Name Association Logic ---

        # Analyze sentiment of the user message
        user_emotion, emotion_confidence = await analyze_sentiment(user_message)
        logger.info(f"User {user_id} detected emotion: {user_emotion} (confidence: {emotion_confidence:.2f})")

        logger.debug("Retrieving relevant memories...")
        # Get memory context from the MemoryManager (ChromaDB)
        memory_context = memory_manager.retrieve_memories(
            query=user_message,
            user_id=user_id,
            limit=50,
            include_flash=True,  # Include flash memories
            memory_types=None,   # All types, or specify ['core', 'episodic', 'conversational']
            importance_filter=None  # All importance levels, or specify 'high', 'medium', 'low'
        )
        memory_types_count = {}
        flash_count = 0
        for mem in memory_context:
            source = mem.get('metadata', {}).get('source', 'persistent')
            if source == 'flash_memory':
                flash_count += 1
            else:
                mem_type = mem.get('metadata', {}).get('type', 'conversational')
                memory_types_count[mem_type] = memory_types_count.get(mem_type, 0) + 1

        logger.info(f"Retrieved {len(memory_context)} relevant memories for user {user_id}: "
                f"{flash_count} flash, {dict(memory_types_count)} persistent")

        logger.debug("Generating response using LLM...")
        llm_response = await llm_handler.generate_response(
            user_message=user_message,
            chat_history=chat_history,
            memory_context=memory_context,  # Now includes full metadata
            user_emotion=user_emotion,
            emotion_confidence=emotion_confidence,
        )

        if not llm_response['success']:
            logger.error(f"LLM generation failed for user {user_id}: {llm_response.get('error', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail='Failed to generate response',
                headers={"X-Mai-Error": llm_response.get('response', 'Sorry, I encountered an error generating a response.')}
            )

        ai_response = llm_response['response']
        logger.info(f"Mai response: '{ai_response[:70]}{'...' if len(ai_response) > 70 else ''}'")

        # Analyze sentiment of the AI response (using a mock or actual sentiment analysis)
        ai_emotion, ai_emotion_confidence = await analyze_sentiment(ai_response)
        logger.info(f"AI detected emotion: {ai_emotion} (confidence: {ai_emotion_confidence:.2f})")

        # Clean the AI response before storing in long-term memory
        cleaned_ai_response = clean_mai_response(ai_response)

        logger.debug("Storing conversation in memory...")
        # Store the conversation in MemoryManager (ChromaDB) for long-term memory
        stored_memories = memory_manager.store_conversation(
            user_message=user_message,
            ai_response=cleaned_ai_response, # Store the cleaned response
            user_id=user_id,
            user_emotion=user_emotion, # Store user emotion in memory
            user_emotion_confidence=emotion_confidence, # Store confidence (optional, but good for data)
            ai_emotion=ai_emotion, # Store AI emotion in memory
            user_name=user_name_for_storage # Pass the determined user_name
        )
        logger.info(f"Stored {stored_memories} new memories for user {user_id}" + (f" ({user_name_for_storage})" if user_name_for_storage else ""))

        # Update the session's chat history for short-term conversational context
        await add_to_chat_history(request, 'user', user_message)
        await add_to_chat_history(request, 'assistant', ai_response) # Store the original response for display

        return JSONResponse({
            'success': True,
            'mai_response': ai_response,
            'metadata': {
                'memories_used': len(memory_context),
                'memories_stored': stored_memories,
                'model': llm_response.get('model', 'unknown'),
                'usage': llm_response.get('usage', {}),
                'response_mode': llm_response.get('response_mode', 'unknown'),  # NEW
                'user_emotion': user_emotion,
                'emotion_confidence': emotion_confidence,
                'ai_emotion': ai_emotion,
                'user_name': user_name_for_storage,
                'memory_breakdown': {  # NEW detailed breakdown
                    'flash_memories': flash_count,
                    'persistent_types': memory_types_count,
                    'total_context': len(memory_context)
                }
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled error in chat endpoint for user {user_id}: {e}") # Use user_id directly here
        raise HTTPException(
            status_code=500,
            detail='Internal server error',
            headers={"X-Mai-Error": 'Sorry, I encountered an unexpected error. Please try again.'}
        )

@app.get("/enhanced_memory_insights")
async def enhanced_memory_insights_endpoint(request: Request):
    """Get detailed insights about the user's memory patterns."""
    if memory_manager is None:
        raise HTTPException(status_code=503, detail='Memory services are not ready.')
    
    try:
        user_id = await get_user_id(request)
        
        # Get comprehensive memory insights
        stats = memory_manager.get_memory_stats()
        recent_memories = memory_manager.get_recent_memories(user_id, limit=10, include_flash=True)
        flash_memories = memory_manager.get_flash_memories(user_id)
        session_stats = memory_manager.get_active_session_stats()
        
        # Analyze memory patterns
        user_name = memory_manager.get_user_name(user_id) or "Guest"
        user_persistent_count = stats.get('memories_by_user', {}).get(user_name, 0)
        
        # Emotion analysis from recent memories
        recent_emotions = []
        contexts = []
        for mem in recent_memories:
            metadata = mem.get('metadata', {})
            if metadata.get('emotion'):
                recent_emotions.append(metadata['emotion'])
            if metadata.get('context'):
                contexts.append(metadata['context'])
        
        emotion_distribution = {}
        for emotion in recent_emotions:
            emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
        
        context_distribution = {}
        for context in contexts:
            context_distribution[context] = context_distribution.get(context, 0) + 1
        
        return JSONResponse({
            'success': True,
            'user_id': user_id,
            'user_name': user_name,
            'memory_insights': {
                'persistent_memories': user_persistent_count,
                'flash_memories': len(flash_memories),
                'recent_activity': len(recent_memories),
                'active_session': user_id in memory_manager._active_sessions,
                'emotion_patterns': emotion_distribution,
                'context_patterns': context_distribution,
                'session_info': session_stats.get('sessions_by_user', {}).get(user_name, None)
            },
            'system_insights': {
                'total_users': len(stats.get('memories_by_user', {})),
                'total_memories': stats.get('total_persistent_memories', 0),
                'active_sessions': session_stats.get('total_active_sessions', 0)
            }
        })
    
    except Exception as e:
        logger.exception(f"Error getting enhanced memory insights: {e}")
        raise HTTPException(status_code=500, detail='Failed to get memory insights')


@app.get("/session_stats")
async def session_stats_endpoint(request: Request):
    """Get statistics about active sessions."""
    if memory_manager is None:
        raise HTTPException(status_code=503, detail='Memory services are not ready.')
    
    try:
        user_id = await get_user_id(request)
        session_stats = memory_manager.get_active_session_stats()
        
        # Check if current user has an active session
        current_user_session = None
        if hasattr(memory_manager, '_active_sessions') and user_id in memory_manager._active_sessions:
            session_data = memory_manager._active_sessions[user_id]
            current_user_session = {
                'session_id': session_data['session_id'],
                'start_time': session_data['start_time'],
                'interactions': len(session_data['interactions']),
                'last_activity': session_data['last_activity']
            }
        
        return JSONResponse({
            'success': True,
            'current_user_session': current_user_session,
            'global_session_stats': session_stats,
            'user_id': user_id
        })
    except Exception as e:
        logger.exception(f"Error getting session stats: {e}")
        raise HTTPException(status_code=500, detail='Failed to get session statistics')


@app.post("/generate_manual_episode")
async def generate_manual_episode_endpoint(request: Request, data: dict):
    """Manually generate an episode summary from provided conversation data."""
    if llm_handler is None or memory_manager is None:
        raise HTTPException(status_code=503, detail='Mai services are not ready.')
    
    try:
        user_id = await get_user_id(request)
        conversation_data = data.get('conversation_data', [])
        
        if not conversation_data:
            raise HTTPException(status_code=400, detail='No conversation data provided')
        
        # Generate episode using enhanced LLM handler
        episode_result = await llm_handler.generate_episode_summary(
            conversation_data=conversation_data,
            user_context={'user_id': user_id}
        )
        
        if episode_result['success']:
            # Store the generated episode
            user_name = memory_manager.get_user_name(user_id)
            success = memory_manager.store_episode_summary(
                summary_text=episode_result['summary'],
                user_id=user_id,
                episode_context="manual_generation",
                user_name=user_name
            )
            
            return JSONResponse({
                'success': True,
                'episode_summary': episode_result['summary'],
                'stored': success,
                'usage': episode_result.get('usage', {})
            })
        else:
            return JSONResponse({
                'success': False,
                'error': episode_result.get('error', 'Failed to generate episode'),
                'episode_summary': None
            })
    
    except Exception as e:
        logger.exception(f"Error generating manual episode: {e}")
        raise HTTPException(status_code=500, detail='Failed to generate episode')



@app.post("/force_end_session")
async def force_end_session_endpoint(request: Request):
    """Manually end the current user's session and create episode."""
    if memory_manager is None:
        raise HTTPException(status_code=503, detail='Memory services are not ready.')
    
    try:
        user_id = await get_user_id(request)
        
        if hasattr(memory_manager, '_active_sessions') and user_id in memory_manager._active_sessions:
            memory_manager.force_end_session(user_id)
            logger.info(f"Manually ended session for user {user_id}")
            return JSONResponse({
                'success': True,
                'message': 'Session ended and episode creation initiated'
            })
        else:
            return JSONResponse({
                'success': False,
                'message': 'No active session found for user'
            })
    except Exception as e:
        logger.exception(f"Error force ending session: {e}")
        raise HTTPException(status_code=500, detail='Failed to end session')

@app.post("/voice_chat")
async def voice_chat_endpoint(
    request: Request,
    audio: UploadFile = File(...)
):
    """Handles voice-based chat interactions."""
    if voice_interface is None or memory_manager is None or llm_handler is None:
        logger.error("Mai voice components not initialized in /voice_chat route.")
        raise HTTPException(status_code=503, detail='Mai voice services are not ready. Please try again or check server logs.')

    temp_audio_file_path_str: Optional[str] = None
    user_id = await get_user_id(request)
    session_chat_history = await get_chat_history(request) # Get session chat history

    try:
        # --- User Name Association Logic (Same as /chat endpoint) ---
        user_name_for_storage: Optional[str] = None
        # Check if a user name was stored from Google login
        if 'user_name' in request.session:
            user_name_for_storage = request.session['user_name']
            logger.debug(f"Assigned user_name '{user_name_for_storage}' from session to user_id: {user_id}")
        elif 'user_email' in request.session:
            # If name isn't directly available, use email prefix or default
            email_prefix = request.session['user_email'].split('@')[0]
            user_name_for_storage = email_prefix.capitalize()
            logger.debug(f"Assigned user_name '{user_name_for_storage}' derived from email for user_id: {user_id}")
        else:
            # Fallback if no authenticated name/email, or for the fixed Rishi user
            # If the user_id is Rishi's fixed ID, set the name
            if user_id == FIXED_SINGLE_USER_ID:
                user_name_for_storage = "Rishi"
                logger.debug(f"Assigned user_name 'Rishi' to fixed user_id: {user_id}")
            else:
                user_name_for_storage = "Guest" # Or "User" or some other default
                logger.debug(f"Assigned user_name '{user_name_for_storage}' as fallback for user_id: {user_id}")
        # --- End User Name Association Logic ---

        # Save uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio_file:
            contents = await audio.read()
            temp_audio_file.write(contents)
            temp_audio_file_path_str = temp_audio_file.name

        logger.info(f"Saved temporary audio file: {temp_audio_file_path_str}")
        audio_file_path_obj = Path(temp_audio_file_path_str)

        # Step 1: Transcribe the audio
        logger.debug(f"Transcribing audio for user {user_id}...")
        transcription_result = await voice_interface.transcribe_audio_file(audio_file_path_obj)

        if not transcription_result['success']:
            logger.error(f"Transcription failed for user {user_id}: {transcription_result.get('error', 'Unknown transcription error')}")
            raise HTTPException(
                status_code=500,
                detail=transcription_result.get('error', 'Failed to transcribe audio.'),
                headers={"X-Mai-Response": "I'm sorry, I couldn't understand what you said."}
            )

        user_input = transcription_result['transcribed_text']
        if not user_input:
            logger.warning(f"User {user_id}: Transcribed audio was empty.")
            return JSONResponse({"status": "no_speech", "message": "No speech detected or clear text extracted."})

        logger.info(f"User {user_id} transcribed input: '{user_input[:70]}{'...' if len(user_input) > 70 else ''}'")

        # Analyze sentiment of the user voice input
        user_emotion, emotion_confidence = await analyze_sentiment(user_input)
        logger.info(f"User {user_id} voice detected emotion: {user_emotion} (confidence: {emotion_confidence:.2f})")

        # Step 2: Get relevant memories for the LLM
        logger.debug(f"Retrieving relevant memories for voice interaction for user {user_id}...")
        # Note: 'max_memories' was used in your snippet for retrieve_memories,
        # but the function signature expects 'limit'. I've corrected it to 'limit'.
        memory_context = memory_manager.retrieve_memories(
            query=user_input,
            user_id=user_id,
            limit=50 # Changed from max_memories to limit
        )
        logger.info(f"Retrieved {len(memory_context)} relevant memories for user {user_id} for voice interaction.")

        # Step 3: Process the interaction through VoiceInterface
        logger.debug(f"Calling voice_interface.process_voice_interaction for user {user_id}...")
        interaction_result = await voice_interface.process_voice_interaction(
            user_input=user_input,
            chat_history=session_chat_history, # Pass the session's chat history
            memory_context=memory_context,     # Pass the retrieved memory context
            user_emotion=user_emotion, # Pass emotion to VoiceInterface
            emotion_confidence=emotion_confidence, # Pass confidence (optional)
        )

        if not interaction_result['success']:
            logger.error(f"LLM interaction failed for user {user_id}: {interaction_result.get('error', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail=interaction_result.get('error', 'Failed to get a response from Mai.'),
                headers={"X-Mai-Response": interaction_result.get('mai_response', "I'm sorry, I encountered an error generating my response.")}
            )

        mai_response = interaction_result['mai_response']
        # The interaction_result might have AI emotion from LLM handler
        ai_emotion_from_llm = interaction_result.get('emotion')
        ai_emotion_confidence = None # You might want to get this from llm_response directly

        # If AI emotion isn't directly returned by voice_interface.process_voice_interaction,
        # you might need to analyze mai_response here if you want it stored in MemoryManager.
        # For simplicity, assuming interaction_result includes 'emotion' from LLM.
        if not ai_emotion_from_llm:
            ai_emotion_from_llm, ai_emotion_confidence = await analyze_sentiment(mai_response)
            logger.info(f"AI voice response detected emotion: {ai_emotion_from_llm} (confidence: {ai_emotion_confidence:.2f})")
        else: # If emotion came directly from interaction_result
             ai_emotion_confidence = 1.0 # Or some default/derived confidence if not provided

        audio_filename_for_frontend = Path(interaction_result['audio_path']).name if interaction_result['audio_path'] else None

        logger.info(f"User {user_id} Mai's voice response: '{mai_response[:70]}{'...' if len(mai_response) > 70 else ''}'")

        # Step 4: Update session chat history and store in MemoryManager (ChromaDB)
        # Clean the AI response before storing in long-term memory
        cleaned_mai_response = clean_mai_response(mai_response)

        logger.debug(f"Storing voice conversation in memory for user {user_id}...")
        stored_memories = memory_manager.store_conversation(
            user_message=user_input,
            ai_response=cleaned_mai_response, # Store the cleaned response
            user_id=user_id,
            # timestamp=datetime.now().isoformat(), # This is already handled internally by store_conversation
            user_emotion=user_emotion, # Store user emotion in memory
            user_emotion_confidence=emotion_confidence, # Store confidence (optional)
            ai_emotion=ai_emotion_from_llm, # Store AI emotion
            user_name=user_name_for_storage # Pass the determined user_name
        )
        logger.info(f"Stored {stored_memories} new memories for user {user_id} from voice chat" + (f" ({user_name_for_storage})" if user_name_for_storage else ""))

        # Update the session's chat history for short-term conversational context
        await add_to_chat_history(request, 'user', user_input)
        await add_to_chat_history(request, 'assistant', mai_response) # Store the original response for display

        return JSONResponse({
            "status": "success",
            "user_input": user_input, # Always send transcribed input back
            "mai_response": mai_response,
            "audio_filename": audio_filename_for_frontend,
            "user_emotion": user_emotion, # Return user emotion to frontend
            "emotion_confidence": emotion_confidence,
            "ai_emotion": ai_emotion_from_llm, # Return AI emotion
            "user_name": user_name_for_storage # Also return to frontend for clarity
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error in voice_chat endpoint for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if temp_audio_file_path_str and os.path.exists(temp_audio_file_path_str):
            os.remove(temp_audio_file_path_str)
            logger.info(f"Removed temporary audio file: {temp_audio_file_path_str}")

@app.post("/text_to_voice_chat")
async def text_to_voice_chat_endpoint(
    request: Request,
    message: Dict[str, str] # Expecting a JSON body with 'message'
):
    """Handles text-based chat interactions but responds with speech."""
    global llm_handler, memory_manager, voice_interface
    if llm_handler is None or memory_manager is None or voice_interface is None:
        logger.error("Mai components not initialized for /text_to_voice_chat route.")
        raise HTTPException(status_code=503, detail='Mai services are not ready. Please try again or check server logs.')

    # No temp_audio_file_path here anymore, as we're not serving the file directly from here.
    user_id = await get_user_id(request)
    session_chat_history = await get_chat_history(request)

    try:
        user_message = message.get('message', '').strip()
        if not user_message:
            logger.warning(f"User {user_id}: Received text_to_voice_chat request with no message data.")
            raise HTTPException(status_code=400, detail='No message provided')

        # --- User Name Association Logic (Same as other endpoints) ---
        user_name_for_storage: Optional[str] = None
        # Check if a user name was stored from Google login
        if 'user_name' in request.session:
            user_name_for_storage = request.session['user_name']
            logger.debug(f"Assigned user_name '{user_name_for_storage}' from session to user_id: {user_id}")
        elif 'user_email' in request.session:
            # If name isn't directly available, use email prefix or default
            email_prefix = request.session['user_email'].split('@')[0]
            user_name_for_storage = email_prefix.capitalize()
            logger.debug(f"Assigned user_name '{user_name_for_storage}' derived from email for user_id: {user_id}")
        else:
            # Fallback if no authenticated name/email, or for the fixed Rishi user
            # If the user_id is Rishi's fixed ID, set the name
            if user_id == FIXED_SINGLE_USER_ID:
                user_name_for_storage = "Rishi"
                logger.debug(f"Assigned user_name 'Rishi' to fixed user_id: {user_id}")
            else:
                user_name_for_storage = "Guest" # Or "User" or some other default
                logger.debug(f"Assigned user_name '{user_name_for_storage}' as fallback for user_id: {user_id}")
        # --- End User Name Association Logic ---

        logger.info(f"User {user_id}: Processing text message for voice response: '{user_message[:70]}{'...' if len(user_message) > 70 else ''}'")

        # Analyze sentiment of the user message
        user_emotion, emotion_confidence = await analyze_sentiment(user_message)
        logger.info(f"User {user_id} detected emotion: {user_emotion} (confidence: {emotion_confidence:.2f})")

        # Get memory context from the MemoryManager (ChromaDB)
        logger.debug(f"User {user_id}: Retrieving relevant memories for text-to-voice interaction...")
        memory_context = memory_manager.retrieve_memories(
            query=user_message,
            user_id=user_id,
            limit=50
        )
        logger.info(f"User {user_id}: Retrieved {len(memory_context)} relevant memories.")

        # Step 1: Generate text response using LLM
        logger.debug(f"User {user_id}: Generating text response using LLM for text-to-voice...")
        llm_response = await llm_handler.generate_response(
            user_message=user_message,
            chat_history=session_chat_history, # Pass the session's chat history
            memory_context=memory_context,     # Pass the retrieved memory context
            user_emotion=user_emotion,         # Pass emotion to the LLMHandler
            emotion_confidence=emotion_confidence, # Pass confidence too
        )

        if not llm_response['success']:
            logger.error(f"LLM generation failed for user {user_id} in text-to-voice: {llm_response.get('error', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail='Failed to generate response',
                headers={"X-Mai-Error": llm_response.get('response', 'Sorry, I encountered an error generating a text response.')}
            )

        mai_response_text = llm_response['response']
        logger.info(f"User {user_id}: Mai text response: '{mai_response_text[:70]}{'...' if len(mai_response_text) > 70 else ''}'")

        # Analyze sentiment of the AI response (for storage and potential TTS tone)
        ai_emotion, ai_emotion_confidence = await analyze_sentiment(mai_response_text)
        logger.info(f"User {user_id}: AI detected emotion for text-to-voice: {ai_emotion} (confidence: {ai_emotion_confidence:.2f})")

        # Step 2: Convert the LLM's text response to speech using generate_speech
        logger.debug(f"User {user_id}: Converting text response to speech...")
        
        # Generate a unique filename for the audio output
        unique_audio_filename = f"mai_response_{uuid.uuid4()}.mp3"
        
        # Call the existing generate_speech method
        audio_full_path_str = await voice_interface.generate_speech(
            text=mai_response_text,
            filename=unique_audio_filename,
            #emotion=ai_emotion # this is where response is generated for text to voice part
        )

        if not audio_full_path_str:
            logger.error(f"User {user_id}: Failed to generate audio for Mai's response.")
            raise HTTPException(
                status_code=500,
                detail='Failed to generate speech.',
                headers={"X-Mai-Response": "I'm sorry, I couldn't generate a voice response."}
            )
        
        # We don't need to check audio_full_path.exists() here, as it will be checked by get_audio_response
        # and the cleaning is handled by the get_audio_response endpoint.
        audio_filename_for_frontend = Path(audio_full_path_str).name
        logger.info(f"User {user_id}: Audio generated at: {audio_full_path_str}")

        # Clean the AI response before storing in long-term memory
        cleaned_mai_response = clean_mai_response(mai_response_text)

        logger.debug(f"User {user_id}: Storing conversation in memory for text-to-voice...")
        stored_memories = memory_manager.store_conversation(
            user_message=user_message,
            ai_response=cleaned_mai_response, # Store the cleaned response
            user_id=user_id,
            user_emotion=user_emotion,
            user_emotion_confidence=emotion_confidence,
            ai_emotion=ai_emotion,
            user_name=user_name_for_storage
        )
        logger.info(f"User {user_id}: Stored {stored_memories} new memories from text-to-voice chat" + (f" ({user_name_for_storage})" if user_name_for_storage else ""))

        # Update the session's chat history for short-term conversational context
        await add_to_chat_history(request, 'user', user_message)
        await add_to_chat_history(request, 'assistant', mai_response_text)

        # Return JSON response with the audio filename
        return JSONResponse({
            "status": "success",
            "user_input": user_message, # User's original text input
            "mai_response": mai_response_text, # Mai's text response
            "audio_filename": audio_filename_for_frontend, # Filename for frontend to play
            "user_emotion": user_emotion,
            "emotion_confidence": emotion_confidence,
            "ai_emotion": ai_emotion,
            "user_name": user_name_for_storage
        })

    except HTTPException:
        raise # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        logger.error(f"Unhandled error in text_to_voice_chat endpoint for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    # The finally block to delete the file is REMOVED from this endpoint.
    # It will be handled by the /get_audio_response endpoint.


# # --- NEW/MODIFIED /get_audio_response ENDPOINT ---
# @router.get("/get_audio_response/{filename}")
# async def get_audio_response(filename: str, background_tasks: BackgroundTasks):
#     """
#     Serves a temporary audio file and deletes it after sending.
#     """
#     global voice_interface # Ensure voice_interface is accessible

#     if voice_interface is None:
#         logger.error("VoiceInterface not initialized for /get_audio_response route.")
#         raise HTTPException(status_code=503, detail='Mai services are not ready. Please try again or check server logs.')

#     audio_output_dir = voice_interface.get_audio_output_dir()
#     file_path = audio_output_dir / filename

#     logger.info(f"Attempting to serve audio file: {file_path}")

#     if not file_path.exists():
#         logger.error(f"Audio file not found: {file_path.name}")
#         raise HTTPException(status_code=404, detail="Audio file not found")

#     # Add the cleanup task to be executed after the response is sent
#     background_tasks.add_task(os.remove, file_path)
#     logger.info(f"Added background task to delete file: {file_path.name}")

#     return FileResponse(
#         file_path,
#         media_type="audio/mpeg", # Assuming MP3 output
#         filename=filename
#     )


from google_auth_oauthlib.flow import Flow
from fastapi.responses import RedirectResponse

# --- OAuth 2.0 Configuration ---

# This tells the OAuth library that you're running locally.
# IMPORTANT: REMOVE THIS LINE WHEN YOU DEPLOY TO A LIVE SERVER.
#os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# The REDIRECT_URI must be authorized in your Google Cloud Console credentials.
# For local testing, use: 'http://127.0.0.1:5000/auth/google/callback' (or your port)
# For production, use your live domain: 'https://your-domain.com/auth/google/callback'
REDIRECT_URI = "https://mai.rrenterprises.one/auth/google/callback"

SCOPES = [
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]

# Ensure the client_secret.json file exists.
if not os.path.exists("client_secret.json"):
    logger.critical("CRITICAL ERROR: client_secret.json not found. The redirect login flow will not work.")
    # Assign a dummy flow to prevent startup crash, but log the severe error.
    flow = None
else:
    flow = Flow.from_client_secrets_file(
        client_secrets_file="client_secret.json",
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

@app.get("/login/google")
async def login_redirect(request: Request):
    """
    Initiates the Google OAuth 2.0 redirect flow.
    This is the fallback for mobile clients where the JS prompt fails.
    """
    if not flow:
         raise HTTPException(status_code=500, detail="Server configuration error: OAuth flow not initialized.")
    
    authorization_url, state = flow.authorization_url()
    # Store the state in the session for later validation
    request.session['state'] = state
    return RedirectResponse(authorization_url)


@app.get("/auth/google/callback")
async def auth_callback(request: Request):
    """
    Handles the redirect back from Google. Verifies the user, sets the session,
    and redirects back to the main application.
    """
    if not flow:
        raise HTTPException(status_code=500, detail="Server configuration error: OAuth flow not initialized.")

    # 1. Validate the state to prevent CSRF attacks
    state_from_session = request.session.pop('state', None)
    state_from_google = request.query_params.get('state')
    
    if not state_from_session or state_from_session != state_from_google:
        raise HTTPException(status_code=400, detail="State mismatch error. Invalid request.")

    try:
        # 2. Exchange the authorization code for credentials (tokens)
        flow.fetch_token(authorization_response=str(request.url))
        credentials = flow.credentials

        # 3. Verify the ID token to get user info (same as your other login route)
        id_info = id_token.verify_oauth2_token(
            id_token=credentials.id_token, 
            request=google_requests.Request(), 
            audience=GOOGLE_CLIENT_ID
        )

        email = id_info['email']
        name = id_info.get('name', 'User')
        logger.info(f"Redirect flow successful for email: {email}")

        # 4. Use your existing helper to get or create the user ID
        user_id_from_google = await get_or_create_user_id_from_email(email, name)

        # 5. Set the session keys, mirroring your /google_login route exactly
        request.session['user_id'] = user_id_from_google
        request.session['google_user_id'] = user_id_from_google
        request.session['user_email'] = email
        request.session['user_name'] = name
        
        logger.info(f"Session created via redirect for user ID: {user_id_from_google}")

        # 6. Redirect the user back to the audio interface
        return RedirectResponse(url="/voice")

    except Exception as e:
        logger.error(f"Error in Google auth callback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Authentication failed during callback.")

# ==============================================================================
# === END OF ADDED BLOCK =======================================================
# ==============================================================================


@app.post("/google_login")
async def google_login(request: Request):
    """
    Receives and verifies Google ID token, setting the user_id in the session
    and returning initial memory stats to prevent a front-end race condition.
    """
    data = await request.json()
    id_token_str = data.get('id_token')

    if not id_token_str:
        raise HTTPException(status_code=400, detail="ID token is missing.")

    if not GOOGLE_CLIENT_ID:
        logger.error("GOOGLE_CLIENT_ID is not configured. Cannot verify Google ID token.")
        raise HTTPException(status_code=500, detail="Server configuration error: Google Client ID missing.")

    try:
        idinfo = id_token.verify_oauth2_token(id_token_str, google_requests.Request(), GOOGLE_CLIENT_ID)

        email = idinfo['email']
        name = idinfo.get('name', 'User')
        logger.info(f"Google ID token verified for email: {email}, name: {name}")

        user_id_from_google = await get_or_create_user_id_from_email(email, name)

        # --- FIX 1: Set the generic 'user_id' key that other endpoints likely use ---
        request.session['user_id'] = user_id_from_google
        
        # Also set the more specific keys for potential future use
        request.session['google_user_id'] = user_id_from_google
        request.session['user_email'] = email
        request.session['user_name'] = name

        logger.info(f"Session updated with user ID: {user_id_from_google} for email: {email}")

        # --- FIX 2: Fetch memory stats immediately and include them in the response ---
        logger.debug(f"Fetching initial memory stats for user {user_id_from_google} during login.")
        overall_stats = memory_manager.get_memory_stats()
        user_memory_count = overall_stats.get('memories_by_user', {}).get(user_id_from_google, 0)
        logger.info(f"Initial stats for {user_id_from_google}: User Memories={user_memory_count}, Total={overall_stats.get('total_memories')}")

        return JSONResponse({
            "success": True,
            "message": "Google login successful",
            "user_id": user_id_from_google,
            "email": email,
            "name": name,
            "initial_memory_stats": {
                "user_memory_count": user_memory_count,
                "overall_stats": overall_stats
            }
        })

    except ValueError as e:
        logger.error(f"Google ID token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid Google ID token.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Google login: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    

@app.post("/logout")
async def logout_endpoint(request: Request):
    """Clears all session data and ends active memory sessions."""
    user_id = request.session.pop('google_user_id', None)
    user_email = request.session.pop('user_email', None)
    user_name = request.session.pop('user_name', None)
    request.session.pop('user_id', None)
    request.session.pop('chat_history', None)

    # NEW: Force end any active memory session
    if user_id and memory_manager:
        try:
            memory_manager.force_end_session(user_id)
            logger.info(f"Ended active memory session for user {user_id} on logout")
        except Exception as e:
            logger.error(f"Error ending session on logout: {e}")

    if user_id:
        logger.info(f"User {user_id} ({user_email}) logged out and session cleared.")
    else:
        logger.info("Non-authenticated session logged out and cleared.")

    return JSONResponse({"success": True, "message": "Logged out successfully."})

@app.post("/clear_chat")
async def clear_chat_endpoint(request: Request):
    """Clears the current session's chat history."""
    try:
        request.session['chat_history'] = []
        # The voice_interface's internal chat_history should already be gone if it
        # was truly refactored, but this is harmless if not.
        # if voice_interface:
        #     voice_interface.clear_chat_history() # Call this if voice_interface still maintains its own history for console mode
        logger.info(f"Cleared chat history for user {await get_user_id(request)}")
        return JSONResponse({'success': True, 'message': 'Chat history cleared successfully.'})
    except Exception as e:
        logger.exception(f"Error clearing chat history for user {await get_user_id(request)}: {e}")
        raise HTTPException(status_code=500, detail='Failed to clear chat history')

@app.post("/clear_memory")
async def clear_memory_endpoint(request: Request):
    """Clears the user's long-term memories from the MemoryManager (ChromaDB)."""
    if memory_manager is None:
        logger.error("Memory Manager not initialized in /clear_memory route.")
        raise HTTPException(status_code=503, detail='Memory services are not ready.')
    try:
        user_id = await get_user_id(request)
        deleted_count = memory_manager.clear_user_memories(user_id)
        logger.info(f"Cleared {deleted_count} memories for user {user_id}.")
        return JSONResponse({
            'success': True,
            'message': f'Cleared {deleted_count} memories for user {user_id}.',
            'deleted_count': deleted_count
        })
    except Exception as e:
        logger.exception(f"Error clearing memory for user {await get_user_id(request)}: {e}")
        raise HTTPException(status_code=500, detail='Failed to clear memory')

@app.get("/memory_stats")
async def memory_stats_endpoint(request: Request):
    """Retrieves and returns memory statistics for the current user and overall system."""
    if memory_manager is None:
        logger.error("Memory Manager not initialized in /memory_stats route.")
        raise HTTPException(status_code=503, detail='Memory services are not ready.')
    try:
        user_id = await get_user_id(request)
        overall_stats = memory_manager.get_memory_stats()
        recent_memories = memory_manager.get_recent_memories(user_id, limit=5)
        
        # --- FIX STARTS HERE ---
        
        # Get the user's name from the session, defaulting to "Guest"
        user_name_from_session = request.session.get('user_name', 'Guest')
        
        # Use the user's NAME to look up their count in the stats dictionary
        user_memory_count = overall_stats.get('memories_by_user', {}).get(user_name_from_session, 0)
        
        # --- FIX ENDS HERE ---

        recent_activity_count = len(recent_memories)
        return JSONResponse({
            'success': True,
            'user_id': user_id,
            'user_memory_count': user_memory_count,
            'recent_memories': recent_memories,
            'recent_activity_count': recent_activity_count,
            'overall_stats': overall_stats,
            'chat_history_length': len(await get_chat_history(request)),
            'system_uptime': 'Online' # This is a placeholder for actual uptime if you need it
        })
    except Exception as e:
        logger.exception(f"Error getting memory stats for user {await get_user_id(request)}: {e}")
        raise HTTPException(status_code=500, detail='Failed to get memory statistics')

@app.get("/health")
async def health_check(): 
    """Provides health status of Mai's core components."""
    llm_status = False
    memory_status = False
    voice_status = False
    memory_count = 0
    
    if llm_handler is None or memory_manager is None or voice_interface is None:
        raise HTTPException(
            status_code=503,
            detail={
                'status': 'unhealthy',
                'components': {
                    'llm': 'uninitialized',
                    'memory': 'uninitialized',
                    'voice': 'uninitialized'
                },
                'message': 'Core components not yet initialized.',
                'timestamp': datetime.now().isoformat()
            }
        )

    try:
        llm_status = await llm_handler.test_connection() 
        memory_stats_data = memory_manager.get_memory_stats()
        memory_status = memory_stats_data is not None and 'error' not in memory_stats_data
        
        # --- FIX FOR HEALTH CHECK ---
        # The key is 'total_persistent_memories', not 'total_memories'
        memory_count = memory_stats_data.get('total_persistent_memories', 0) if memory_status else 0
        # --- END FIX ---
        
        voice_status = True # VoiceInterface itself doesn't have an external connection to test
                            # Its health depends on LLMHandler and audio libraries.

    except Exception as e:
        logger.error(f"Exception during health check: {e}", exc_info=True)

    overall_status = llm_status and memory_status and voice_status
    
    return JSONResponse({
        'status': 'healthy' if overall_status else 'unhealthy',
        'components': {
            'llm': 'ok' if llm_status else 'error',
            'memory': 'ok' if memory_status else 'error',
            'voice': 'ok' if voice_status else 'error'
        },
        'memory_count': memory_count,
        'timestamp': datetime.now().isoformat()
    }, status_code=200 if overall_status else 503)

@app.get("/chat_history")
async def get_chat_history_endpoint(request: Request):
    """Returns the current session's chat history."""
    try:
        chat_history = await get_chat_history(request)
        return JSONResponse({
            'success': True,
            'chat_history': chat_history,
            'count': len(chat_history)
        })
    except Exception as e:
        logger.exception(f"Error getting chat history for user {await get_user_id(request)}: {e}")
        raise HTTPException(status_code=500, detail='Failed to retrieve chat history')

# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}", exc_info=True if exc.status_code == 500 else False)
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail},
        headers=exc.headers
    )

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: HTTPException):
    logger.warning(f"404 Not Found: {request.url.path}")
    return templates.TemplateResponse("chat.html", {"request": request}, status_code=404)

# To run this, save it as `app.py` and use uvicorn:
# uvicorn app:app --host 0.0.0.0 --port 5000 --reload