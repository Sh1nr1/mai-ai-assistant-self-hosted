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

# Import Mai's core components
from llm_handler import LLMHandler
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
        def store_conversation(self, user_message: str, ai_response: str, user_id: str) -> int:
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

TOGETHER_API_KEY_DIRECT = os.getenv("TOGETHER_API_KEY")
AUDIO_OUTPUT_DIR = Path("audio_output")
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronously initialize Mai's core components at startup
    and gracefully shut down resources at application exit.
    """
    global llm_handler, memory_manager, voice_interface
    
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

        # 2. Initialize Memory Manager
        logger.info("Initializing memory manager...")
        memory_manager = MemoryManager() 
        logger.info("Memory manager initialized successfully.")

        # 3. Initialize Voice Interface
        logger.info("Initializing voice interface...")
        if llm_handler is None:
            raise RuntimeError("LLMHandler was not initialized, cannot initialize VoiceInterface.")
        voice_interface = VoiceInterface(llm_handler=llm_handler)
        logger.info("Voice interface initialized successfully.")

    except ValueError as ve:
        logger.critical(f"CRITICAL ERROR: API key is missing or invalid - {ve}")
        raise RuntimeError(f"API key missing or invalid: {ve}") from ve
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Unexpected error during component initialization: {e}", exc_info=True)
        raise RuntimeError(f"An unexpected error occurred during Mai component setup: {e}") from e

    logger.info("All Mai components initialized.")
    yield  # This yields control to the application, which will now start processing requests

    # --- Code after yield runs on shutdown ---
    logger.info("FastAPI application is shutting down (via lifespan)...")
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

async def get_user_id(request: Request) -> str:
    if 'user_id' not in request.session:
        request.session['user_id'] = str(uuid.uuid4())
        logger.info(f"Created new user session: {request.session['user_id']}")
    return request.session['user_id']

async def get_chat_history(request: Request) -> List[Dict]:
    if 'chat_history' not in request.session:
        request.session['chat_history'] = []
    return request.session['chat_history']

async def add_to_chat_history(request: Request, role: str, content: str, max_history: int = 20):
    chat_history = await get_chat_history(request)
    chat_history.append({
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat()
    })
    if len(chat_history) > max_history:
        request.session['chat_history'] = chat_history[-max_history:]
    else:
        request.session['chat_history'] = chat_history

# --- FastAPI Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/audio_interface", response_class=HTMLResponse)
async def audio_interface_route(request: Request):
    return templates.TemplateResponse("audio_chat.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(
    request: Request,
    message: dict
): 
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
        
        logger.debug("Retrieving relevant memories...")
        memory_context = memory_manager.retrieve_memories(
            query=user_message,
            user_id=user_id,
            max_memories=5
        )
        logger.info(f"Retrieved {len(memory_context)} relevant memories for user {user_id}.")
        
        logger.debug("Generating response using LLM...")
        llm_response = await llm_handler.generate_response(
            user_message=user_message,
            chat_history=chat_history,
            memory_context=memory_context
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
        
        logger.debug("Storing conversation in memory...")
        stored_memories = memory_manager.store_conversation(
            user_message=user_message,
            ai_response=ai_response,
            user_id=user_id
        )
        logger.info(f"Stored {stored_memories} new memories for user {user_id}.")
        
        await add_to_chat_history(request, 'user', user_message)
        await add_to_chat_history(request, 'assistant', ai_response)
        
        return JSONResponse({
            'success': True,
            'message': ai_response,
            'metadata': {
                'memories_used': len(memory_context),
                'memories_stored': stored_memories,
                'model': llm_response.get('model', 'unknown'),
                'usage': llm_response.get('usage', {})
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unhandled error in chat endpoint for user {await get_user_id(request)}: {e}")
        raise HTTPException(
            status_code=500, 
            detail='Internal server error', 
            headers={"X-Mai-Error": 'Sorry, I encountered an unexpected error. Please try again.'}
        )

@app.post("/voice_chat")
async def voice_chat_endpoint(
    request: Request,
    audio: UploadFile = File(...)
): 
    if voice_interface is None:
        logger.error("VoiceInterface not initialized in /voice_chat route.")
        raise HTTPException(status_code=503, detail='Mai voice services are not ready. Please try again or check server logs.')

    temp_audio_file_path_str: Optional[str] = None
    try:
        # Save uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio_file:
            contents = await audio.read()
            temp_audio_file.write(contents)
            temp_audio_file_path_str = temp_audio_file.name
        
        logger.info(f"Saved temporary audio file: {temp_audio_file_path_str}")

        audio_file_path_obj = Path(temp_audio_file_path_str)

        transcription_result = await voice_interface.transcribe_audio_file(audio_file_path_obj)

        if transcription_result['success']:
            user_input = transcription_result['transcribed_text']
            if not user_input:
                logger.warning("Transcribed audio was empty.")
                return JSONResponse({"status": "no_speech", "message": "No speech detected or clear text extracted."})

            interaction_result = await voice_interface.process_voice_interaction(user_input=user_input)

            if interaction_result['success']:
                audio_filename_for_frontend = Path(interaction_result['audio_path']).name if interaction_result['audio_path'] else None

                return JSONResponse({
                    "status": "success",
                    "user_input": interaction_result['user_input'],
                    "mai_response": interaction_result['mai_response'],
                    "audio_filename": audio_filename_for_frontend
                })
            else:
                logger.error(f"LLM interaction failed: {interaction_result.get('error', 'Unknown error')}")
                raise HTTPException(
                    status_code=500, 
                    detail=interaction_result.get('error', 'Failed to get a response from Mai.'), 
                    headers={"X-Mai-Response": interaction_result.get('mai_response', "I'm sorry, I encountered an error generating my response.")}
                )
        else:
            logger.error(f"Transcription failed: {transcription_result.get('error', 'Unknown transcription error')}")
            raise HTTPException(
                status_code=500, 
                detail=transcription_result.get('error', 'Failed to transcribe audio.'), 
                headers={"X-Mai-Response": "I'm sorry, I couldn't understand what you said."}
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error in voice_chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if temp_audio_file_path_str and os.path.exists(temp_audio_file_path_str):
            os.remove(temp_audio_file_path_str)
            logger.info(f"Removed temporary audio file: {temp_audio_file_path_str}")

@app.get("/get_audio_response/{filename}")
async def get_audio_response_endpoint(filename: str):
    file_path = AUDIO_OUTPUT_DIR / filename

    # Security check to prevent directory traversal
    if not file_path.resolve().parent == AUDIO_OUTPUT_DIR.resolve():
        logger.warning(f"Attempted to access file outside audio_output directory: {filename}")
        raise HTTPException(status_code=400, detail='Invalid file path')

    if not file_path.is_file():
        logger.error(f"Audio file not found: {filename}")
        raise HTTPException(status_code=404, detail='Audio file not found')
    
    return FileResponse(path=file_path, media_type="audio/mpeg")

@app.post("/clear_chat")
async def clear_chat_endpoint(request: Request):
    try:
        request.session['chat_history'] = []
        if voice_interface:
            voice_interface.clear_chat_history()
        logger.info(f"Cleared chat history for user {await get_user_id(request)}")
        return JSONResponse({'success': True, 'message': 'Chat history cleared successfully.'})
    except Exception as e:
        logger.exception(f"Error clearing chat history for user {await get_user_id(request)}: {e}")
        raise HTTPException(status_code=500, detail='Failed to clear chat history')

@app.post("/clear_memory")
async def clear_memory_endpoint(request: Request):
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
    if memory_manager is None:
        logger.error("Memory Manager not initialized in /memory_stats route.")
        raise HTTPException(status_code=503, detail='Memory services are not ready.')
    try:
        user_id = await get_user_id(request)
        overall_stats = memory_manager.get_memory_stats()
        recent_memories = memory_manager.get_recent_memories(user_id, limit=5)
        user_memory_count = overall_stats.get('memories_by_user', {}).get(user_id, 0)
        recent_activity_count = len(recent_memories)
        return JSONResponse({
            'success': True,
            'user_id': user_id,
            'user_memory_count': user_memory_count,
            'recent_memories': recent_memories,
            'recent_activity_count': recent_activity_count,
            'overall_stats': overall_stats,
            'chat_history_length': len(await get_chat_history(request)),
            'system_uptime': 'Online'
        })
    except Exception as e:
        logger.exception(f"Error getting memory stats for user {await get_user_id(request)}: {e}")
        raise HTTPException(status_code=500, detail='Failed to get memory statistics')

@app.get("/health")
async def health_check(): 
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
        memory_count = memory_stats_data.get('total_memories', 0) if memory_status else 0
        voice_status = True 

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

# To run this, save it as `main.py` (or `app.py`) and use uvicorn:
# uvicorn app:app --host 0.0.0.0 --port 5000 --reload
