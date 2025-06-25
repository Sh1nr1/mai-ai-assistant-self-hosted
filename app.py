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

TOGETHER_API_KEY_DIRECT = os.getenv("TOGETHER_API_KEY")
AUDIO_OUTPUT_DIR = Path("audio_output")
AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)
FIXED_SINGLE_USER_ID: Optional[str] = "6653d9f1-b272-434f-8f2b-b0a96c35a1d2" #change the global variable to ur own user_id 

# Define MAX_CHAT_HISTORY_LENGTH for session history trimming
MAX_CHAT_HISTORY_LENGTH = 20 # Keep last 20 messages (10 turns) in session

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
        # VoiceInterface itself should not manage chat history or memory context for web interactions
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
    """
    Retrieves or creates a user ID for the current session.
    If FIXED_SINGLE_USER_ID is set globally, it will always return that ID.
    """
    if FIXED_SINGLE_USER_ID:
        # If a fixed user ID is defined, always use it
        return FIXED_SINGLE_USER_ID
    
    # Fallback to session-based UUID generation if no fixed ID is set
    if 'user_id' not in request.session:
        request.session['user_id'] = str(uuid.uuid4())
        logger.info(f"Created new user session: {request.session['user_id']}")
    return request.session['user_id']

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

@app.get("/audio_interface", response_class=HTMLResponse)
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
        # Specific user ID for "Rishi"
        specific_rishi_user_id = "6653d9f1-b272-434f-8f2b-b0a96c35a1d2"

        if user_id == specific_rishi_user_id:
            user_name_for_storage = "Rishi"
            logger.debug(f"Assigned user_name 'Rishi' to user_id: {user_id}")
        # --- End User Name Association Logic ---

        # Analyze sentiment of the user message
        user_emotion, emotion_confidence = await analyze_sentiment(user_message)
        logger.info(f"User {user_id} detected emotion: {user_emotion} (confidence: {emotion_confidence:.2f})")

        logger.debug("Retrieving relevant memories...")
        # Get memory context from the MemoryManager (ChromaDB)
        memory_context = memory_manager.retrieve_memories(
            query=user_message,
            user_id=user_id,
            limit=50
        )
        logger.info(f"Retrieved {len(memory_context)} relevant memories for user {user_id}.")

        logger.debug("Generating response using LLM...")
        llm_response = await llm_handler.generate_response(
            user_message=user_message,
            chat_history=chat_history, # Pass the session's chat history
            memory_context=memory_context, # Pass the retrieved memory context
            user_emotion=user_emotion, # Pass emotion to the LLMHandler
            emotion_confidence=emotion_confidence, # Pass confidence too (optional for LLM, good for logging/memory)
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
                'user_emotion': user_emotion,
                'emotion_confidence': emotion_confidence,
                'ai_emotion': ai_emotion, # Return AI emotion
                'user_name': user_name_for_storage # Also return to frontend for clarity
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
        specific_rishi_user_id = "6653d9f1-b272-434f-8f2b-b0a96c35a1d2"

        if user_id == specific_rishi_user_id:
            user_name_for_storage = "Rishi"
            logger.debug(f"Assigned user_name 'Rishi' to user_id: {user_id} for voice chat.")
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
        specific_rishi_user_id = "6653d9f1-b272-434f-8f2b-b0a96c35a1d2"

        if user_id == specific_rishi_user_id:
            user_name_for_storage = "Rishi"
            logger.debug(f"Assigned user_name 'Rishi' to user_id: {user_id} for text-to-voice chat.")
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
            filename=unique_audio_filename
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

@app.get("/get_audio_response/{filename}")
async def get_audio_response_endpoint(filename: str):
    """Serves generated audio response files."""
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
        memory_count = memory_stats_data.get('total_memories', 0) if memory_status else 0
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