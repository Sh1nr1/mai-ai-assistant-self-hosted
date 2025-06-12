"""
Mai - Emotionally Intelligent AI Assistant
Flask web application that coordinates LLM and memory systems
"""

import os
import json
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from typing import List, Dict, Optional
import logging

# Import Mai's core components
# Assuming llm_handler.py and memory_manager.py are in the same directory
from llm_handler import LLMHandler
# You'll need to create a basic MemoryManager class if you haven't already.
# For now, it will simply pass or raise a NotImplementedError.
try:
    from memory_manager import MemoryManager
except ImportError:
    logging.warning("MemoryManager not found. Using a placeholder class. Ensure memory_manager.py exists and is correctly implemented.")
    class MemoryManager:
        def __init__(self, *args, **kwargs):
            logging.info("Placeholder MemoryManager initialized.")
            # This might raise an error if your actual MemoryManager expects certain init args.
            # Adjust if your real MemoryManager needs specific setup.
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

# Initialize Flask app
app = Flask(__name__)
# IMPORTANT: Change this secret key in production!
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'mai-secret-key-change-in-production')

# Global instances (initialized at application startup or on first request)
llm_handler: Optional[LLMHandler] = None
memory_manager: Optional[MemoryManager] = None

# --- Configuration for API Key ---
# Option 1: Provide API key directly here for development (LESS SECURE FOR PROD)
# TOGETHER_API_KEY_DIRECT = "744909d3560953a8d2a7edf6e44ebbebf21eebe4c5b03a6191dc99ccce660004"
TOGETHER_API_KEY_DIRECT = "744909d3560953a8d2a7edf6e44ebbebf21eebe4c5b03a6191dc99ccce660004" # Set to None to force environment variable or for deployment

# Option 2: Will fall back to environment variable TOGETHER_API_KEY
# if TOGETHER_API_KEY_DIRECT is None or not set.
# This logic is handled within the initialize_mai_components function.
# --- End API Key Configuration ---

def initialize_mai_components(api_key: Optional[str] = None):
    """
    Initialize Mai's core components (LLM handler and memory manager).
    This function will be called on the first request to ensure components are ready.
    
    Args:
        api_key: Optional API key to pass to LLMHandler. If None, LLMHandler
                 will look for the TOGETHER_API_KEY environment variable.
    """
    global llm_handler, memory_manager
    
    if llm_handler is None:
        try:
            logger.info("Initializing LLM handler...")
            # Pass the determined API key to LLMHandler
            llm_handler = LLMHandler(api_key=api_key) 
            logger.info("LLM handler initialized successfully.")
        except ValueError as ve: # Catch specific ValueError from LLMHandler if API key is missing
            logger.critical(f"CRITICAL ERROR: Failed to initialize LLM handler - {ve}")
            raise RuntimeError("API key is missing or invalid. Please set TOGETHER_API_KEY or provide it directly.") from ve
        except Exception as e:
            logger.critical(f"CRITICAL ERROR: Unexpected error initializing LLM handler: {e}", exc_info=True)
            raise RuntimeError("An unexpected error occurred during LLM setup.") from e
    
    if memory_manager is None:
        try:
            logger.info("Initializing memory manager...")
            # If your MemoryManager needs an API key or other init args, pass them here.
            # Example: memory_manager = MemoryManager(api_key=api_key, db_path='my_db.sqlite')
            memory_manager = MemoryManager() 
            logger.info("Memory manager initialized successfully.")
        except Exception as e:
            logger.critical(f"CRITICAL ERROR: Failed to initialize memory manager: {e}", exc_info=True)
            raise RuntimeError("An unexpected error occurred during Memory Manager setup.") from e

# Use Flask's `before_request` to ensure components are initialized
# This is a common pattern for "lazy" initialization in Flask
@app.before_request
def before_first_request():
    """Ensure Mai components are initialized before handling any request."""
    # Only initialize if they haven't been already
    if llm_handler is None or memory_manager is None:
        logger.info("First request detected. Initializing Mai components...")
        try:
            initialize_mai_components(api_key=TOGETHER_API_KEY_DIRECT)
            logger.info("Mai components initialized for first request.")
        except RuntimeError as e:
            logger.error(f"Failed to initialize Mai components on first request: {e}")
            # If initialization fails, we might want to return an error page or similar
            # For now, just re-raise to show the error
            raise # This will cause a 500 error, which is caught by @app.errorhandler(500)

def get_user_id() -> str:
    """Get or create a unique user ID for session management"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        logger.info(f"Created new user session: {session['user_id']}")
    return session['user_id']

def get_chat_history() -> List[Dict]:
    """Get chat history from session"""
    if 'chat_history' not in session:
        session['chat_history'] = []
    return session['chat_history']

def add_to_chat_history(role: str, content: str, max_history: int = 20):
    """Add message to chat history with size limit"""
    chat_history = get_chat_history()
    
    chat_history.append({
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat()
    })
    
    # Keep only recent messages to prevent session from growing too large
    if len(chat_history) > max_history:
        # This creates a new list, which is fine for small histories.
        # For very large histories, consider a more efficient deque or database.
        session['chat_history'] = chat_history[-max_history:]
    else:
        session['chat_history'] = chat_history # Ensure session variable is updated

# --- Flask Routes ---
@app.route('/')
def index():
    """Main chat interface"""
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages from the user"""
    # Components are guaranteed to be initialized by @app.before_request
    assert llm_handler is not None and memory_manager is not None, "Mai components not initialized!"

    try:
        # Get request data
        data = request.get_json()
        if not data or 'message' not in data:
            logger.warning("Received chat request with no message data.")
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        user_message = data['message'].strip()
        if not user_message:
            logger.warning("Received empty user message.")
            return jsonify({
                'success': False,
                'error': 'Empty message'
            }), 400
        
        # Get user context
        user_id = get_user_id()
        chat_history = get_chat_history()
        
        logger.info(f"User {user_id}: Processing message: '{user_message[:70]}{'...' if len(user_message) > 70 else ''}'")
        
        # Retrieve relevant memories
        logger.debug("Retrieving relevant memories...") # Changed to debug
        memory_context = memory_manager.retrieve_memories(
            query=user_message,
            user_id=user_id,
            max_memories=5
        )
        logger.info(f"Retrieved {len(memory_context)} relevant memories for user {user_id}.")
        
        # Generate response using LLM
        logger.debug("Generating response using LLM...") # Changed to debug
        llm_response = llm_handler.generate_response(
            user_message=user_message,
            chat_history=chat_history,
            memory_context=memory_context
        )
        
        if not llm_response['success']:
            logger.error(f"LLM generation failed for user {user_id}: {llm_response.get('error', 'Unknown error')}")
            return jsonify({
                'success': False,
                'error': 'Failed to generate response',
                'message': llm_response.get('response', 'Sorry, I encountered an error generating a response.')
            }), 500
        
        ai_response = llm_response['response']
        logger.info(f"Mai response: '{ai_response[:70]}{'...' if len(ai_response) > 70 else ''}'")
        
        # Store conversation in long-term memory
        logger.debug("Storing conversation in memory...") # Changed to debug
        stored_memories = memory_manager.store_conversation(
            user_message=user_message,
            ai_response=ai_response,
            user_id=user_id
        )
        logger.info(f"Stored {stored_memories} new memories for user {user_id}.")
        
        # Add to short-term chat history
        add_to_chat_history('user', user_message)
        add_to_chat_history('assistant', ai_response)
        
        # Return response
        return jsonify({
            'success': True,
            'message': ai_response,
            'metadata': {
                'memories_used': len(memory_context),
                'memories_stored': stored_memories,
                'model': llm_response.get('model', 'unknown'),
                'usage': llm_response.get('usage', {})
            }
        })
        
    except Exception as e:
        logger.exception(f"Unhandled error in chat endpoint for user {get_user_id()}: {e}") # Use exception for full traceback
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Sorry, I encountered an unexpected error. Please try again.'
        }), 500

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """Clear current chat session (but keep long-term memories)"""
    try:
        session['chat_history'] = []
        logger.info(f"Cleared chat history for user {get_user_id()}")
        
        return jsonify({
            'success': True,
            'message': 'Chat history cleared successfully.'
        })
        
    except Exception as e:
        logger.exception(f"Error clearing chat history for user {get_user_id()}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to clear chat history'
        }), 500

@app.route('/clear_memory', methods=['POST'])
def clear_memory():
    """Clear all long-term memories for the current user"""
    # Components are guaranteed to be initialized by @app.before_request
    assert memory_manager is not None, "Memory Manager not initialized!"

    try:
        user_id = get_user_id()
        deleted_count = memory_manager.clear_user_memories(user_id)
        
        logger.info(f"Cleared {deleted_count} memories for user {user_id}.")
        
        return jsonify({
            'success': True,
            'message': f'Cleared {deleted_count} memories for user {user_id}.',
            'deleted_count': deleted_count
        })
        
    except Exception as e:
        logger.exception(f"Error clearing memory for user {get_user_id()}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to clear memory'
        }), 500

@app.route('/memory_stats')
def memory_stats_endpoint():
    """Get memory statistics for the current user and overall."""
    # Components are guaranteed to be initialized by @app.before_request
    assert memory_manager is not None, "Memory Manager not initialized!"

    try:
        user_id = get_user_id()
        
        overall_stats = memory_manager.get_memory_stats()
        recent_memories = memory_manager.get_recent_memories(user_id, limit=5)
        
        user_memory_count = overall_stats.get('memories_by_user', {}).get(user_id, 0)
        recent_activity_count = len(recent_memories)
        return jsonify({
            'success': True,
            'user_id': user_id,
            'user_memory_count': user_memory_count,
            'recent_memories': recent_memories,
            'recent_activity_count': recent_activity_count,
            'overall_stats': overall_stats,
            'chat_history_length': len(get_chat_history()),
            'system_uptime': 'Online'
        })
        
    except Exception as e:
        logger.exception(f"Error getting memory stats for user {get_user_id()}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get memory statistics'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint to verify component status."""
    # Components are guaranteed to be initialized by @app.before_request
    assert llm_handler is not None and memory_manager is not None, "Mai components not initialized!"

    llm_status = False
    memory_status = False
    memory_count = 0
    
    try:
        llm_status = llm_handler.test_connection()
        memory_stats_data = memory_manager.get_memory_stats()
        memory_status = memory_stats_data is not None and 'error' not in memory_stats_data
        memory_count = memory_stats_data.get('total_memories', 0) if memory_status else 0

    except Exception as e:
        logger.error(f"Exception during health check: {e}", exc_info=True)
        # Status remains False if an exception occurs

    overall_status = llm_status and memory_status
    
    return jsonify({
        'status': 'healthy' if overall_status else 'unhealthy',
        'components': {
            'llm': 'ok' if llm_status else 'error',
            'memory': 'ok' if memory_status else 'error'
        },
        'memory_count': memory_count,
        'timestamp': datetime.now().isoformat()
    }), 200 if overall_status else 503

@app.route('/chat_history')
def get_chat_history_endpoint():
    """Get current chat history for the session."""
    try:
        chat_history = get_chat_history()
        return jsonify({
            'success': True,
            'chat_history': chat_history,
            'count': len(chat_history)
        })
        
    except Exception as e:
        logger.exception(f"Error getting chat history for user {get_user_id()}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to retrieve chat history'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors (Page Not Found)."""
    logger.warning(f"404 Not Found: {request.path}")
    return render_template('chat.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors (Internal Server Error)."""
    logger.exception(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'Sorry, something went wrong on our end. Please try again later.'
    }), 500

if __name__ == '__main__':
    # Determine API key for initialization
    # Priority: Direct value in app.py -> Environment variable
    final_api_key = TOGETHER_API_KEY_DIRECT
    if not final_api_key:
        final_api_key = os.getenv('TOGETHER_API_KEY')

    # If no API key found, print error and exit early
    if not final_api_key:
        logger.critical("CRITICAL ERROR: TOGETHER_API_KEY is not set. "
                        "Please set the environment variable or provide it directly in app.py.")
        exit(1) # Exit immediately if API key is not available

    # Set up environment for Flask app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development' # Flask's default debug behavior
    
    logger.info("Starting Mai - Emotionally Intelligent AI Assistant")
    logger.info(f"Running on port {port}, debug mode: {debug}")
    
    try:
        # IMPORTANT: If you prefer to initialize components *at startup* rather than
        # on the first request (which is usually better for performance but means
        # the app won't start if LLM/Memory fail), uncomment the line below.
        # initialize_mai_components(api_key=final_api_key)

        # Pass the determined API key to Flask's global context if needed by other parts,
        # or ensure initialize_mai_components uses it during lazy init.
        # For this setup, initialize_mai_components is called with final_api_key via before_request.
        
        # Run Flask app
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug,
            threaded=True
        )
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        exit(1)