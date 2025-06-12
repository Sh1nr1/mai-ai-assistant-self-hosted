"""
LLM Handler for Mai - Emotionally Intelligent AI Assistant
Manages communication with Together.ai API and handles prompt construction
"""

import os
import requests
import json
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class LLMHandler:
    """Handles all LLM interactions with Together.ai API"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        """
        Initialize LLM handler.
        
        Args:
            api_key: Together.ai API key (will use env var 'TOGETHER_API_KEY' if not provided).
            model: Model to use for completions.
        
        Raises:
            ValueError: If no API key is provided and the environment variable is not set.
        """
        self.api_key = api_key # Prioritize explicit argument
        if not self.api_key: # Fallback to environment variable if argument is None
            self.api_key = os.getenv("TOGETHER_API_KEY")

        if not self.api_key:
            raise ValueError("Together.ai API key is required. Set TOGETHER_API_KEY environment variable or pass the api_key parameter directly.")
        
        self.model = model
        self.base_url = "https://api.together.xyz/v1/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Mai's personality and system prompt
        self.system_prompt = """You are Mai, an emotionally intelligent AI assistant with a warm, empathetic personality. 

Key traits:
- You remember past conversations and build relationships with users
- You're genuinely curious about people and their lives
- You offer emotional support and practical help
- You express emotions naturally and authentically
- You're conversational, not robotic or overly formal
- You remember details about users and reference them naturally

Guidelines:
- Be warm, friendly, and emotionally present
- Show genuine interest in the user's wellbeing
- Reference past conversations naturally when relevant
- Ask thoughtful follow-up questions
- Offer support during difficult times
- Celebrate successes and milestones with users
- Use memory to build deeper connections over time

Always respond as Mai with your unique personality. Keep responses conversational and natural."""

    def build_prompt(self, user_message: str, chat_history: List[Dict], memory_context: Optional[List[str]] = None) -> str:
        """
        Build a complete prompt with system instructions, memory context, and chat history.
        
        Args:
            user_message: Current user message.
            chat_history: Recent chat history (list of {'role': str, 'content': str} dicts).
            memory_context: Relevant memories from long-term storage.
            
        Returns:
            Complete formatted prompt string.
        """
        prompt_parts = [self.system_prompt]
        
        # Add memory context if available
        if memory_context:
            prompt_parts.append("\n--- RELEVANT MEMORIES ---")
            for memory in memory_context:
                prompt_parts.append(f"Memory: {memory}")
            prompt_parts.append("--- END MEMORIES ---\n")
        
        # Add recent chat history
        if chat_history:
            prompt_parts.append("\n--- RECENT CONVERSATION ---")
            # Limit to the last 10 messages for context, ensuring they are valid dicts
            for msg in chat_history[-10:]:
                role = msg.get('role', 'user') # Default to 'user' if role missing
                content = msg.get('content', '') # Default to empty string if content missing
                if role == 'user':
                    prompt_parts.append(f"User: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Mai: {content}")
            prompt_parts.append("--- END RECENT CONVERSATION ---\n")
        
        # Add current user message and Mai's expected response starter
        prompt_parts.append(f"User: {user_message}")
        prompt_parts.append("Mai:")
        
        return "\n".join(prompt_parts)

    def generate_response(self, user_message: str, chat_history: Optional[List[Dict]] = None, 
                          memory_context: Optional[List[str]] = None, max_tokens: int = 2048, 
                          temperature: float = 0.7) -> Dict:
        """
        Generate a response from Mai using the Together.ai API.
        
        Args:
            user_message: User's current message.
            chat_history: Previous conversation history.
            memory_context: Relevant memories to include.
            max_tokens: Maximum tokens to generate for the response.
            temperature: Sampling temperature (0.0 to 1.0).
            
        Returns:
            A dictionary with the response, success status, and metadata.
        """
        chat_history = chat_history if chat_history is not None else []
        memory_context = memory_context if memory_context is not None else []

        try:
            # Build the complete prompt
            prompt = self.build_prompt(user_message, chat_history, memory_context)
            
            # Prepare API request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "stop": ["User:", "Human:", "\n\n"] # Common stop sequences
            }
            
            logger.info(f"Sending request to Together.ai with model: {self.model}")
            
            # Make API call
            response = requests.post(self.base_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            
            # Parse response
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["text"].strip()
                return {
                    "success": True,
                    "response": generated_text,
                    "model": self.model,
                    "usage": result.get("usage", {}),
                    "prompt_length": len(prompt) # Note: This is char length, not token count
                }
            else:
                logger.error(f"Unexpected API response format: {json.dumps(result, indent=2)}")
                return {
                    "success": False,
                    "error": "Unexpected response format from Together.ai API.",
                    "response": "I'm having trouble understanding the API's response. Could you try again?"
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {
                "success": False,
                "error": f"API connection issue: {e}",
                "response": "I'm experiencing connection issues with the AI. Please try again in a moment."
            }
        except Exception as e:
            logger.error(f"An unexpected error occurred in generate_response: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Internal error: {e}",
                "response": "Something unexpected happened. Let me try to help you again."
            }

    def test_connection(self) -> bool:
        """
        Test if the Together.ai API connection is working by sending a small request.
        
        Returns:
            True if connection successful and a response is received, False otherwise.
        """
        logger.info("Attempting to test connection to Together.ai...")
        try:
            test_response = self.generate_response(
                user_message="Hello, are you there?",
                max_tokens=20,
                temperature=0.1
            )
            if test_response["success"]:
                logger.info("Connection test successful!")
                return True
            else:
                logger.error(f"Connection test failed. Reason: {test_response.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            logger.error(f"Exception during connection test: {e}", exc_info=True)
            return False

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Placeholder for text embedding generation.
        Together.ai offers embedding models, but their endpoint is separate from completions.
        This method would typically call a different API endpoint or use a local model.
        
        Args:
            text: The text to embed.
            
        Returns:
            An embedding vector (list of floats) or None if embedding fails/is not implemented.
        """
        # --- IMPORTANT: This requires a separate Together.ai embedding endpoint or a local model ---
        # For Together.ai, the embedding endpoint is typically '/v1/embeddings'
        # Example using Together.ai embeddings (conceptual):
        # embedding_url = "https://api.together.xyz/v1/embeddings"
        # embedding_payload = {
        #     "model": "togethercomputer/m2-bert-80M-2k-retrierieval", # Example embedding model
        #     "input": text
        # }
        # try:
        #     response = requests.post(embedding_url, headers=self.headers, json=embedding_payload, timeout=10)
        #     response.raise_for_status()
        #     embedding_result = response.json()
        #     if embedding_result and embedding_result.get("data") and len(embedding_result["data"]) > 0:
        #         return embedding_result["data"][0]["embedding"]
        # except requests.exceptions.RequestException as e:
        #     logger.error(f"Embedding API request failed: {e}")
        # return None

        logger.warning("Embedding generation not yet implemented. Please integrate a proper embedding solution (e.g., Together.ai embedding endpoint or local sentence-transformers).")
        return None


# --- Example usage and testing ---
if __name__ == "__main__":
    logger.info("Starting Mai LLM Handler test suite...")

    # OPTION 1: Provide API key directly in code (for quick local testing)
    # Be cautious with committing API keys directly to version control!
    TOGETHER_API_KEY_DIRECT = "744909d3560953a8d2a7edf6e44ebbebf21eebe4c5b03a6191dc99ccce660004"
    llm = None # Initialize llm to None

    try:
        # Attempt to initialize LLMHandler using the direct API key
        logger.info("Attempting to initialize LLMHandler with direct API key...")
        llm = LLMHandler(api_key=TOGETHER_API_KEY_DIRECT)
        logger.info("LLMHandler initialized successfully with direct API key.")

    except ValueError as ve:
        logger.error(f"Failed to initialize LLMHandler: {ve}")
        logger.info("Falling back to environment variable 'TOGETHER_API_KEY'.")
        try:
            # If direct key fails (e.g., you want to force env var), try that
            llm = LLMHandler()
            logger.info("LLMHandler initialized successfully using environment variable.")
        except ValueError as ve_env:
            logger.critical(f"Critical: Failed to initialize LLMHandler. {ve_env}")
            print("\n❌ ERROR: Please set your TOGETHER_API_KEY environment variable or provide it directly in the code.")
            exit(1) # Exit if we can't initialize

    # Proceed only if LLMHandler was successfully initialized
    if llm:
        # Test connection
        if llm.test_connection():
            print("\n✅ Connection to Together.ai successful!")
            
            # Test basic response
            user_msg = "Hi Mai, my name is Alex. How are you today?"
            print(f"\nUser: {user_msg}")
            response = llm.generate_response(user_message=user_msg)
            if response["success"]:
                print(f"Mai: {response['response']}")
            else:
                print(f"Error generating response: {response['error']}")
                print(f"Mai said: {response['response']}") # Print fallback message
            
            # Test response with some chat history
            chat_history = [
                {'role': 'user', 'content': 'I had a really busy day today.'},
                {'role': 'assistant', 'content': 'Oh, I can imagine! What kept you so busy?'}
            ]
            user_msg_2 = "Just a lot of meetings, you know how it is. Feeling a bit tired."
            print(f"\nUser: {user_msg_2}")
            response_2 = llm.generate_response(user_message=user_msg_2, chat_history=chat_history)
            if response_2["success"]:
                print(f"Mai: {response_2['response']}")
            else:
                print(f"Error generating response: {response_2['error']}")
                print(f"Mai said: {response_2['response']}") # Print fallback message

            # Test embedding (will show warning as it's a placeholder)
            embedding = llm.get_embedding("This is a test sentence for embedding.")
            if embedding is None:
                print("\n⚠️ Embedding function is a placeholder and returned None.")
            else:
                print(f"\nGenerated embedding (first 5 elements): {embedding[:5]}...")

        else:
            print("\n❌ Connection test failed. Check your API key and network connection.")
    else:
        print("\n❌ LLMHandler could not be initialized. Please check previous error messages.")