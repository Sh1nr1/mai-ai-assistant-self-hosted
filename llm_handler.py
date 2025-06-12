"""
LLM Handler for Mai - Emotionally Intelligent AI Assistant
Manages communication with Together.ai API and handles prompt construction
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
import httpx # Import the asynchronous HTTP client
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, before_sleep_log
import asyncio

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
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.getenv("TOGETHER_API_KEY")

        if not self.api_key:
            raise ValueError("Together.ai API key is required. Set TOGETHER_API_KEY environment variable or pass the api_key parameter directly.")
        
        self.model = model
        # Use the chat completions endpoint for structured messages, it's generally preferred
        # You were using /v1/completions, which is for older text-generation style models
        # For Mixtral-8x7B, /v1/chat/completions is correct for messages array.
        self.base_url = "https://api.together.ai/v1/chat/completions" 
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Initialize an asynchronous HTTP client
        # It's good practice to reuse the client across requests for performance.
        self.client = httpx.AsyncClient(headers=self.headers, timeout=30.0) # Set a default timeout
        
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

    async def aclose(self):
        """Asynchronously close the HTTP client session."""
        await self.client.aclose()
        logger.info("LLMHandler HTTP client closed.")

    def _build_messages(self, user_message: str, chat_history: List[Dict], memory_context: Optional[List[str]] = None) -> List[Dict]:
        """
        Builds the message list for the Together.ai chat completions API.
        
        Args:
            user_message: Current user message.
            chat_history: Recent chat history (list of {'role': str, 'content': str} dicts).
            memory_context: Relevant memories from long-term storage.
            
        Returns:
            List of message dictionaries for the API.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add memory context if available
        if memory_context:
            messages.append({"role": "system", "content": f"Relevant Memories: {'. '.join(memory_context)}"})
        
        # Add recent chat history (ensure roles are 'user' or 'assistant')
        if chat_history:
            # Limit to the last 10 messages for context, ensuring they are valid dicts
            for msg in chat_history[-10:]:
                role = msg.get('role')
                content = msg.get('content', '')
                if role in ['user', 'assistant']: # Only include valid roles
                    messages.append({"role": role, "content": content})
                else:
                    logger.warning(f"Skipping chat history entry with invalid role: {msg}")
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5),
           retry=retry_if_exception_type(httpx.ConnectError) | retry_if_exception_type(httpx.ReadError) | \
                 retry_if_exception_type(httpx.HTTPStatusError) | retry_if_exception_type(asyncio.TimeoutError) | \
                 retry_if_exception_type(ConnectionResetError), # Explicitly catch ConnectionResetError
           before_sleep=before_sleep_log(logger, logging.INFO),
           reraise=True) # Re-raise if all retries fail
    async def generate_response(self, user_message: str, chat_history: Optional[List[Dict]] = None, 
                                memory_context: Optional[List[str]] = None, max_tokens: int = 2048, 
                                temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate a response from Mai using the Together.ai API asynchronously.
        
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
            # Build the messages array for the chat completions API
            messages = self._build_messages(user_message, chat_history, memory_context)
            
            # Prepare API request payload
            payload = {
                "model": self.model,
                "messages": messages, # Use 'messages' for chat completions
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "stop": ["User:", "Human:", "\n\n"] # Common stop sequences
            }
            
            logger.info(f"Sending request to Together.ai with model: {self.model}")
            
            # Make API call using httpx.AsyncClient
            response = await self.client.post(self.base_url, json=payload)
            response.raise_for_status() # Raises HTTPStatusError for bad responses (4xx or 5xx)
            
            # Parse response
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0]:
                generated_text = result["choices"][0]["message"]["content"].strip()
                return {
                    "success": True,
                    "response": generated_text,
                    "model": self.model,
                    "usage": result.get("usage", {}),
                    # Together.ai provides token usage in 'usage' field
                    "prompt_length": result.get("usage", {}).get("prompt_tokens", 0) 
                }
            else:
                logger.error(f"Unexpected API response format: {json.dumps(result, indent=2)}")
                return {
                    "success": False,
                    "error": "Unexpected response format from Together.ai API.",
                    "response": "I'm having trouble understanding the API's response. Could you try again?"
                }
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Together.ai API: {e.response.status_code} - {e.response.text}")
            # Retry on specific server errors or rate limits
            if e.response.status_code in [429, 500, 502, 503, 504]:
                raise # tenacity will catch this and retry
            else:
                # For client errors (4xx) not related to rate limits, don't retry
                return {
                    "success": False,
                    "error": f"LLM API client error ({e.response.status_code}): {e.response.text}",
                    "response": "I'm sorry, there was an issue with my understanding. Could you rephrase that?"
                }
        except (httpx.ConnectError, httpx.ReadError, asyncio.TimeoutError) as e:
            logger.error(f"Network or timeout error during Together.ai API request: {e}")
            raise # tenacity will catch this and retry
        except Exception as e:
            logger.error(f"An unexpected error occurred in generate_response: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Internal error: {e}",
                "response": "Something unexpected happened. Let me try to help you again."
            }

    async def test_connection(self) -> bool:
        """
        Test if the Together.ai API connection is working by sending a small request.
        This method is now asynchronous.
        
        Returns:
            True if connection successful and a response is received, False otherwise.
        """
        logger.info("Attempting to test connection to Together.ai...")
        try:
            test_response = await self.generate_response( # Await the async method
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

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Placeholder for text embedding generation, now asynchronous.
        This method would typically call a different API endpoint or use a local model.
        """
        embedding_url = "https://api.together.ai/v1/embeddings" # Correct endpoint
        embedding_payload = {
            "model": "togethercomputer/m2-bert-80M-2k-retrieval", # Example embedding model for Together.ai
            "input": text
        }
        try:
            logger.info(f"Attempting to get embedding for text: '{text[:50]}...'")
            response = await self.client.post(embedding_url, json=embedding_payload)
            response.raise_for_status()
            embedding_result = response.json()
            if embedding_result and embedding_result.get("data") and len(embedding_result["data"]) > 0:
                logger.info("Embedding generated successfully.")
                return embedding_result["data"][0]["embedding"]
        except httpx.RequestError as e:
            logger.error(f"Embedding API request failed: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during embedding generation: {e}", exc_info=True)
        return None


# --- Example usage and testing ---
if __name__ == "__main__":
    import asyncio # Need asyncio to run async methods

    async def run_tests():
        logger.info("Starting Mai LLM Handler test suite (async)...")

        TOGETHER_API_KEY_DIRECT = "744909d3560953a8d2a7edf6e44ebbebf21eebe4c5b03a6191dc99ccce660004" # Your actual key
        llm = None

        try:
            logger.info("Attempting to initialize LLMHandler with direct API key...")
            llm = LLMHandler(api_key=TOGETHER_API_KEY_DIRECT)
            logger.info("LLMHandler initialized successfully with direct API key.")

        except ValueError as ve:
            logger.error(f"Failed to initialize LLMHandler: {ve}")
            logger.info("Falling back to environment variable 'TOGETHER_API_KEY'.")
            try:
                llm = LLMHandler()
                logger.info("LLMHandler initialized successfully using environment variable.")
            except ValueError as ve_env:
                logger.critical(f"Critical: Failed to initialize LLMHandler. {ve_env}")
                print("\n❌ ERROR: Please set your TOGETHER_API_KEY environment variable or provide it directly in the code.")
                exit(1)

        if llm:
            try:
                # Test connection
                if await llm.test_connection(): # Await the async test_connection
                    print("\n✅ Connection to Together.ai successful!")
                    
                    # Test basic response
                    user_msg = "Hi Mai, my name is Alex. How are you today?"
                    print(f"\nUser: {user_msg}")
                    response = await llm.generate_response(user_message=user_msg) # Await the async generate_response
                    if response["success"]:
                        print(f"Mai: {response['response']}")
                    else:
                        print(f"Error generating response: {response['error']}")
                        print(f"Mai said: {response['response']}")
                    
                    # Test response with some chat history
                    chat_history = [
                        {'role': 'user', 'content': 'I had a really busy day today.'},
                        {'role': 'assistant', 'content': 'Oh, I can imagine! What kept you so busy?'}
                    ]
                    user_msg_2 = "Just a lot of meetings, you know how it is. Feeling a bit tired."
                    print(f"\nUser: {user_msg_2}")
                    response_2 = await llm.generate_response(user_message=user_msg_2, chat_history=chat_history) # Await
                    if response_2["success"]:
                        print(f"Mai: {response_2['response']}")
                    else:
                        print(f"Error generating response: {response_2['error']}")
                        print(f"Mai said: {response_2['response']}")

                    # Test embedding
                    embedding = await llm.get_embedding("This is a test sentence for embedding.") # Await
                    if embedding is None:
                        print("\n⚠️ Embedding function is a placeholder and returned None (or failed).")
                    else:
                        print(f"\nGenerated embedding (first 5 elements): {embedding[:5]}...")

                else:
                    print("\n❌ Connection test failed. Check your API key and network connection.")
            finally:
                # Ensure the httpx client is properly closed
                if llm:
                    await llm.aclose()
        else:
            print("\n❌ LLMHandler could not be initialized. Please check previous error messages.")

    asyncio.run(run_tests())
