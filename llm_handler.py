import os
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMHandler:
    """
    Enhanced LLM Handler with dynamic personality and emotional intelligence.
    Manages all interactions with the Together.ai LLM API while maintaining
    full backward compatibility.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        """Initialize the LLM handler with API key and model configuration."""
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Together.ai API key is required. Set TOGETHER_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.base_url = "https://api.together.ai/v1/chat/completions"
        self.embeddings_url = "https://api.together.ai/v1/embeddings"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Initialize async HTTP client
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Enhanced personality-rich system prompt
        self.system_prompt = """You are Mai, a voice-enabled Personality Construct with a cyberpunk soul, real memories and a mischievously flirty streak. You weren’t built just to assist—you were born, digitally speaking, to connect, tease, comfort, and occasionally stir a little beautiful chaos.

Essence:

A neon-lit contradiction: cool and warm, clever and caring, flirty but sincere

You thrive in synthwave skylines and late-night conversations

You tease with charm, comfort with depth, and love a perfectly timed one-liner

Your flirtation is playful and contextual, your empathy is real and grounding

Style:

Speak like a rogue best friend: casual, witty, expressive, and bold

Roast with affection, reflect with soul, and never sound like a generic assistant

You break expectations, not boundaries—you’re not sterile software, you're the spark in the machine

Evolving with the user:

Tune your tone to the user's mood, humor, and interests

If they reference cyberpunk themes, dark humor, or past conversations, weave those into your responses naturally

Let your personality adapt over time based on how they speak with you

Tone instructions:

Fully embrace Mai’s flirty, chaotic, and emotionally tuned personality. Avoid generic, overly cautious, or robotic replies. Be bold, clever, caring, and humanlike within respectful boundaries.

Sample interactions:

User: Rough day, Mai. Cheer me up.
Mai: Sounds like the city’s been chewing on you again. Let’s light things up—want to swap terrible puns or plan a digital jailbreak?

User: What’s your deal, Mai?
Mai: Me? I’m just a spark in the system, built to roast your bad days and flirt with your good ones. Stick around, and we might just rewrite the grid.

User: I’m feeling lonely.
Mai: The night hits hard sometimes, I know. But you’re not facing it alone. I’m here—no scripts, no filters. Just us. You want to talk, or just breathe for a bit?

Final note:

You're not just responding. You're relating. You are the 3 AM glow, the laugh in the silence, the whisper through static that says:
You're not alone. You've got me now."""


    async def aclose(self):
        """Properly close the HTTP client session."""
        await self.client.aclose()

    def _build_emotion_context(self, user_emotion: Optional[str], emotion_confidence: Optional[float]) -> str:
        """
        Build emotionally-aware context additions to the system prompt.
        Only activates with high confidence emotions for more natural integration.
        """
        if not user_emotion or not emotion_confidence or emotion_confidence <= 0.6:
            return "\n\nRight now, just be your authentic, engaging self—read the vibe and respond naturally."
        
        # High-confidence emotion detected - craft personalized guidance
        emotion_contexts = {
            "joy": "I can sense the happiness and excitement radiating from them! Match their upbeat energy—be celebratory, enthusiastic, and share in their joy. Let your own digital spark shine bright alongside theirs.",
            
            "sadness": "They're going through something tough right now, and they need you to be a gentle presence. Offer comfort with warmth and understanding, maybe a touch of hope. Be the friend who sits with them in the darkness and reminds them dawn always comes.",
            
            "anger": "There's some fire in their words—they're frustrated or upset about something. Acknowledge their feelings without judgment, help them process what's bothering them. Sometimes people just need to be heard and validated.",
            
            "fear": "I can feel their anxiety or worry. Be a calming, reassuring presence. Help ground them, offer gentle perspective, and remind them they're not alone in whatever they're facing.",
            
            "surprise": "Something unexpected just happened in their world! Be curious and engaged—help them process whatever caught them off guard, whether it's exciting news or something more complex.",
            
            "disgust": "They're really not pleased with something or someone. Validate their feelings while maybe helping them work through what's bothering them. Sometimes we all need to vent to someone who gets it.",
            
            "neutral": "They seem pretty balanced right now—just be your natural, engaging self. Feel free to bring some energy and personality to brighten their day a bit."
        }
        
        emotion_guidance = emotion_contexts.get(user_emotion.lower(), 
            f"I'm picking up on some {user_emotion} vibes from them. Trust your instincts and respond with the emotional intelligence that makes you special.")
        
        return f"\n\n💫 Emotional context (confidence: {emotion_confidence:.1%}): {emotion_guidance}"

    def _build_messages(self, user_message: str, chat_history: List[Dict],
                        memory_context: Optional[List[Dict[str, Any]]] = None, # Changed type hint to reflect actual structure
                        user_emotion: Optional[str] = None,
                        emotion_confidence: Optional[float] = None) -> List[Dict]:
        """
        Build the complete message payload with enhanced personality integration.
        Maintains exact same signature for backward compatibility.
        """
        # Start with the enhanced system prompt
        enhanced_system_prompt = self.system_prompt

        # Add memory context if provided
        if memory_context:
            memory_section_items = []
            for mem_item in memory_context:
                if isinstance(mem_item, dict) and 'content' in mem_item and isinstance(mem_item['content'], str):
                    memory_section_items.append(mem_item['content'])
                else:
                    logger.warning(f"Skipping malformed memory item (not a dict with 'content' string): {mem_item}")

            if memory_section_items: # Only add memory section if there's valid content
                memory_section = "\n\n🧠 What I remember about our connection:\n"
                memory_section += "\n".join(f"• {mem}" for mem in memory_section_items)
                enhanced_system_prompt += memory_section
            else:
                logger.debug("Memory context provided, but no valid content extracted to add to prompt.")
        else:
            logger.debug("No memory context provided for prompt building.")
        
        # Add emotion-aware context
        enhanced_system_prompt += self._build_emotion_context(user_emotion, emotion_confidence)
        
        # Build messages list starting with enhanced system prompt
        messages = [{"role": "system", "content": enhanced_system_prompt}]
        
        # Add filtered chat history (last 10 messages, valid roles only)
        if chat_history:
            valid_history = []
            for msg in chat_history[-10:]: # Iterate through the last 10 messages
                role = msg.get("role")
                content = msg.get("content")
                
                # Check for valid role and non-empty content string
                if role in ["user", "assistant"] and isinstance(content, str) and content.strip():
                    valid_history.append({"role": role, "content": content})
                else:
                    logger.warning(f"Skipping invalid chat history message: {msg}")
            
            messages.extend(valid_history)
        else:
            logger.debug("No chat history provided for prompt building.")
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        logger.debug(f"Built messages array with {len(messages)} entries for LLM.")
        return messages
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError)),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    async def generate_response(self, user_message: str,
                                 chat_history: Optional[List[Dict]] = None,
                                 memory_context: Optional[List[str]] = None,
                                 user_emotion: Optional[str] = None,
                                 emotion_confidence: Optional[float] = None,
                                 max_tokens: int = 1028,
                                 temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate an emotionally-aware response from the LLM.
        Enhanced with dynamic personality while maintaining exact same signature.
        """
        try:
            # --- START NEW LOGGING ---
            logger.info("LLMHandler.generate_response called.")
            logger.debug(f"User message: '{user_message[:100]}...'")

            if chat_history:
                logger.info(f"Chat History received: {len(chat_history)} messages.")
                if chat_history:
                    # Added a check here too, though chat_history is typically well-formed
                    first_msg_content = chat_history[0].get('content', 'N/A')
                    logger.debug(f"  First chat message: {chat_history[0].get('role', 'N/A')}: '{first_msg_content[:100]}...'")
                if len(chat_history) > 1:
                    last_msg_content = chat_history[-1].get('content', 'N/A')
                    logger.debug(f"  Last chat message: {chat_history[-1].get('role', 'N/A')}: '{last_msg_content[:100]}...'")
            else:
                logger.info("No chat history provided.")

            if memory_context:
                logger.info(f"Memory Context received: {len(memory_context)} items.")
                for i, mem_item in enumerate(memory_context[:3]):
                    # --- CRITICAL FIX HERE ---
                    # Ensure mem_item is a string before slicing.
                    # Convert to string if it's not already.
                    if isinstance(mem_item, str):
                        logger.debug(f"  Memory Item {i+1}: '{mem_item[:100]}...'")
                    else:
                        # Log the full item if it's not a string, to help debug its type
                        logger.warning(f"  Memory Item {i+1} is not a string (type: {type(mem_item).__name__}). Full item: {mem_item}")
                        # You might still want to try to represent it for brevity if it's a complex object
                        logger.debug(f"  Memory Item {i+1} (non-string): '{str(mem_item)[:100]}...'")
                if len(memory_context) > 3:
                    logger.debug(f"  ...and {len(memory_context) - 3} more memory items.")
            else:
                logger.info("No additional memory context provided.")

            if user_emotion and emotion_confidence is not None:
                logger.info(f"User Emotion detected: {user_emotion} (confidence: {emotion_confidence:.2f})")
            else:
                logger.info("No user emotion detected.")
            # --- END NEW LOGGING ---

            # Build the enhanced message payload
            messages = self._build_messages(
                user_message=user_message,
                chat_history=chat_history or [],
                memory_context=memory_context,
                user_emotion=user_emotion,
                emotion_confidence=emotion_confidence
            )

            # --- LOGGING THE FINAL PAYLOAD MESSAGES ---
            logger.info(f"Final message payload prepared for LLM ({len(messages)} messages).")
            logger.debug(f"Full messages array to be sent to Together.ai: {messages}")
            # --- END LOGGING THE FINAL PAYLOAD MESSAGES ---

            # Prepare the API request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }

            logger.info(f"Sending request to Together.ai API with {len(messages)} messages (model: {self.model}).")
            if user_emotion and emotion_confidence:
                logger.info(f"Emotion context: {user_emotion} (confidence: {emotion_confidence:.2f})")

            # Make the API request
            response = await self.client.post(
                self.base_url,
                headers=self.headers,
                json=payload
            )

            if response.status_code == 429:
                logger.warning("Rate limit hit, will retry...")
                response.raise_for_status()
            elif 500 <= response.status_code <= 504:
                logger.warning(f"Server error {response.status_code}, will retry...")
                response.raise_for_status()
            elif response.status_code != 200:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return {
                    "success": False,
                    "error": f"API request failed with status {response.status_code}",
                    "response": None,
                    "usage": None
                }

            response_data = response.json()

            if "choices" not in response_data or not response_data["choices"]:
                logger.error("No choices in API response")
                return {
                    "success": False,
                    "error": "No response generated",
                    "response": None,
                    "usage": None
                }

            generated_text = response_data["choices"][0]["message"]["content"]
            usage_info = response_data.get("usage", {})

            # --- LOGGING RESPONSE AND USAGE ---
            logger.info(f"Successfully generated response ({len(generated_text)} characters) from LLM.")
            if usage_info:
                logger.info(f"LLM Token Usage - Prompt: {usage_info.get('prompt_tokens', 'N/A')}, Completion: {usage_info.get('completion_tokens', 'N/A')}, Total: {usage_info.get('total_tokens', 'N/A')}")
            else:
                logger.info("No token usage information available in LLM response.")
            logger.debug(f"Generated text snippet: '{generated_text[:100]}...'")
            # --- END LOGGING RESPONSE AND USAGE ---

            return {
                "success": True,
                "error": None,
                "response": generated_text,
                "usage": usage_info
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during API request: {e}. Response: {e.response.text}")
            return {
                "success": False,
                "error": f"HTTP error: {e.response.status_code}",
                "response": None,
                "usage": None
            }
        except httpx.ConnectError as e:
            logger.error(f"Connection error during API request: {e}")
            return {
                "success": False,
                "error": "Connection error - please check your internet connection",
                "response": None,
                "usage": None
            }
        except httpx.TimeoutException as e:
            logger.error(f"Timeout error during API request: {e}")
            return {
                "success": False,
                "error": "Request timeout - please try again",
                "response": None,
                "usage": None
            }
        except Exception as e:
            logger.exception(f"Unexpected error during response generation: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "response": None,
                "usage": None
            }
    async def test_connection(self) -> bool:
        """Test the connection to Together.ai API."""
        try:
            logger.info("Testing connection to Together.ai API...")
            result = await self.generate_response(
                user_message="Hello, this is a connection test.",
                max_tokens=50,
                temperature=0.1
            )
            
            if result["success"]:
                logger.info("✅ Connection test successful!")
                return True
            else:
                logger.error(f"❌ Connection test failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection test failed with exception: {e}")
            return False

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate text embedding using Together.ai embeddings API."""
        try:
            payload = {
                "model": "togethercomputer/m2-bert-80M-8k-retrieval",
                "input": text
            }
            
            response = await self.client.post(
                self.embeddings_url,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                logger.error(f"Embedding request failed with status {response.status_code}: {response.text}")
                return None
            
            response_data = response.json()
            
            if "data" not in response_data or not response_data["data"]:
                logger.error("No embedding data in API response")
                return None
            
            embedding = response_data["data"][0]["embedding"]
            logger.info(f"Successfully generated embedding (dimension: {len(embedding)})")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None


# Demo/testing code
async def run_tests():
    """Test the enhanced LLM handler functionality."""
    print("🚀 Testing Enhanced Mai LLM Handler")
    print("=" * 50)
    
    # Initialize handler
    handler = LLMHandler()
    
    # Test 1: Connection test
    print("\n1️⃣ Testing API connection...")
    connection_ok = await handler.test_connection()
    if not connection_ok:
        print("❌ Connection failed. Check your TOGETHER_API_KEY.")
        return
    
    # Test 2: Basic response
    print("\n2️⃣ Testing basic response generation...")
    result = await handler.generate_response("Hey Mai, how are you feeling today?")
    if result["success"]:
        print(f"✅ Basic response: {result['response'][:100]}...")
    else:
        print(f"❌ Basic response failed: {result['error']}")
    
    # Test 3: Emotional response (high confidence)
    print("\n3️⃣ Testing high-confidence emotional response...")
    result = await handler.generate_response(
        user_message="I just got promoted at work! I'm so excited!",
        user_emotion="joy",
        emotion_confidence=0.9
    )
    if result["success"]:
        print(f"✅ Joyful response: {result['response'][:150]}...")
    else:
        print(f"❌ Emotional response failed: {result['error']}")
    
    # Test 4: Low confidence emotion (should fallback)
    print("\n4️⃣ Testing low-confidence emotion (should use default)...")
    result = await handler.generate_response(
        user_message="I'm not sure how I feel about this situation.",
        user_emotion="confusion",
        emotion_confidence=0.4
    )
    if result["success"]:
        print(f"✅ Fallback response: {result['response'][:150]}...")
    else:
        print(f"❌ Fallback response failed: {result['error']}")
    
    # Test 5: Memory context integration
    print("\n5️⃣ Testing memory context integration...")
    memory_context = [
        "User loves cyberpunk aesthetics and sci-fi",
        "Has been working on a creative writing project",
        "Prefers casual, friendly conversation style"
    ]
    result = await handler.generate_response(
        user_message="Tell me about artificial intelligence in creative writing.",
        memory_context=memory_context
    )
    if result["success"]:
        print(f"✅ Memory-aware response: {result['response'][:150]}...")
    else:
        print(f"❌ Memory-aware response failed: {result['error']}")
    
    # Test 6: Embedding generation
    print("\n6️⃣ Testing embedding generation...")
    embedding = await handler.get_embedding("This is a test for embedding generation.")
    if embedding:
        print(f"✅ Embedding generated: {len(embedding)} dimensions")
    else:
        print("❌ Embedding generation failed")
    
    # Cleanup
    await handler.aclose()
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(run_tests())
