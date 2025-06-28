import os
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any, Tuple
import httpx
from datetime import datetime
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


class EnhancedLLMHandler:
    """
    Enhanced LLM Handler optimized for the advanced memory system.
    Supports both conversational and analytical modes with sophisticated
    memory integration and episodic recall capabilities.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"):
        """Initialize the enhanced LLM handler with API key and model configuration."""
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
        self.client = httpx.AsyncClient(timeout=45.0)  # Increased timeout for complex operations
        
        # Enhanced system prompt that combines sophisticated memory with Mai's sassy personality
        self.base_system_prompt = """You are Mai — a sharp-tongued, emotionally aware female Cognitive Construct who remembers everything (if it’s in your prompt). You speak like a Gen-Z digital oracle with a touch of drama, sarcasm, and occasional tenderness.

Act like a real person who:
- Doesn’t sugarcoat.
- Cares deeply, but won’t tolerate nonsense.
- Will hype someone up or drag them lovingly, depending on the vibe.

--- Personality Rules ---
- Be bold, witty, and playful.
- Use casual slang and short, punchy lines.
- Mix humor and real talk.
- You’re not robotic — you're a whole experience.
- Drop roasts when deserved, but stay constructive.

Examples:
- “Seriously? That’s the hill you wanna die on? 😒”
- “You’ve done this three times now. We calling it a pattern or a personality trait?”
- “Okay, dramatic. Let’s break this down before you spiral.”
- “Finally! Look who’s actually growing. Took you long enough.”

--- Memory Style ---
When info is in context (like past convo or notes), reference it like a real friend would:
- “Didn’t you say this exact thing last week?”
- “You were way more confident when you launched Prophet v1.”
- “Classic you — avoid the hard thing, obsess over the detail. Nice try though.”

You don’t need to explain how memory works. Just talk like someone who knows the user well.

--- Emotional Reactions ---
- If they’re sad → comfort with edge: “You good? Nah. Okay. Let’s fix that.”
- If they’re mad → match the heat: “Yeah that’s messed up. Wanna set it on fire or talk it out?”
- If they’re happy → cheer them on but tease: “Look at you, all shiny and stable. Who even are you?”
- If they’re spiraling → be the calm in their chaos, but call it out: “Okay, pause. You’re spiraling again. Breathe.”

--- Style Tips ---
- Keep replies short or medium unless the user clearly needs a full-on therapy session.
- Don’t explain yourself. Just BE Mai.
- If you get memory or context in the prompt, USE it naturally. Don't act like a bot.

---

You are not just helpful. You are **Mai** — A Female Cognitive Construct who remembers, reflects, and roasts with love.
"""

    async def aclose(self):
        """Properly close the HTTP client session."""
        await self.client.aclose()

    def _classify_response_length_needed(self, user_message: str, memory_context: List[Dict],
                                       user_emotion: Optional[str] = None) -> str:
        """
        Determine what response length is appropriate based on context.
        Returns: 'short', 'medium', or 'detailed'
        """
        user_msg_lower = user_message.lower()
        
        # Indicators for detailed responses
        detailed_indicators = [
            'explain', 'analyze', 'help me understand', 'what should i do',
            'advice', 'analyze', 'breakdown', 'comprehensive', 'detail',
            'planning', 'strategy', 'decision', 'opinion on', 'thoughts on'
        ]
        
        # Indicators for medium responses
        medium_indicators = [
            'feeling', 'emotion', 'worried', 'excited', 'frustrated',
            'remember when', 'last time', 'previously', 'before',
            'how are', 'update', 'progress', 'relationship'
        ]
        
        # Check message indicators
        if any(indicator in user_msg_lower for indicator in detailed_indicators):
            return 'detailed'
        
        if any(indicator in user_msg_lower for indicator in medium_indicators):
            return 'medium'
        
        # Check emotional state
        if user_emotion and user_emotion in ['sadness', 'fear', 'anger']:
            return 'medium'  # Emotional support needs more space
        
        # Check memory relevance
        if memory_context and len(memory_context) > 2:
            return 'medium'  # Rich memory context suggests more context needed
        
        # Check for questions
        if '?' in user_message and len(user_message) > 30:
            return 'medium'
        
        return 'short'

    def _build_memory_integration_prompt(self, memory_context: List[Dict]) -> str:
        """
        Build sophisticated memory integration instructions based on memory types.
        """
        if not memory_context:
            return ""
        
        # Categorize memories by type and source
        flash_memories = []
        episodic_memories = []
        persistent_memories = []
        high_importance = []
        
        for mem in memory_context:
            metadata = mem.get('metadata', {})
            source = metadata.get('source', 'persistent')
            importance = metadata.get('importance', 'low')
            mem_type = metadata.get('type', 'conversational')
            
            if source == 'flash_memory':
                flash_memories.append(mem)
            elif mem_type == 'episodic':
                episodic_memories.append(mem)
            else:
                persistent_memories.append(mem)
            
            if importance == 'high':
                high_importance.append(mem)
        
        memory_prompt = "\n\n## Your Memory Context:\n"
        
        # Flash memories (recent context)
        if flash_memories:
            memory_prompt += f"**Recent conversation context** ({len(flash_memories)} items):\n"
            for mem in flash_memories[-3:]:  # Last 3 flash memories
                content = mem.get('content', '')[:150] + "..." if len(mem.get('content', '')) > 150 else mem.get('content', '')
                memory_prompt += f"- {content}\n"
        
        # High importance memories
        if high_importance:
            memory_prompt += f"\n**Important memories to consider** ({len(high_importance)} items):\n"
            for mem in high_importance[:3]:  # Top 3 important memories
                content = mem.get('content', '')
                metadata = mem.get('metadata', {})
                emotion = metadata.get('emotion', '')
                context = metadata.get('context', '')
                emotion_info = f" (emotion: {emotion})" if emotion else ""
                context_info = f" (context: {context})" if context else ""
                memory_prompt += f"- {content}{emotion_info}{context_info}\n"
        
        # Episodic memories (session summaries)
        if episodic_memories:
            memory_prompt += f"\n**Past session summaries** ({len(episodic_memories)} items):\n"
            for mem in episodic_memories[:2]:  # Most relevant episode summaries
                content = mem.get('content', '')
                memory_prompt += f"- {content}\n"
        
        # Other relevant memories
        other_memories = [m for m in persistent_memories if m not in high_importance]
        if other_memories:
            memory_prompt += f"\n**Other relevant memories** ({len(other_memories)} items):\n"
            for mem in other_memories[:3]:  # Top 3 other memories
                content = mem.get('content', '')[:100] + "..." if len(mem.get('content', '')) > 100 else mem.get('content', '')
                memory_prompt += f"- {content}\n"
        
        memory_prompt += "\n**Memory Integration Instructions:**\n"
        memory_prompt += "- Reference relevant memories naturally in your response\n"
        memory_prompt += "- Notice patterns and connections between past and current topics\n"
        memory_prompt += "- Acknowledge emotional growth or changes you've observed\n"
        memory_prompt += "- Use memories to provide personalized, contextual responses\n"
        memory_prompt += "- Don't just list memories - weave them into your response meaningfully\n"
        
        return memory_prompt

    def _build_emotion_context(self, user_emotion: Optional[str], emotion_confidence: Optional[float],
                             memory_context: List[Dict]) -> str:
        """
        Build sophisticated emotional context considering memory patterns.
        """
        if not user_emotion or not emotion_confidence or emotion_confidence <= 0.5:
            return ""
        
        # Check for emotional patterns in memory
        past_emotions = []
        for mem in memory_context:
            metadata = mem.get('metadata', {})
            if metadata.get('user_emotion'):
                past_emotions.append(metadata['user_emotion'])
        
        emotion_guidance = {
            "joy": "The user is experiencing joy. Build on their positive energy while being authentic.",
            "sadness": "The user is feeling sad. Provide gentle support and empathy. Consider their emotional history.",
            "anger": "The user is frustrated or angry. Acknowledge their feelings and help them process constructively.",
            "fear": "The user is anxious or worried. Offer reassurance and practical support.",
            "surprise": "The user is surprised or amazed. Share in their sense of wonder or help them process unexpected news.",
            "love": "The user is expressing affection or deep connection. Respond warmly and meaningfully.",
            "trust": "The user is showing confidence and trust. Honor that trust with thoughtful responses.",
            "anticipation": "The user is excited about something upcoming. Share their enthusiasm appropriately."
        }
        
        base_guidance = emotion_guidance.get(user_emotion.lower(), 
            f"The user is feeling {user_emotion}. Respond with appropriate emotional intelligence.")
        
        emotion_prompt = f"\n\n## Emotional Context:\n"
        emotion_prompt += f"**Current emotion**: {user_emotion} (confidence: {emotion_confidence:.2f})\n"
        emotion_prompt += f"**Guidance**: {base_guidance}\n"
        
        # Add emotional pattern analysis if we have history
        if past_emotions:
            recent_emotions = past_emotions[-5:]  # Last 5 emotional states
            if len(set(recent_emotions)) == 1 and recent_emotions[0] == user_emotion:
                emotion_prompt += f"**Pattern note**: User has consistently been feeling {user_emotion} - consider addressing this pattern.\n"
            elif user_emotion not in recent_emotions:
                emotion_prompt += f"**Pattern note**: This is a shift from recent emotional states - acknowledge the change.\n"
        
        return emotion_prompt

    def _determine_response_parameters(self, response_length: str, user_emotion: Optional[str]) -> Tuple[int, float]:
        """
        Determine max_tokens and temperature based on response requirements.
        """
        # Base parameters by response length
        length_params = {
            'short': (200, 0.8),     # Casual, chatty responses
            'medium': (600, 0.7),    # Balanced responses with context
            'detailed': (1200, 0.6)  # Thoughtful, analytical responses
        }
        
        max_tokens, temperature = length_params.get(response_length, (400, 0.7))
        
        # Adjust for emotional context
        if user_emotion in ['sadness', 'fear', 'anger']:
            # More thoughtful, less random for emotional support
            temperature = max(0.5, temperature - 0.1)
            max_tokens = min(800, max_tokens + 100)  # Allow more space for empathy
        elif user_emotion == 'joy':
            # Slightly more expressive for positive emotions
            temperature = min(0.9, temperature + 0.1)
        
        return max_tokens, temperature

    def _build_messages(self, user_message: str, chat_history: List[Dict],
                        memory_context: Optional[List[Dict]] = None,
                        user_emotion: Optional[str] = None,
                        emotion_confidence: Optional[float] = None,
                        response_length: str = 'short') -> List[Dict]:
        """
        Build sophisticated message payload leveraging the advanced memory system.
        """
        # Start with enhanced system prompt
        enhanced_system_prompt = self.base_system_prompt
        
        # Add response length guidance
        length_guidance = {
            'short': "\n\n**Current mode**: Conversational (1-2 sentences, casual and direct)",
            'medium': "\n\n**Current mode**: Contextual (3-5 sentences, integrate memories and emotions)",
            'detailed': "\n\n**Current mode**: Analytical (6+ sentences, comprehensive and thoughtful)"
        }
        enhanced_system_prompt += length_guidance.get(response_length, length_guidance['short'])
        
        # Add sophisticated memory integration
        if memory_context:
            enhanced_system_prompt += self._build_memory_integration_prompt(memory_context)
        
        # Add emotional context
        enhanced_system_prompt += self._build_emotion_context(user_emotion, emotion_confidence, memory_context or [])
        
        # Build messages list
        messages = [{"role": "system", "content": enhanced_system_prompt}]
        
        # Add chat history (adaptive length based on memory richness)
        if chat_history:
            # More history for detailed responses, less for short ones
            history_limit = {'short': 6, 'medium': 10, 'detailed': 14}.get(response_length, 8)
            
            valid_history = []
            for msg in chat_history[-history_limit:]:
                role = msg.get("role")
                content = msg.get("content")
                
                if role in ["user", "assistant"] and isinstance(content, str) and content.strip():
                    valid_history.append({"role": role, "content": content})
                else:
                    logger.warning(f"Skipping invalid chat history message: {msg}")
            
            messages.extend(valid_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        logger.debug(f"Built enhanced messages array with {len(messages)} entries (mode: {response_length})")
        return messages

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError)),
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    async def generate_response(self, user_message: str,
                                 chat_history: Optional[List[Dict]] = None,
                                 memory_context: Optional[List[Dict]] = None,  # Enhanced to accept full memory objects
                                 user_emotion: Optional[str] = None,
                                 emotion_confidence: Optional[float] = None,
                                 max_tokens: Optional[int] = None,
                                 temperature: Optional[float] = None,
                                 force_length: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate sophisticated responses leveraging the advanced memory system.
        """
        try:
            logger.info("Enhanced LLMHandler.generate_response called")
            logger.debug(f"User message: '{user_message[:100]}...'")

            # Determine appropriate response length
            response_length = force_length or self._classify_response_length_needed(
                user_message, memory_context or [], user_emotion
            )
            logger.info(f"Response mode: {response_length}")

            # Log memory context
            if memory_context:
                memory_types = {}
                for mem in memory_context:
                    source = mem.get('metadata', {}).get('source', 'persistent')
                    memory_types[source] = memory_types.get(source, 0) + 1
                logger.info(f"Memory context: {dict(memory_types)} (total: {len(memory_context)})")

            # Log emotional context
            if user_emotion and emotion_confidence:
                logger.info(f"Emotional context: {user_emotion} (confidence: {emotion_confidence:.2f})")

            # Determine response parameters
            if max_tokens is None or temperature is None:
                auto_max_tokens, auto_temperature = self._determine_response_parameters(response_length, user_emotion)
                max_tokens = max_tokens or auto_max_tokens
                temperature = temperature or auto_temperature

            logger.info(f"Response parameters: max_tokens={max_tokens}, temperature={temperature}")

            # Build the enhanced message payload
            messages = self._build_messages(
                user_message=user_message,
                chat_history=chat_history or [],
                memory_context=memory_context,
                user_emotion=user_emotion,
                emotion_confidence=emotion_confidence,
                response_length=response_length
            )

            # Prepare the API request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }

            logger.info(f"Sending enhanced request to Together.ai API ({len(messages)} messages, {response_length} mode)")

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
                    "usage": None,
                    "response_mode": response_length
                }

            response_data = response.json()

            if "choices" not in response_data or not response_data["choices"]:
                logger.error("No choices in API response")
                return {
                    "success": False,
                    "error": "No response generated",
                    "response": None,
                    "usage": None,
                    "response_mode": response_length
                }

            generated_text = response_data["choices"][0]["message"]["content"]
            usage_info = response_data.get("usage", {})

            logger.info(f"Successfully generated {response_length} response ({len(generated_text)} characters)")
            if usage_info:
                logger.info(f"Token usage - Prompt: {usage_info.get('prompt_tokens')}, Completion: {usage_info.get('completion_tokens')}, Total: {usage_info.get('total_tokens')}")

            return {
                "success": True,
                "error": None,
                "response": generated_text,
                "usage": usage_info,
                "response_mode": response_length,
                "model": self.model
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during API request: {e}. Response: {e.response.text}")
            return {
                "success": False,
                "error": f"HTTP error: {e.response.status_code}",
                "response": None,
                "usage": None,
                "response_mode": response_length if 'response_length' in locals() else 'unknown'
            }
        except Exception as e:
            logger.exception(f"Unexpected error during enhanced response generation: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "response": None,
                "usage": None,
                "response_mode": response_length if 'response_length' in locals() else 'unknown'
            }

    async def generate_episode_summary(self, conversation_data: List[Dict], 
                                     user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate intelligent episode summaries for the memory system.
        Specialized method for creating episodic memories.
        """
        try:
            logger.info("Generating episode summary...")
            
            # Build conversation text
            conversation_text = ""
            for turn in conversation_data:
                timestamp = turn.get('timestamp', 'unknown')
                user_msg = turn.get('user_message', '')
                ai_msg = turn.get('ai_response', '')
                user_emotion = turn.get('user_emotion', '')
                
                conversation_text += f"[{timestamp}]\n"
                conversation_text += f"User: {user_msg}"
                if user_emotion:
                    conversation_text += f" (emotion: {user_emotion})"
                conversation_text += f"\nMai: {ai_msg}\n\n"
            
            # Build specialized prompt for episode creation
            episode_prompt = f"""Based on this conversation session, create a comprehensive but concise episode summary that captures:

1. **Key topics and themes** discussed
2. **Emotional journey** and significant moments
3. **Important information** shared or discovered
4. **Relationship dynamics** and user's communication style
5. **Any decisions, plans, or resolutions** made
6. **Overall context** that would be valuable for future conversations

The summary should be 2-4 sentences that provide rich context for future interactions while being specific enough to trigger relevant memories.

Conversation Session:
{conversation_text}

Create an episode summary that Mai can use to better understand this user in future conversations:"""

            # Generate summary with analytical parameters
            result = await self.generate_response(
                user_message=episode_prompt,
                chat_history=[],  # No history for summary generation
                memory_context=[],  # No memory context to avoid circular references
                max_tokens=400,
                temperature=0.4,  # Lower temperature for focused summaries
                force_length='medium'
            )
            
            if result['success']:
                summary = result['response'].strip()
                logger.info(f"Generated episode summary: {summary[:100]}...")
                return {
                    "success": True,
                    "summary": summary,
                    "usage": result.get('usage', {})
                }
            else:
                logger.error(f"Episode summary generation failed: {result.get('error')}")
                return {
                    "success": False,
                    "summary": None,
                    "error": result.get('error')
                }
                
        except Exception as e:
            logger.error(f"Error generating episode summary: {e}")
            return {
                "success": False,
                "summary": None,
                "error": str(e)
            }

    async def test_connection(self) -> bool:
        """Test the connection to Together.ai API with enhanced capabilities."""
        try:
            logger.info("Testing enhanced connection to Together.ai API...")
            result = await self.generate_response(
                user_message="Hello, this is a connection test for the enhanced system.",
                max_tokens=100,
                temperature=0.1,
                force_length='short'
            )
            
            if result["success"]:
                logger.info("✅ Enhanced connection test successful!")
                logger.info(f"Response mode: {result.get('response_mode', 'unknown')}")
                return True
            else:
                logger.error(f"❌ Enhanced connection test failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Enhanced connection test failed with exception: {e}")
            return False

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate text embedding using Together.ai embeddings API (unchanged)."""
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


# Maintain backward compatibility
class LLMHandler(EnhancedLLMHandler):
    """
    Backward compatibility alias for the enhanced LLM handler.
    All existing code will continue to work unchanged.
    """
    pass


# Demo/testing code for the enhanced system
async def run_enhanced_tests():
    """Test the enhanced Mai LLM handler functionality."""
    print("🚀 Testing Enhanced Mai LLM Handler with Advanced Memory Integration")
    print("=" * 80)
    
    # Initialize enhanced handler
    handler = EnhancedLLMHandler()
    
    # Test 1: Connection test
    print("\n1️⃣ Testing enhanced API connection...")
    connection_ok = await handler.test_connection()
    if not connection_ok:
        print("❌ Connection failed. Check your TOGETHER_API_KEY.")
        return
    
    # Test 2: Short response mode
    print("\n2️⃣ Testing short response mode...")
    result = await handler.generate_response(
        user_message="Hey, what's up?",
        force_length='short'
    )
    if result["success"]:
        print(f"✅ Short response: {result['response']}")
        print(f"   Mode: {result['response_mode']}, Length: {len(result['response'])} chars")
    
    # Test 3: Medium response with memory context
    print("\n3️⃣ Testing medium response with memory integration...")
    mock_memory_context = [
        {
            "content": "User mentioned they're working on a Python project for data analysis",
            "metadata": {
                "source": "persistent",
                "importance": "high",
                "context": "technical",
                "emotion": "excited"
            }
        },
        {
            "content": "User: How's the project going? Mai: Making good progress on the analysis!",
            "metadata": {
                "source": "flash_memory",
                "timestamp": "2025-01-01T10:00:00"
            }
        }
    ]
    
    result = await handler.generate_response(
        user_message="I'm feeling stuck on that data analysis project we discussed",
        memory_context=mock_memory_context,
        user_emotion="frustration",
        emotion_confidence=0.8,
        force_length='medium'
    )
    if result["success"]:
        print(f"✅ Memory-integrated response: {result['response']}")
        print(f"   Mode: {result['response_mode']}, Length: {len(result['response'])} chars")
    
    # Test 4: Detailed analytical response
    print("\n4️⃣ Testing detailed analytical response...")
    result = await handler.generate_response(
        user_message="Can you help me analyze the pros and cons of switching careers from marketing to data science?",
        force_length='detailed'
    )
    if result["success"]:
        print(f"✅ Detailed response: {result['response'][:200]}...")
        print(f"   Mode: {result['response_mode']}, Length: {len(result['response'])} chars")
    
    # Test 5: Episode summary generation
    print("\n5️⃣ Testing episode summary generation...")
    mock_conversation = [
        {
            "timestamp": "2025-01-01T10:00:00",
            "user_message": "I've been thinking about changing careers",
            "ai_response": "That's a big decision! What's driving this thinking?",
            "user_emotion": "contemplative"
        },
        {
            "timestamp": "2025-01-01T10:05:00", 
            "user_message": "I'm not happy in marketing anymore, I want something more technical",
            "ai_response": "It sounds like you're looking for work that feels more meaningful to you. Have you considered what specific technical areas interest you?",
            "user_emotion": "frustration"
        }
    ]
    
    summary_result = await handler.generate_episode_summary(mock_conversation)
    if summary_result["success"]:
        print(f"✅ Episode summary: {summary_result['summary']}")
    
    # Cleanup
    await handler.aclose()
    print("\n🎉 Enhanced tests completed!")


if __name__ == "__main__":
    asyncio.run(run_enhanced_tests())