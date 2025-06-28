import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING
import re
import json
import asyncio
from pathlib import Path
from collections import defaultdict, deque
import math

# Type checking imports to avoid circular imports
if TYPE_CHECKING:
    from llm_handler import LLMHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Enhanced Memory Manager with session-end triggered episodes for Mai AI:
    - Flash Memory (short-term in-memory storage)
    - Session-based Episodic Memory (intelligent session summaries via LLM)
    - Memory Linkage (relationship mapping)
    - Advanced sentiment/tone scoring
    - Extended metadata with importance, lifespan, and time context
    """
    
    def __init__(self, collection_name: str = "mai_memories", persist_directory: str = "./mai_memory", 
                 llm_handler: Optional['LLMHandler'] = None):
        """
        Initialize the Enhanced Memory Manager with session-based episodic memory.
        
        Args:
            collection_name: Name of the ChromaDB collection for storing memories
            persist_directory: Directory path for persistent storage
            llm_handler: Optional LLM handler for intelligent summary generation
        """
        try:
            # Ensure persist_directory exists
            self.persist_directory = Path(persist_directory)
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            # Store LLM handler reference
            self.llm_handler = llm_handler

            # Initialize embedding model with caching for better performance
            logger.info("Initializing SentenceTransformer model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB client with enhanced settings
            logger.info(f"Initializing ChromaDB client with persist directory: {self.persist_directory}")
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection with improved metadata indexing
            self.collection_name = collection_name
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Enhanced Mai AI memories with session-based episodic architecture"}
            )

            # User ID to Name/Email Mapping Management
            self.user_id_map_file = self.persist_directory / "user_id_email_map.json"
            self._user_email_to_id_map: Dict[str, str] = {}
            self._user_id_to_name_map: Dict[str, str] = {}
            self._load_user_id_map()

            # =================== SESSION TRACKING SYSTEM ===================
            # Session management for episodic memory
            self._active_sessions: Dict[str, Dict] = {}  # user_id -> session_data
            self._session_timeout_minutes = 30  # Session expires after 30 minutes of inactivity
            self._min_session_interactions = 3  # Minimum interactions before creating episode
            self._session_file = self.persist_directory / "active_sessions.json"
            self._load_active_sessions()

            # =================== FLASH MEMORY SYSTEM ===================
            # Flash Memory: Short-term in-memory storage (last 10-20 conversation turns per user)
            self._flash_memory: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
            self._flash_memory_max_size = 20
            logger.info("Flash Memory system initialized with max size: 20 per user")

            # =================== SENTIMENT & EMOTION ENHANCEMENT ===================
            # Enhanced emotion keywords with scoring weights
            self.emotion_keywords = {
                'joy': {
                    'primary': ['happy', 'excited', 'thrilled', 'delighted', 'ecstatic', 'euphoric'],
                    'secondary': ['cheerful', 'elated', 'joyful', 'blissful', 'jubilant', 'overjoyed'],
                    'tertiary': ['gleeful', 'exuberant', 'radiant', 'beaming', 'buoyant', 'uplifted']
                },
                'sadness': {
                    'primary': ['sad', 'depressed', 'heartbroken', 'devastated', 'crushed', 'shattered'],
                    'secondary': ['melancholy', 'disappointed', 'sorrowful', 'grief', 'mourning', 'despondent'],
                    'tertiary': ['dejected', 'downcast', 'crestfallen', 'disheartened', 'discouraged', 'dismayed']
                },
                'anger': {
                    'primary': ['angry', 'furious', 'rage', 'livid', 'enraged', 'infuriated'],
                    'secondary': ['irritated', 'annoyed', 'frustrated', 'mad', 'irate', 'incensed'],
                    'tertiary': ['outraged', 'indignant', 'resentful', 'bitter', 'hostile', 'aggressive']
                },
                'fear': {
                    'primary': ['scared', 'terrified', 'panic', 'horrified', 'petrified', 'paralyzed'],
                    'secondary': ['afraid', 'anxious', 'worried', 'nervous', 'fearful', 'frightened'],
                    'tertiary': ['alarmed', 'startled', 'shocked', 'stunned', 'insecure', 'uncertain']
                },
                'surprise': {
                    'primary': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'flabbergasted'],
                    'secondary': ['bewildered', 'dumbfounded', 'astounded', 'thunderstruck', 'awestruck'],
                    'tertiary': ['speechless', 'breathless', 'startling', 'jarring', 'unexpected', 'sudden']
                },
                'love': {
                    'primary': ['love', 'adore', 'cherish', 'devoted', 'passionate', 'romantic'],
                    'secondary': ['affection', 'caring', 'tender', 'gentle', 'warm', 'sweet'],
                    'tertiary': ['kind', 'compassionate', 'nurturing', 'protective', 'loyal', 'faithful']
                },
                'trust': {
                    'primary': ['trust', 'confident', 'secure', 'reliable', 'dependable', 'faith'],
                    'secondary': ['belief', 'conviction', 'certainty', 'assurance', 'confidence', 'safety'],
                    'tertiary': ['stable', 'steady', 'solid', 'firm', 'strong', 'honest']
                },
                'anticipation': {
                    'primary': ['excited', 'eager', 'hopeful', 'optimistic', 'looking forward', 'anticipating'],
                    'secondary': ['expecting', 'awaiting', 'planning', 'preparing', 'ready', 'motivated'],
                    'tertiary': ['inspired', 'driven', 'determined', 'focused', 'purposeful', 'meaningful']
                }
            }

            # Enhanced context patterns with importance scoring
            self.context_patterns = {
                'project_planning': {
                    'high_importance': ['project', 'roadmap', 'milestone', 'deadline', 'strategy', 'launch'],
                    'medium_importance': ['plan', 'timeline', 'schedule', 'task', 'deliverable', 'goal'],
                    'low_importance': ['meeting', 'standup', 'checkin', 'update', 'status', 'review']
                },
                'technical': {
                    'high_importance': ['architecture', 'system', 'algorithm', 'database', 'security', 'performance'],
                    'medium_importance': ['code', 'programming', 'api', 'framework', 'deployment', 'testing'],
                    'low_importance': ['debug', 'fix', 'update', 'config', 'setup', 'install']
                },
                'personal': {
                    'high_importance': ['relationship', 'family', 'health', 'life', 'dreams', 'values'],
                    'medium_importance': ['feel', 'think', 'believe', 'experience', 'growth', 'journey'],
                    'low_importance': ['hobby', 'interest', 'routine', 'daily', 'casual', 'chat']
                },
                'emotional': {
                    'high_importance': ['love', 'heartbroken', 'devastated', 'terrified', 'euphoric', 'crisis'],
                    'medium_importance': ['happy', 'sad', 'angry', 'worried', 'excited', 'disappointed'],
                    'low_importance': ['okay', 'fine', 'alright', 'neutral', 'normal', 'usual']
                },
                'business': {
                    'high_importance': ['revenue', 'profit', 'investment', 'funding', 'acquisition', 'ipo'],
                    'medium_importance': ['customer', 'market', 'sales', 'strategy', 'competition', 'growth'],
                    'low_importance': ['meeting', 'email', 'call', 'update', 'report', 'admin']
                },
                'creative': {
                    'high_importance': ['innovation', 'breakthrough', 'vision', 'masterpiece', 'inspiration', 'genius'],
                    'medium_importance': ['design', 'art', 'creative', 'idea', 'concept', 'prototype'],
                    'low_importance': ['sketch', 'draft', 'edit', 'tweak', 'adjust', 'practice']
                },
                'learning': {
                    'high_importance': ['breakthrough', 'epiphany', 'mastery', 'expertise', 'discovery', 'insight'],
                    'medium_importance': ['learn', 'study', 'understand', 'knowledge', 'skill', 'practice'],
                    'low_importance': ['review', 'recap', 'notes', 'reminder', 'reference', 'lookup']
                }
            }

            # Memory type definitions
            self.memory_types = {
                'core': 'Fundamental personality and relationship memories',
                'episodic': 'Session-level summaries and significant events', 
                'volatile': 'Temporary memories with limited lifespan',
                'conversational': 'Regular conversation exchanges',
                'system': 'System-generated insights and analyses'
            }

            # Lifespan definitions (in days)
            self.lifespan_days = {
                'permanent': None,  # Never expires
                'volatile': 30,     # 30 days
                'ephemeral': 7      # 7 days
            }

            logger.info(f"✅ Enhanced MemoryManagerV4 with Session Episodes initialized with collection '{collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced MemoryManagerV4: {e}", exc_info=True)
            raise

    # =================== SESSION MANAGEMENT SYSTEM ===================
    
    def _load_active_sessions(self):
        """Load active sessions from persistent storage."""
        if self._session_file.exists():
            try:
                with open(self._session_file, 'r') as f:
                    self._active_sessions = json.load(f)
                logger.info(f"Loaded {len(self._active_sessions)} active sessions")
                # Clean up expired sessions on load
                self._cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"Error loading active sessions: {e}")
                self._active_sessions = {}
        else:
            logger.info("No active sessions file found. Starting fresh.")

    def _save_active_sessions(self):
        """Save active sessions to persistent storage."""
        try:
            with open(self._session_file, 'w') as f:
                json.dump(self._active_sessions, f, indent=2)
            logger.debug("Saved active sessions to file")
        except Exception as e:
            logger.error(f"Failed to save active sessions: {e}")

    def _cleanup_expired_sessions(self):
        """Clean up expired sessions and create episodes for them."""
        current_time = datetime.now()
        expired_sessions = []
        
        for user_id, session_data in self._active_sessions.items():
            try:
                last_activity = datetime.fromisoformat(session_data['last_activity'])
                if (current_time - last_activity).total_seconds() > (self._session_timeout_minutes * 60):
                    expired_sessions.append(user_id)
            except Exception as e:
                logger.warning(f"Error parsing session time for user {user_id}: {e}")
                expired_sessions.append(user_id)  # Remove malformed sessions
        
        for user_id in expired_sessions:
            asyncio.create_task(self._end_session_and_create_episode(user_id))

    async def _end_session_and_create_episode(self, user_id: str):
        """End a session and create an episode summary using LLM."""
        if user_id not in self._active_sessions:
            return
        
        session_data = self._active_sessions[user_id]
        
        # Check if session has enough interactions to warrant an episode
        if len(session_data['interactions']) < self._min_session_interactions:
            logger.info(f"Session for user {user_id} has insufficient interactions ({len(session_data['interactions'])}), skipping episode creation")
            del self._active_sessions[user_id]
            self._save_active_sessions()
            return
        
        logger.info(f"Creating episode for user {user_id} with {len(session_data['interactions'])} interactions")
        
        try:
            # Generate episode summary using LLM
            episode_summary = await self._generate_episode_summary(session_data)
            
            if episode_summary:
                # Store the episode
                success = self.store_episode_summary(
                    summary_text=episode_summary,
                    user_id=user_id,
                    episode_context=f"session_summary_{session_data['session_id']}",
                    user_name=session_data.get('user_name')
                )
                
                if success:
                    logger.info(f"✅ Successfully created episode for user {user_id}")
                else:
                    logger.error(f"Failed to store episode for user {user_id}")
            else:
                logger.warning(f"LLM failed to generate episode summary for user {user_id}")
        
        except Exception as e:
            logger.error(f"Error creating episode for user {user_id}: {e}")
        
        finally:
            # Remove session regardless of episode creation success
            del self._active_sessions[user_id]
            self._save_active_sessions()

    async def _generate_episode_summary(self, session_data: Dict) -> Optional[str]:
        """Generate an intelligent episode summary using the LLM handler."""
        if not self.llm_handler:
            logger.warning("No LLM handler available for episode summary generation")
            return self._generate_basic_episode_summary(session_data)
        
        try:
            # Prepare conversation history for LLM
            interactions = session_data['interactions']
            conversation_text = ""
            
            for interaction in interactions:
                timestamp = interaction.get('timestamp', 'unknown')
                user_msg = interaction.get('user_message', '')
                ai_msg = interaction.get('ai_response', '')
                
                conversation_text += f"[{timestamp}]\n"
                conversation_text += f"User: {user_msg}\n"
                conversation_text += f"Mai: {ai_msg}\n\n"
            
            # Create prompt for episode summary
            summary_prompt = f"""Based on this conversation session, create a concise but comprehensive episode summary. Focus on:
1. Key topics discussed
2. Important emotional moments or breakthroughs
3. Decisions made or problems solved
4. Any personal information shared
5. The overall arc/flow of the conversation

Conversation:
{conversation_text}

Create a summary that captures the essence of this interaction session in 2-3 sentences, focusing on what would be most important to remember for future conversations."""

            # Generate summary using LLM
            llm_response = await self.llm_handler.generate_response(
                user_message=summary_prompt,
                chat_history=[],  # No chat history for summary generation
                memory_context=[],  # No memory context to avoid circular references
                max_tokens=300,  # Limit summary length
                temperature=0.3  # Lower temperature for more focused summaries
            )
            
            if llm_response['success']:
                summary = llm_response['response'].strip()
                logger.info(f"LLM generated episode summary: {summary[:100]}...")
                return summary
            else:
                logger.error(f"LLM failed to generate summary: {llm_response.get('error')}")
                return self._generate_basic_episode_summary(session_data)
                
        except Exception as e:
            logger.error(f"Error generating LLM episode summary: {e}")
            return self._generate_basic_episode_summary(session_data)

    def _generate_basic_episode_summary(self, session_data: Dict) -> str:
        """Generate a basic episode summary as fallback."""
        interactions = session_data['interactions']
        session_start = session_data['start_time']
        session_end = session_data['last_activity']
        
        # Extract key topics and emotions
        topics = set()
        emotions = set()
        
        for interaction in interactions:
            user_emotion = interaction.get('user_emotion')
            ai_emotion = interaction.get('ai_emotion')
            context = interaction.get('context', 'general')
            
            topics.add(context)
            if user_emotion:
                emotions.add(user_emotion)
            if ai_emotion:
                emotions.add(ai_emotion)
        
        # Create basic summary
        duration_minutes = (datetime.fromisoformat(session_end) - datetime.fromisoformat(session_start)).total_seconds() / 60
        
        summary = f"Conversation session with {len(interactions)} exchanges over {duration_minutes:.0f} minutes. "
        summary += f"Topics: {', '.join(topics)}. "
        
        if emotions:
            summary += f"Emotional tone: {', '.join(emotions)}. "
        
        summary += f"Session from {session_start} to {session_end}."
        
        return summary

    def _start_or_update_session(self, user_id: str, user_message: str, ai_response: str, 
                                user_emotion: Optional[str] = None, ai_emotion: Optional[str] = None,
                                user_name: Optional[str] = None):
        """Start a new session or update an existing one."""
        current_time = datetime.now()
        timestamp = current_time.isoformat()
        
        # Check if user has an active session
        if user_id not in self._active_sessions:
            # Start new session
            session_id = str(uuid.uuid4())
            self._active_sessions[user_id] = {
                'session_id': session_id,
                'start_time': timestamp,
                'last_activity': timestamp,
                'user_name': user_name,
                'interactions': []
            }
            logger.info(f"Started new session {session_id} for user {user_id}")
        else:
            # Check if session has expired (but not cleaned up yet)
            last_activity = datetime.fromisoformat(self._active_sessions[user_id]['last_activity'])
            if (current_time - last_activity).total_seconds() > (self._session_timeout_minutes * 60):
                # Session expired, create episode for old session and start new one
                asyncio.create_task(self._end_session_and_create_episode(user_id))
                
                # Start fresh session
                session_id = str(uuid.uuid4())
                self._active_sessions[user_id] = {
                    'session_id': session_id,
                    'start_time': timestamp,
                    'last_activity': timestamp,
                    'user_name': user_name,
                    'interactions': []
                }
                logger.info(f"Started new session {session_id} for user {user_id} (previous expired)")
        
        # Add interaction to current session
        interaction = {
            'timestamp': timestamp,
            'user_message': user_message,
            'ai_response': ai_response,
            'user_emotion': user_emotion,
            'ai_emotion': ai_emotion,
            'context': self._classify_context(user_message + " " + ai_response)
        }
        
        self._active_sessions[user_id]['interactions'].append(interaction)
        self._active_sessions[user_id]['last_activity'] = timestamp
        
        # Update user name if provided
        if user_name:
            self._active_sessions[user_id]['user_name'] = user_name
        
        self._save_active_sessions()
        logger.debug(f"Updated session for user {user_id} with new interaction")

    def force_end_session(self, user_id: str):
        """Manually end a session and create episode (useful for explicit logout)."""
        if user_id in self._active_sessions:
            asyncio.create_task(self._end_session_and_create_episode(user_id))
            logger.info(f"Manually ended session for user {user_id}")

    def get_active_session_stats(self) -> Dict:
        """Get statistics about active sessions."""
        current_time = datetime.now()
        stats = {
            'total_active_sessions': len(self._active_sessions),
            'sessions_by_user': {},
            'session_durations': []
        }
        
        for user_id, session_data in self._active_sessions.items():
            user_name = session_data.get('user_name', user_id[:8] + "...")
            start_time = datetime.fromisoformat(session_data['start_time'])
            duration_minutes = (current_time - start_time).total_seconds() / 60
            
            stats['sessions_by_user'][user_name] = {
                'interactions': len(session_data['interactions']),
                'duration_minutes': round(duration_minutes, 1),
                'last_activity': session_data['last_activity']
            }
            stats['session_durations'].append(duration_minutes)
        
        return stats

    # =================== USER ID MAPPING (PRESERVED) ===================
    def _load_user_id_map(self):
        """Loads the user_id to email/name mapping from a JSON file."""
        if self.user_id_map_file.exists():
            try:
                with open(self.user_id_map_file, 'r') as f:
                    loaded_map = json.load(f)
                    self._user_email_to_id_map = loaded_map.get('email_to_id', {})
                    self._user_id_to_name_map = loaded_map.get('id_to_name', {})
                logger.info(f"Loaded {len(self._user_email_to_id_map)} email-to-ID mappings")
            except Exception as e:
                logger.error(f"Error loading user ID map: {e}")
                self._user_email_to_id_map = {}
                self._user_id_to_name_map = {}
        else:
            logger.info("User ID map file not found. Starting with empty map.")

    def _save_user_id_map(self):
        """Saves the current user_id to email/name mapping to a JSON file."""
        try:
            with open(self.user_id_map_file, 'w') as f:
                json.dump({
                    'email_to_id': self._user_email_to_id_map,
                    'id_to_name': self._user_id_to_name_map
                }, f, indent=4)
            logger.debug(f"Saved user ID map to {self.user_id_map_file}")
        except Exception as e:
            logger.error(f"Failed to save user ID map: {e}", exc_info=True)

    def get_or_create_user_id(self, email: str, default_name: Optional[str] = None) -> str:
        """Retrieves or creates user ID for email with name mapping."""
        if email in self._user_email_to_id_map:
            user_id = self._user_email_to_id_map[email]
            if default_name and self._user_id_to_name_map.get(user_id) != default_name:
                self.set_user_name(user_id, default_name)
            return user_id
        else:
            new_user_id = str(uuid.uuid4())
            self._user_email_to_id_map[email] = new_user_id
            if default_name:
                self._user_id_to_name_map[new_user_id] = default_name
            self._save_user_id_map()
            logger.info(f"Generated new user ID '{new_user_id}' for email '{email}'")
            return new_user_id

    def set_user_name(self, user_id: str, user_name: str):
        """Sets or updates the human-readable name for a user ID."""
        if self._user_id_to_name_map.get(user_id) != user_name:
            self._user_id_to_name_map[user_id] = user_name
            self._save_user_id_map()
            logger.info(f"Updated name for user_id '{user_id}' to '{user_name}'")

    def get_user_name(self, user_id: str) -> Optional[str]:
        """Retrieves the human-readable name for a user ID."""
        return self._user_id_to_name_map.get(user_id)

    # =================== ENHANCED EMOTION & SENTIMENT ANALYSIS ===================
    def _detect_emotion_with_scores(self, text: str) -> Tuple[Optional[str], Dict[str, float]]:
        """
        Enhanced emotion detection with confidence scoring.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (dominant_emotion, emotion_scores_dict)
        """
        text_lower = text.lower()
        emotion_scores = {}
        
        for emotion, keyword_groups in self.emotion_keywords.items():
            total_score = 0.0
            
            # Weight different keyword groups
            for group_type, keywords in keyword_groups.items():
                weight = {'primary': 1.0, 'secondary': 0.7, 'tertiary': 0.4}.get(group_type, 0.5)
                group_score = sum(weight for keyword in keywords if keyword in text_lower)
                total_score += group_score
            
            if total_score > 0:
                emotion_scores[emotion] = round(total_score, 2)
        
        # Normalize scores to create probabilities
        if emotion_scores:
            total = sum(emotion_scores.values())
            normalized_scores = {k: round(v/total, 3) for k, v in emotion_scores.items()}
            dominant_emotion = max(normalized_scores, key=normalized_scores.get)
            return dominant_emotion, normalized_scores
        
        return None, {}

    def _calculate_sentiment_valence(self, text: str, emotion_scores: Dict[str, float]) -> float:
        """
        Calculate sentiment valence from -1 (negative) to +1 (positive).
        
        Args:
            text: Input text
            emotion_scores: Emotion scores from detection
            
        Returns:
            Sentiment valence score (-1 to +1)
        """
        # Emotion-based valence mapping
        emotion_valence = {
            'joy': 0.8, 'love': 0.9, 'trust': 0.6, 'anticipation': 0.5,
            'sadness': -0.7, 'fear': -0.6, 'anger': -0.8, 'surprise': 0.1
        }
        
        if not emotion_scores:
            return 0.0
        
        # Calculate weighted valence
        total_valence = 0.0
        total_weight = 0.0
        
        for emotion, score in emotion_scores.items():
            if emotion in emotion_valence:
                total_valence += emotion_valence[emotion] * score
                total_weight += score
        
        if total_weight == 0:
            return 0.0
        
        valence = total_valence / total_weight
        return round(max(-1.0, min(1.0, valence)), 3)

    def _determine_importance(self, text: str, context: str, emotion_scores: Dict[str, float]) -> str:
        """
        Determine memory importance based on content, context, and emotional intensity.
        
        Args:
            text: Memory content
            context: Classified context
            emotion_scores: Detected emotions with scores
            
        Returns:
            Importance level: 'high', 'medium', or 'low'
        """
        importance_score = 0.0
        
        # Context-based scoring
        if context in self.context_patterns:
            patterns = self.context_patterns[context]
            text_lower = text.lower()
            
            for importance_level, keywords in patterns.items():
                matches = sum(1 for keyword in keywords if keyword in text_lower)
                if matches > 0:
                    level_weight = {'high_importance': 3.0, 'medium_importance': 2.0, 'low_importance': 1.0}
                    importance_score += level_weight.get(importance_level, 1.0) * matches
        
        # Emotional intensity boost
        if emotion_scores:
            max_emotion_score = max(emotion_scores.values())
            importance_score += max_emotion_score * 2.0
        
        # Length-based scoring (longer content often more important)
        length_score = min(len(text) / 200, 1.0)
        importance_score += length_score
        
        # Question/uncertainty indicators (often important for AI relationships)
        if '?' in text or any(word in text.lower() for word in ['help', 'advice', 'what should', 'how do']):
            importance_score += 1.0
        
        # Classify based on total score
        if importance_score >= 4.0:
            return 'high'
        elif importance_score >= 2.0:
            return 'medium'
        else:
            return 'low'

    def _determine_time_of_day(self, timestamp: Optional[str] = None) -> str:
        """
        Determine time of day from timestamp.
        
        Args:
            timestamp: ISO timestamp string, defaults to current time
            
        Returns:
            Time period: 'morning', 'afternoon', 'evening', 'night'
        """
        try:
            if timestamp:
                dt = datetime.fromisoformat(timestamp)
            else:
                dt = datetime.now()
            
            hour = dt.hour
            
            if 5 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 22:
                return 'evening'
            else:
                return 'night'
                
        except Exception:
            return 'unknown'

    def _determine_lifespan(self, importance: str, context: str, memory_type: str) -> str:
        """
        Determine memory lifespan based on importance, context, and type.
        
        Args:
            importance: Memory importance level
            context: Memory context
            memory_type: Type of memory
            
        Returns:
            Lifespan: 'permanent', 'volatile', or 'ephemeral'
        """
        # Core memories are always permanent
        if memory_type == 'core':
            return 'permanent'
        
        # High importance memories are usually permanent
        if importance == 'high':
            return 'permanent'
        
        # Emotional contexts get longer lifespan
        if context in ['emotional', 'personal', 'love', 'trust']:
            return 'permanent' if importance == 'medium' else 'volatile'
        
        # Technical/business contexts
        if context in ['technical', 'business', 'project_planning']:
            return 'volatile' if importance == 'medium' else 'ephemeral'
        
        # Default to volatile for medium importance, ephemeral for low
        return 'volatile' if importance == 'medium' else 'ephemeral'

    # =================== FLASH MEMORY SYSTEM ===================
    def _add_to_flash_memory(self, user_id: str, conversation_turn: Dict):
        """
        Add a conversation turn to flash memory.
        
        Args:
            user_id: User identifier
            conversation_turn: Dictionary containing turn data
        """
        self._flash_memory[user_id].append(conversation_turn)
        logger.debug(f"Added turn to flash memory for user {user_id}. Total: {len(self._flash_memory[user_id])}")

    def get_flash_memories(self, user_id: str) -> List[Dict]:
        """
        Retrieve flash memory (recent conversation turns) for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of recent conversation turns
        """
        try:
            flash_memories = list(self._flash_memory.get(user_id, []))
            logger.debug(f"Retrieved {len(flash_memories)} flash memories for user {user_id}")
            return flash_memories
        except Exception as e:
            logger.error(f"Failed to get flash memories: {e}")
            return []

    def clear_flash_memory(self, user_id: str):
        """Clear flash memory for a specific user."""
        if user_id in self._flash_memory:
            self._flash_memory[user_id].clear()
            logger.info(f"Cleared flash memory for user {user_id}")

    # =================== EPISODIC MEMORY SYSTEM ===================
    def store_episode_summary(
        self, 
        summary_text: str, 
        user_id: str, 
        episode_context: str = "session_summary",
        user_name: Optional[str] = None
    ) -> bool:
        """
        Store an episodic memory summary.
        
        Args:
            summary_text: The episode summary text
            user_id: User identifier
            episode_context: Context for the episode
            user_name: Optional user name
            
        Returns:
            True if successfully stored
        """
        try:
            current_timestamp = datetime.now().isoformat()
            episode_id = self._create_memory_id(user_id, current_timestamp + "_episode")
            
            # Generate embedding for the summary
            embedding = self._generate_embedding(summary_text)
            
            # Create comprehensive metadata for episodic memory
            metadata = {
                "user_id": user_id,
                "speaker": "system",
                "timestamp": current_timestamp,
                "context": "episodic",
                "type": "episode",
                "importance": "high",  # Episodes are typically important
                "lifespan": "permanent",
                "time_of_day": self._determine_time_of_day(current_timestamp),
                "episode_context": episode_context,
                "summary": f"Episode summary: {summary_text[:100]}...",
                "emotion_scores": "{}",  # Empty JSON string for episodes
                "sentiment_valence": 0.0,
                "is_validated": True,
                "linked_to": ""  # Empty string instead of empty list
            }
            
            if user_name:
                metadata["user_name"] = user_name
            
            # Store in ChromaDB
            self.collection.add(
                ids=[episode_id],
                documents=[summary_text],
                embeddings=[embedding],
                metadatas=[metadata]
            )
            
            logger.info(f"✅ Stored episodic memory for user {user_id}: {summary_text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store episode summary: {e}", exc_info=True)
            return False

    # =================== MEMORY LINKAGE SYSTEM ===================
    def link_memory_to_others(self, current_memory_id: str, linked_memory_ids: List[str]):
        """
        Link a memory to other related memories.
        
        Args:
            current_memory_id: ID of the current memory
            linked_memory_ids: List of memory IDs to link to
        """
        try:
            # Get current memory
            result = self.collection.get(ids=[current_memory_id], include=["metadatas"])
            
            if not result["metadatas"]:
                logger.warning(f"Memory {current_memory_id} not found for linking")
                return
            
            current_metadata = result["metadatas"][0]
            
            # Update linked_to field - convert list to comma-separated string for ChromaDB
            existing_links_str = current_metadata.get("linked_to", "")
            existing_links = [link.strip() for link in existing_links_str.split(",") if link.strip()]
            
            # Merge with new links and remove duplicates
            all_links = list(set(existing_links + linked_memory_ids))
            
            # Convert back to comma-separated string
            current_metadata["linked_to"] = ",".join(all_links) if all_links else ""
            
            # Update the memory in ChromaDB
            self.collection.update(
                ids=[current_memory_id],
                metadatas=[current_metadata]
            )
            
            logger.info(f"✅ Linked memory {current_memory_id} to {len(all_links)} other memories")
            
        except Exception as e:
            logger.error(f"Failed to link memories: {e}", exc_info=True)

    def get_linked_memories(self, memory_id: str) -> List[Dict]:
        """
        Retrieve memories linked to a specific memory.
        
        Args:
            memory_id: ID of the memory to find links for
            
        Returns:
            List of linked memory dictionaries
        """
        try:
            # Get the memory and its links
            result = self.collection.get(ids=[memory_id], include=["metadatas"])
            
            if not result["metadatas"]:
                return []
            
            linked_ids_str = result["metadatas"][0].get("linked_to", "")
            
            if not linked_ids_str:
                return []
            
            # Parse comma-separated string back to list
            linked_ids = [link.strip() for link in linked_ids_str.split(",") if link.strip()]
            
            if not linked_ids:
                return []
            
            # Get linked memories
            linked_result = self.collection.get(
                ids=linked_ids,
                include=["documents", "metadatas"]
            )
            
            linked_memories = []
            for i, doc in enumerate(linked_result["documents"]):
                linked_memories.append({
                    "id": linked_result["ids"][i],
                    "content": doc,
                    "metadata": linked_result["metadatas"][i]
                })
            
            logger.debug(f"Retrieved {len(linked_memories)} linked memories for {memory_id}")
            return linked_memories
            
        except Exception as e:
            logger.error(f"Failed to get linked memories: {e}")
            return []

    def _parse_emotion_scores(self, emotion_scores_str: str) -> Dict[str, float]:
        """
        Parse emotion scores from JSON string stored in ChromaDB.
        
        Args:
            emotion_scores_str: JSON string of emotion scores
            
        Returns:
            Dictionary of emotion scores
        """
        try:
            if not emotion_scores_str or emotion_scores_str == "{}":
                return {}
            return json.loads(emotion_scores_str)
        except Exception as e:
            logger.warning(f"Failed to parse emotion scores '{emotion_scores_str}': {e}")
            return {}

    # =================== ENHANCED CORE METHODS ===================
    def _classify_context(self, text: str) -> str:
        """Enhanced context classification with importance-aware scoring."""
        text_lower = text.lower()
        context_scores = {}
        
        for context, importance_patterns in self.context_patterns.items():
            total_score = 0.0
            
            for importance_level, keywords in importance_patterns.items():
                matches = sum(1 for keyword in keywords if keyword in text_lower)
                if matches > 0:
                    # Weight by importance level
                    weight = {'high_importance': 3.0, 'medium_importance': 2.0, 'low_importance': 1.0}
                    total_score += weight.get(importance_level, 1.0) * matches
            
            if total_score > 0:
                context_scores[context] = total_score
        
        if context_scores:
            dominant_context = max(context_scores, key=context_scores.get)
            logger.debug(f"Classified context as '{dominant_context}' with score {context_scores[dominant_context]}")
            return dominant_context
        
        return 'general'

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector with enhanced preprocessing."""
        try:
            processed_text = self._preprocess_for_embedding(text)
            embedding = self.embedding_model.encode(processed_text, normalize_embeddings=True)
            logger.debug(f"Generated embedding of dimension {len(embedding)}")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * 384

    def _preprocess_for_embedding(self, text: str) -> str:
        """Preprocess text for optimal embedding generation."""
        processed = ' '.join(text.split())
        if len(processed) > 500:
            processed = processed[:500] + "..."
            logger.debug("Truncated text for embedding generation")
        return processed

    def _create_memory_id(self, user_id: str, timestamp: str) -> str:
        """Create a unique memory ID with enhanced collision avoidance."""
        unique_suffix = str(uuid.uuid4())[:8]
        clean_timestamp = timestamp.replace(':', '').replace('-', '').replace('.', '')[:14]
        memory_id = f"mem_{user_id[:8]}_{clean_timestamp}_{unique_suffix}"
        return memory_id

    def _is_memorable_content(self, content: str) -> bool:
        """Enhanced memorability assessment with multi-factor analysis."""
        if not content or len(content.strip()) < 10:
            return False
        
        content_lower = content.lower()
        memorability_score = 0.0
        
        # Emotional content detection
        _, emotion_scores = self._detect_emotion_with_scores(content)
        if emotion_scores:
            memorability_score += max(emotion_scores.values()) * 2.0
        
        # Context importance
        context = self._classify_context(content)
        if context in ['emotional', 'personal', 'project_planning', 'learning']:
            memorability_score += 1.5
        
        # Question/advice patterns
        if '?' in content or any(word in content_lower for word in ['help', 'advice', 'what should', 'how do']):
            memorability_score += 1.0
        
        # Length consideration
        if len(content) > 100:
            memorability_score += 0.5
        
        # Decision threshold
        return memorability_score >= 1.0

    def _is_memorable_response(self, response: str) -> bool:
        """Enhanced AI response memorability assessment."""
        if not response or len(response.strip()) < 15:
            return False
        
        response_lower = response.lower()
        memorability_score = 0.0
        
        # Empathetic/supportive patterns
        empathy_patterns = ['understand', 'feel', 'support', 'here for you', 'believe in you']
        empathy_matches = sum(1 for pattern in empathy_patterns if pattern in response_lower)
        memorability_score += empathy_matches * 0.5
        
        # Emotional content
        _, emotion_scores = self._detect_emotion_with_scores(response)
        if emotion_scores:
            memorability_score += max(emotion_scores.values())
        
        # Questions and engagement
        if '?' in response:
            memorability_score += 0.3
        
        # Length consideration
        if len(response) > 150:
            memorability_score += 0.4
        
        return memorability_score >= 0.8

    # =================== ENHANCED STORE CONVERSATION WITH SESSION TRACKING ===================
    def store_conversation(
        self,
        user_message: str,
        ai_response: str,
        user_id: str = "default_user",
        user_emotion: Optional[str] = None,
        ai_emotion: Optional[str] = None,
        user_emotion_confidence: Optional[float] = None,
        user_name: Optional[str] = None,
        memory_type: str = "conversational",
        force_store: bool = False
    ) -> int:
        """
        Enhanced conversation storage with session tracking and episodic memory.

        Args:
            user_message: The user's message
            ai_response: The AI's response
            user_id: Identifier for the user
            user_emotion: Pre-detected emotion for user's message
            ai_emotion: Pre-detected emotion for AI's response
            user_emotion_confidence: Confidence score for user's emotion
            user_name: Optional name of the user
            memory_type: Type of memory ('core', 'episodic', 'volatile', 'conversational')
            force_store: Force storage even if not deemed memorable

        Returns:
            Number of memories successfully stored
        """
        try:
            logger.info(f"Processing enhanced conversation for storage - User: {user_id}" + 
                       (f" ({user_name})" if user_name else ""))

            # Update user name mapping
            if user_name:
                self.set_user_name(user_id, user_name)

            current_timestamp = datetime.now().isoformat()
            
            # =================== SESSION TRACKING ===================
            # Update or start session for this user
            self._start_or_update_session(
                user_id=user_id,
                user_message=user_message,
                ai_response=ai_response,
                user_emotion=user_emotion,
                ai_emotion=ai_emotion,
                user_name=user_name
            )

            # =================== FLASH MEMORY UPDATE ===================
            # Always add to flash memory regardless of long-term storage decision
            flash_turn = {
                "timestamp": current_timestamp,
                "user_message": user_message,
                "ai_response": ai_response,
                "user_emotion": user_emotion,
                "ai_emotion": ai_emotion,
                "context": self._classify_context(user_message + " " + ai_response)
            }
            self._add_to_flash_memory(user_id, flash_turn)

            # Enhanced memorability filtering
            user_memorable = force_store or self._is_memorable_content(user_message)
            ai_memorable = force_store or self._is_memorable_response(ai_response)

            if not user_memorable and not ai_memorable:
                logger.info("Conversation deemed not memorable - stored in flash and session only")
                return 0

            stored_count = 0
            memories_to_store = []

            # Enhanced content extraction
            combined_content = f"User: {user_message}\nMai: {ai_response}"
            context = self._classify_context(combined_content)

            # Store user message if memorable
            if user_memorable:
                user_memory_id = self._create_memory_id(user_id, current_timestamp + "_user")
                user_embedding = self._generate_embedding(user_message)

                # Enhanced emotion analysis
                detected_emotion, emotion_scores = self._detect_emotion_with_scores(user_message)
                final_user_emotion = user_emotion or detected_emotion
                
                # Calculate sentiment valence
                sentiment_valence = self._calculate_sentiment_valence(user_message, emotion_scores)
                
                # Determine enhanced metadata
                importance = self._determine_importance(user_message, context, emotion_scores)
                lifespan = self._determine_lifespan(importance, context, memory_type)
                time_of_day = self._determine_time_of_day(current_timestamp)

                user_metadata = {
                    "user_id": user_id,
                    "speaker": "user",
                    "timestamp": current_timestamp,
                    "context": context,
                    "type": memory_type,
                    "importance": importance,
                    "lifespan": lifespan,
                    "time_of_day": time_of_day,
                    "emotion_scores": json.dumps(emotion_scores) if emotion_scores else "{}",
                    "sentiment_valence": sentiment_valence,
                    "is_validated": True,
                    "linked_to": "",  # Empty string instead of empty list
                    "summary": f"User {final_user_emotion or 'neutral'} message in {context} context"
                }

                # Add optional fields
                if user_name:
                    user_metadata["user_name"] = user_name
                if final_user_emotion:
                    user_metadata["emotion"] = final_user_emotion
                    user_metadata["user_emotion"] = final_user_emotion
                if user_emotion_confidence is not None:
                    user_metadata["user_emotion_confidence"] = user_emotion_confidence

                memories_to_store.append({
                    "id": user_memory_id,
                    "document": user_message,
                    "embedding": user_embedding,
                    "metadata": user_metadata
                })

            # Store AI response if memorable
            if ai_memorable:
                ai_memory_id = self._create_memory_id(user_id, current_timestamp + "_ai")
                ai_embedding = self._generate_embedding(ai_response)

                # Enhanced emotion analysis for AI response
                detected_ai_emotion, ai_emotion_scores = self._detect_emotion_with_scores(ai_response)
                final_ai_emotion = ai_emotion or detected_ai_emotion
                
                # Calculate sentiment valence
                ai_sentiment_valence = self._calculate_sentiment_valence(ai_response, ai_emotion_scores)
                
                # Determine enhanced metadata
                ai_importance = self._determine_importance(ai_response, context, ai_emotion_scores)
                ai_lifespan = self._determine_lifespan(ai_importance, context, memory_type)
                time_of_day = self._determine_time_of_day(current_timestamp)

                ai_metadata = {
                    "user_id": user_id,
                    "speaker": "ai",
                    "timestamp": current_timestamp,
                    "context": context,
                    "type": memory_type,
                    "importance": ai_importance,
                    "lifespan": ai_lifespan,
                    "time_of_day": time_of_day,
                    "emotion_scores": json.dumps(ai_emotion_scores) if ai_emotion_scores else "{}",
                    "sentiment_valence": ai_sentiment_valence,
                    "is_validated": True,
                    "linked_to": "",  # Empty string instead of empty list
                    "summary": f"Mai {final_ai_emotion or 'neutral'} response in {context} context"
                }

                # Add optional fields
                if user_name:
                    ai_metadata["user_name"] = user_name
                if final_ai_emotion:
                    ai_metadata["emotion"] = final_ai_emotion
                    ai_metadata["ai_emotion"] = final_ai_emotion

                memories_to_store.append({
                    "id": ai_memory_id,
                    "document": ai_response,
                    "embedding": ai_embedding,
                    "metadata": ai_metadata
                })

            # Batch insert memories
            if memories_to_store:
                ids = [mem["id"] for mem in memories_to_store]
                documents = [mem["document"] for mem in memories_to_store]
                embeddings = [mem["embedding"] for mem in memories_to_store]
                metadatas = [mem["metadata"] for mem in memories_to_store]

                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )

                stored_count = len(memories_to_store)
                logger.info(f"✅ Successfully stored {stored_count} enhanced memories for user '{user_id}'")

            return stored_count

        except Exception as e:
            logger.error(f"Failed to store enhanced conversation: {e}", exc_info=True)
            return 0

    # =================== PERIODIC CLEANUP METHODS ===================
    def cleanup_expired_sessions(self):
        """Public method to manually trigger session cleanup."""
        self._cleanup_expired_sessions()

    def clean_expired_memories(self) -> int:
        """
        Clean expired memories based on lifespan settings.
        
        Returns:
            Number of memories cleaned
        """
        try:
            logger.info("Starting expired memory cleanup...")
            current_time = datetime.now()
            cleaned_count = 0
            
            # Get all memories to check expiration
            results = self.collection.get(include=["metadatas"])
            
            if not results["metadatas"]:
                logger.info("No memories found for cleanup")
                return 0
            
            expired_ids = []
            
            for i, metadata in enumerate(results["metadatas"]):
                memory_id = results["ids"][i]
                lifespan = metadata.get("lifespan", "permanent")
                timestamp_str = metadata.get("timestamp")
                
                if lifespan == "permanent" or not timestamp_str:
                    continue
                
                try:
                    memory_time = datetime.fromisoformat(timestamp_str)
                    days_old = (current_time - memory_time).days
                    
                    lifespan_limit = self.lifespan_days.get(lifespan)
                    
                    if lifespan_limit and days_old > lifespan_limit:
                        expired_ids.append(memory_id)
                        logger.debug(f"Memory {memory_id} expired ({days_old} days > {lifespan_limit} limit)")
                        
                except Exception as e:
                    logger.warning(f"Could not parse timestamp for memory {memory_id}: {e}")
                    continue
            
            # Delete expired memories
            if expired_ids:
                self.collection.delete(ids=expired_ids)
                cleaned_count = len(expired_ids)
                logger.info(f"✅ Cleaned {cleaned_count} expired memories")
            else:
                logger.info("No expired memories found")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to clean expired memories: {e}", exc_info=True)
            return 0

    # =================== ENHANCED MEMORY RETRIEVAL ===================
    def retrieve_memories(
        self, 
        query: str, 
        user_id: str = "default_user", 
        limit: int = 5, 
        similarity_threshold: float = 0.0,
        include_flash: bool = True,
        memory_types: Optional[List[str]] = None,
        importance_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Enhanced memory retrieval with multi-layered search and advanced filtering.

        Args:
            query: Search query for retrieving relevant memories
            user_id: User identifier to filter memories
            limit: Maximum number of memories to retrieve
            similarity_threshold: Minimum similarity score for relevance
            include_flash: Whether to include flash memories in results
            memory_types: Optional filter by memory types
            importance_filter: Optional filter by importance level

        Returns:
            List of relevant memories with enhanced metadata and flash memories
        """
        try:
            display_user_info = self.get_user_name(user_id) or user_id[:8] + "..."
            logger.info(f"Enhanced memory retrieval for query: '{query}' (user: {display_user_info})")

            all_results = []

            # =================== FLASH MEMORY SEARCH ===================
            if include_flash:
                flash_memories = self.get_flash_memories(user_id)
                flash_results = []
                
                query_lower = query.lower()
                for i, flash_turn in enumerate(flash_memories):
                    # Simple text matching for flash memories
                    user_msg = flash_turn.get("user_message", "").lower()
                    ai_msg = flash_turn.get("ai_response", "").lower()
                    
                    if query_lower in user_msg or query_lower in ai_msg:
                        flash_results.append({
                            "content": f"User: {flash_turn.get('user_message', '')}\nMai: {flash_turn.get('ai_response', '')}",
                            "metadata": {
                                "source": "flash_memory",
                                "timestamp": flash_turn.get("timestamp"),
                                "context": flash_turn.get("context"),
                                "user_emotion": flash_turn.get("user_emotion"),
                                "ai_emotion": flash_turn.get("ai_emotion"),
                                "recency_rank": len(flash_memories) - i
                            },
                            "relevance_score": 0.9 + (0.1 * (len(flash_memories) - i) / len(flash_memories)),
                            "base_similarity": 0.8
                        })
                
                all_results.extend(flash_results[:max(1, limit // 3)])  # Include some flash memories

            # =================== PERSISTENT MEMORY SEARCH ===================
            query_embedding = self._generate_embedding(query)
            query_context = self._classify_context(query)
            query_emotion, query_emotion_scores = self._detect_emotion_with_scores(query)

            # Build where clause with enhanced filtering
            where_clause = {"user_id": user_id}
            
            if memory_types:
                where_clause["type"] = {"$in": memory_types}
            
            if importance_filter:
                where_clause["importance"] = importance_filter

            results = self.collection.query(
                query_embeddings=[query_embedding],
                where=where_clause,
                n_results=min(limit * 2, 200),
                include=["documents", "metadatas", "distances"]
            )

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]

                for i in range(len(documents)):
                    euclidean_distance = distances[i]
                    clamped_distance = min(euclidean_distance, 2.0)
                    base_similarity = 1.0 - (clamped_distance**2 / 2.0)

                    if base_similarity < similarity_threshold:
                        continue

                    memory_metadata = metadatas[i]
                    relevance_boost = 0.0

                    # Context matching boost
                    memory_context = memory_metadata.get("context", "general")
                    if memory_context == query_context:
                        relevance_boost += 0.15

                    # Emotion matching boost
                    memory_emotion = memory_metadata.get("emotion")
                    if query_emotion and memory_emotion == query_emotion:
                        relevance_boost += 0.1

                    # Importance boost
                    importance = memory_metadata.get("importance", "low")
                    importance_weights = {"high": 0.1, "medium": 0.05, "low": 0.0}
                    relevance_boost += importance_weights.get(importance, 0.0)

                    # Recency boost
                    try:
                        memory_time = datetime.fromisoformat(memory_metadata.get("timestamp", ""))
                        hours_ago = (datetime.now() - memory_time).total_seconds() / 3600
                        if hours_ago < 24:
                            relevance_boost += 0.08
                        elif hours_ago < 168:
                            relevance_boost += 0.04
                    except:
                        pass

                    # Type-specific boost
                    memory_type = memory_metadata.get("type", "conversational")
                    type_weights = {"core": 0.15, "episodic": 0.1, "conversational": 0.05, "volatile": 0.0}
                    relevance_boost += type_weights.get(memory_type, 0.0)

                    final_relevance = min(base_similarity + relevance_boost, 1.0)

                    enhanced_memory = {
                        "content": documents[i],
                        "metadata": memory_metadata,
                        "relevance_score": round(final_relevance, 3),
                        "base_similarity": round(base_similarity, 3)
                    }

                    all_results.append(enhanced_memory)

            # Sort all results by relevance score
            all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            final_memories = all_results[:limit]

            logger.info(f"✅ Retrieved {len(final_memories)} enhanced memories " + 
                       f"({sum(1 for m in final_memories if m['metadata'].get('source') == 'flash_memory')} from flash)")

            return final_memories

        except Exception as e:
            logger.error(f"Failed to retrieve enhanced memories: {e}", exc_info=True)
            return []

    # =================== PRESERVED CORE METHODS ===================
    def delete_memories(self, user_id: str = None, ids: List[str] = None, content_contains: str = None) -> int:
        """Enhanced memory deletion with multiple filtering options."""
        try:
            if not any([user_id, ids, content_contains]):
                logger.warning("No deletion criteria provided")
                return 0
            
            # Delete by specific IDs
            if ids:
                logger.info(f"Deleting memories by IDs: {ids}")
                existing_results = self.collection.get(ids=ids, include=["documents"])
                valid_ids = existing_results["ids"] if existing_results["ids"] else []
                
                if valid_ids:
                    self.collection.delete(ids=valid_ids)
                    logger.info(f"✅ Deleted {len(valid_ids)} memories by IDs.")
                    return len(valid_ids)
                else:
                    logger.warning("No valid IDs found for deletion.")
                    return 0
            
            # Build where clause for filtering
            where_clause = {}
            if user_id:
                where_clause["user_id"] = user_id
            
            # Get memories matching criteria
            if where_clause or content_contains:
                results = self.collection.get(
                    where=where_clause if where_clause else None,
                    include=["documents", "metadatas"]
                )
                
                if not results["documents"]:
                    logger.info("No memories found matching deletion criteria.")
                    return 0
                
                memories_to_delete = []
                
                # Filter by content if specified
                if content_contains:
                    content_lower = content_contains.lower()
                    for i, doc in enumerate(results["documents"]):
                        if content_lower in doc.lower():
                            memories_to_delete.append(results["ids"][i])
                else:
                    memories_to_delete = results["ids"]
                
                # Perform deletion
                if memories_to_delete:
                    self.collection.delete(ids=memories_to_delete)
                    deleted_count = len(memories_to_delete)
                    logger.info(f"✅ Deleted {deleted_count} memories matching criteria.")
                    return deleted_count
                
            return 0
            
        except Exception as e:
            logger.error(f"Failed to delete memories: {e}", exc_info=True)
            return 0

    def clear_user_memories(self, user_id: str) -> int:
        """Clear all memories for a specific user with enhanced logging."""
        try:
            display_name = self.get_user_name(user_id) or user_id
            logger.info(f"Clearing all memories for user: {display_name}")
            
            # Clear flash memory first
            self.clear_flash_memory(user_id)
            
            # Force end any active session
            self.force_end_session(user_id)
            
            # Get count before deletion
            results = self.collection.get(
                where={"user_id": user_id},
                include=["documents"]
            )
            
            if not results["ids"]:
                logger.info(f"No persistent memories found for user '{display_name}'.")
                return 0
            
            # Delete all persistent memories for user
            self.collection.delete(where={"user_id": user_id})
            
            deleted_count = len(results["ids"])
            logger.info(f"✅ Successfully cleared {deleted_count} persistent memories for user '{display_name}'.")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to clear memories for user '{user_id}': {e}", exc_info=True)
            return 0

    def delete_all_user_memories(self, user_id: str) -> int:
        """Enhanced deletion of all memories for a specific user."""
        return self.clear_user_memories(user_id)

    def get_memory_stats(self) -> Dict:
        """Enhanced memory statistics with multi-layered analysis and session info."""
        try:
            logger.info("Generating comprehensive enhanced memory statistics")
            
            total_memories = self.collection.count()
            
            # Flash memory stats
            flash_stats = {}
            total_flash_memories = 0
            for user_id, flash_mem in self._flash_memory.items():
                user_name = self.get_user_name(user_id) or user_id[:8] + "..."
                flash_count = len(flash_mem)
                if flash_count > 0:
                    flash_stats[user_name] = flash_count
                    total_flash_memories += flash_count
            
            # Session stats
            session_stats = self.get_active_session_stats()
            
            if total_memories == 0:
                return {
                    "total_persistent_memories": 0,
                    "total_flash_memories": total_flash_memories,
                    "flash_memory_by_user": flash_stats,
                    "active_sessions": session_stats,
                    "memories_by_context": {},
                    "memories_by_user": {},
                    "memories_by_emotion": {},
                    "memories_by_speaker": {},
                    "memories_by_type": {},
                    "memories_by_importance": {},
                    "memories_by_lifespan": {},
                    "collection_name": self.collection_name,
                    "insights": ["No persistent memories stored yet"]
                }
            
            # Get all metadata for comprehensive analysis
            results = self.collection.get(include=["metadatas"])
            metadatas = results["metadatas"]
            
            # Enhanced categorization
            context_counts = {}
            user_counts = {}
            emotion_counts = {}
            speaker_counts = {}
            type_counts = {}
            importance_counts = {}
            lifespan_counts = {}
            recent_activity = {"last_24h": 0, "last_week": 0, "last_month": 0}
            
            current_time = datetime.now()
            
            for metadata in metadatas:
                # Context analysis
                context = metadata.get("context", "unknown")
                context_counts[context] = context_counts.get(context, 0) + 1
                
                # User analysis
                user_id = metadata.get("user_id", "unknown")
                user_display_id = self.get_user_name(user_id) or user_id[:8] + "..."
                user_counts[user_display_id] = user_counts.get(user_display_id, 0) + 1
                
                # Emotion analysis
                emotion = metadata.get("emotion", "neutral")
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                # Speaker analysis
                speaker = metadata.get("speaker", "unknown")
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                
                # Type analysis
                mem_type = metadata.get("type", "conversational")
                type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
                
                # Importance analysis
                importance = metadata.get("importance", "unknown")
                importance_counts[importance] = importance_counts.get(importance, 0) + 1
                
                # Lifespan analysis
                lifespan = metadata.get("lifespan", "unknown")
                lifespan_counts[lifespan] = lifespan_counts.get(lifespan, 0) + 1
                
                # Recency analysis
                try:
                    timestamp = metadata.get("timestamp")
                    if timestamp:
                        memory_time = datetime.fromisoformat(timestamp)
                        time_diff = current_time - memory_time
                        
                        if time_diff.days == 0:
                            recent_activity["last_24h"] += 1
                        if time_diff.days <= 7:
                            recent_activity["last_week"] += 1
                        if time_diff.days <= 30:
                            recent_activity["last_month"] += 1
                except:
                    pass
            
            # Generate enhanced insights
            insights = []
            
            # Multi-layer memory insights
            insights.append(f"Total memories: {total_memories} persistent + {total_flash_memories} flash + {session_stats['total_active_sessions']} active sessions")
            
            # Most active user
            if user_counts:
                most_active_user = max(user_counts, key=user_counts.get)
                insights.append(f"Most active user: {most_active_user} ({user_counts[most_active_user]} memories)")
            
            # Dominant context
            if context_counts:
                dominant_context = max(context_counts, key=context_counts.get)
                insights.append(f"Dominant context: {dominant_context} ({context_counts[dominant_context]} memories)")
            
            # Memory type distribution
            if type_counts:
                top_type = max(type_counts, key=type_counts.get)
                insights.append(f"Most common type: {top_type} ({type_counts[top_type]} memories)")
            
            # Importance distribution
            if importance_counts:
                high_importance = importance_counts.get('high', 0)
                total_importance = sum(importance_counts.values())
                high_percentage = (high_importance / total_importance * 100) if total_importance > 0 else 0
                insights.append(f"High importance memories: {high_importance} ({high_percentage:.1f}%)")
            
            # Emotional insights
            if emotion_counts:
                top_emotion = max(emotion_counts, key=emotion_counts.get)
                insights.append(f"Most common emotion: {top_emotion} ({emotion_counts[top_emotion]} memories)")
            
            # Activity insights
            insights.append(f"Recent activity: {recent_activity['last_24h']} memories in last 24h")
            
            # Session insights
            if session_stats['total_active_sessions'] > 0:
                insights.append(f"Active sessions: {session_stats['total_active_sessions']} users currently engaged")
            
            comprehensive_stats = {
                "total_persistent_memories": total_memories,
                "total_flash_memories": total_flash_memories,
                "flash_memory_by_user": flash_stats,
                "active_sessions": session_stats,
                "memories_by_context": context_counts,
                "memories_by_user": user_counts,
                "memories_by_emotion": emotion_counts,
                "memories_by_speaker": speaker_counts,
                "memories_by_type": type_counts,
                "memories_by_importance": importance_counts,
                "memories_by_lifespan": lifespan_counts,
                "recent_activity": recent_activity,
                "collection_name": self.collection_name,
                "insights": insights,
                "generation_time": current_time.isoformat()
            }
            
            logger.info(f"✅ Generated comprehensive enhanced stats with session tracking")
            return comprehensive_stats
                
        except Exception as e:
            logger.error(f"Failed to generate enhanced memory stats: {e}", exc_info=True)
            return {"error": str(e), "collection_name": self.collection_name}

    def get_recent_memories(self, user_id: str = "default_user", limit: int = 10, include_flash: bool = True) -> List[Dict]:
        """Enhanced recent memories retrieval with flash memory integration."""
        try:
            display_name = self.get_user_name(user_id) or user_id
            logger.info(f"Retrieving recent memories for user: {display_name}")
            
            all_memories = []
            
            # Include flash memories if requested
            if include_flash:
                flash_memories = self.get_flash_memories(user_id)
                for flash_turn in flash_memories:
                    all_memories.append({
                        "id": f"flash_{flash_turn.get('timestamp', 'unknown')}",
                        "content": f"User: {flash_turn.get('user_message', '')}\nMai: {flash_turn.get('ai_response', '')}",
                        "timestamp": flash_turn.get("timestamp"),
                        "context": flash_turn.get("context"),
                        "speaker": "flash_memory",
                        "emotion": flash_turn.get("user_emotion"),
                        "summary": "Flash memory conversation turn",
                        "user_name": self.get_user_name(user_id),
                        "source": "flash"
                    })
            
            # Get persistent memories
            results = self.collection.get(
                where={"user_id": user_id},
                include=["documents", "metadatas"]
            )
            
            if results["documents"]:
                for i in range(len(results["documents"])):
                    doc = results["documents"][i]
                    metadata = results["metadatas"][i]
                    memory_id = results["ids"][i]
                    
                    memory = {
                        "id": memory_id,
                        "content": doc,
                        "timestamp": metadata.get("timestamp"),
                        "context": metadata.get("context"),
                        "speaker": metadata.get("speaker"),
                        "emotion": metadata.get("emotion"),
                        "summary": metadata.get("summary"),
                        "user_name": metadata.get("user_name"),
                        "type": metadata.get("type"),
                        "importance": metadata.get("importance"),
                        "source": "persistent"
                    }
                    all_memories.append(memory)
            
            # Sort by timestamp (most recent first)
            all_memories.sort(key=lambda x: x.get("timestamp", "") or "", reverse=True)
            
            recent_memories = all_memories[:limit]
            logger.info(f"Retrieved {len(recent_memories)} recent memories (flash + persistent)")
            return recent_memories
            
        except Exception as e:
            logger.error(f"Failed to get enhanced recent memories: {e}", exc_info=True)
            return []

    def display_memories(self, user_id: str = "default_user", limit: int = 10, include_ids: bool = False, include_flash: bool = True) -> None:
        """Enhanced memory display with flash memory integration."""
        try:
            display_name = self.get_user_name(user_id) or user_id
            logger.info(f"Displaying enhanced memories for user: '{display_name}' (limit: {limit})")

            # Get recent memories including flash
            memories = self.get_recent_memories(user_id, limit, include_flash)

            if not memories:
                logger.info(f"No memories found for user '{display_name}'")
                print(f"\n📭 No memories found for user '{display_name}'\n")
                return

            # Enhanced display formatting
            print(f"\n🧠 Enhanced Memory Bank for '{display_name}' (Top {len(memories)} memories)")
            print("=" * 100)
            
            for i, memory in enumerate(memories):
                # Format timestamp
                try:
                    if memory.get('timestamp'):
                        formatted_time = datetime.fromisoformat(memory['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        formatted_time = "unknown"
                except:
                    formatted_time = memory.get('timestamp', "unknown")
                
                # Build enhanced display string
                display_parts = []
                if include_ids:
                    display_parts.append(f"ID: {memory['id']}")
                
                display_parts.extend([
                    f"Speaker: {memory.get('speaker', 'unknown')}",
                    f"Context: {memory.get('context', 'unknown')}",
                    f"Source: {memory.get('source', 'unknown')}",
                    f"Time: {formatted_time}"
                ])
                
                # Add enhanced metadata if available
                if memory.get('type'):
                    display_parts.append(f"Type: {memory['type']}")
                if memory.get('importance'):
                    display_parts.append(f"Importance: {memory['importance']}")
                if memory.get('emotion'):
                    display_parts.append(f"Emotion: {memory['emotion']}")

                print(f"\n[{i+1}] {' | '.join(display_parts)}")
                
                # Display content with intelligent truncation
                content = memory.get('content', '')
                if len(content) > 200:
                    truncated_content = content[:200] + "..."
                    print(f"    📝 \"{truncated_content}\"")
                else:
                    print(f"    📝 \"{content}\"")
                
                # Display summary if available
                summary = memory.get('summary', '')
                if summary and summary != content[:50]:
                    print(f"    💡 Summary: {summary}")
            
            print("\n" + "=" * 100)
            logger.info(f"✅ Displayed {len(memories)} enhanced memories for user '{display_name}'")

        except Exception as e:
            logger.error(f"Failed to display enhanced memories: {e}", exc_info=True)
            print(f"\n❌ Error displaying memories: {e}\n")

    def display_all_memories(self, limit: int = 20, include_ids: bool = False, include_flash: bool = True) -> None:
        """Enhanced display of all memories across users with flash memory integration."""
        try:
            logger.info(f"Displaying all enhanced memories (limit: {limit})")

            total_persistent = self.collection.count()
            total_flash = sum(len(flash_mem) for flash_mem in self._flash_memory.values())
            
            if total_persistent == 0 and total_flash == 0:
                logger.info("No memories stored")
                print("\n📭 No memories stored in the system\n")
                return

            all_memories = []
            
            # Collect flash memories if requested
            if include_flash:
                for user_id, flash_mem in self._flash_memory.items():
                    user_name = self.get_user_name(user_id) or user_id[:8] + "..."
                    for flash_turn in flash_mem:
                        all_memories.append({
                            "id": f"flash_{user_id}_{flash_turn.get('timestamp', 'unknown')}",
                            "content": f"User: {flash_turn.get('user_message', '')}\nMai: {flash_turn.get('ai_response', '')}",
                            "timestamp": flash_turn.get("timestamp"),
                            "context": flash_turn.get("context"),
                            "speaker": "flash_memory",
                            "user_id": user_id,
                            "user_name": user_name,
                            "source": "flash"
                        })

            # Collect persistent memories
            if total_persistent > 0:
                results = self.collection.get(include=["documents", "metadatas"])
                
                for i in range(len(results["documents"])):
                    metadata = results["metadatas"][i]
                    user_id = metadata.get("user_id", "unknown")
                    user_name = self.get_user_name(user_id) or user_id[:8] + "..."
                    
                    memory = {
                        "id": results["ids"][i],
                        "content": results["documents"][i],
                        "timestamp": metadata.get("timestamp"),
                        "context": metadata.get("context", "general"),
                        "speaker": metadata.get("speaker", "unknown"),
                        "emotion": metadata.get("emotion"),
                        "user_id": user_id,
                        "user_name": user_name,
                        "summary": metadata.get("summary"),
                        "type": metadata.get("type"),
                        "importance": metadata.get("importance"),
                        "source": "persistent"
                    }
                    all_memories.append(memory)

            # Sort by timestamp (most recent first)
            all_memories.sort(key=lambda x: x.get("timestamp", "") or "", reverse=True)

            # Enhanced display with user grouping insights
            user_distribution = {}
            for memory in all_memories:
                user_name = memory.get("user_name", "unknown")
                user_distribution[user_name] = user_distribution.get(user_name, 0) + 1

            print(f"\n🌐 Enhanced Global Memory Bank")
            print(f"📊 Total: {total_persistent} persistent + {total_flash} flash memories")
            print(f"👥 Active users: {len(user_distribution)}")
            print("=" * 100)
            
            for i, memory in enumerate(all_memories[:limit]):
                # Format timestamp
                try:
                    if memory.get('timestamp'):
                        formatted_time = datetime.fromisoformat(memory['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        formatted_time = "unknown"
                except:
                    formatted_time = memory.get('timestamp', "unknown")

                # Build comprehensive display string
                display_parts = []
                if include_ids:
                    display_parts.append(f"ID: {memory['id']}")
                
                display_parts.extend([
                    f"User: {memory.get('user_name', 'unknown')}",
                    f"Speaker: {memory.get('speaker', 'unknown')}",
                    f"Context: {memory.get('context', 'unknown')}",
                    f"Source: {memory.get('source', 'unknown')}",
                    f"Time: {formatted_time}"
                ])
                
                # Add enhanced metadata
                if memory.get('type'):
                    display_parts.append(f"Type: {memory['type']}")
                if memory.get('importance'):
                    display_parts.append(f"Importance: {memory['importance']}")
                if memory.get('emotion'):
                    display_parts.append(f"Emotion: {memory['emotion']}")
                
                print(f"\n[{i+1}] {' | '.join(display_parts)}")
                
                # Display content
                content = memory.get('content', '')
                if len(content) > 150:
                    truncated_content = content[:150] + "..."
                    print(f"    📝 \"{truncated_content}\"")
                else:
                    print(f"    📝 \"{content}\"")
                
                # Display summary if available
                summary = memory.get('summary', '')
                if summary and len(summary) > 20:
                    print(f"    💡 {summary}")
            
            print("\n" + "=" * 100)
            print(f"📊 Showing {min(limit, len(all_memories))} of {len(all_memories)} total memories")
            logger.info(f"✅ Displayed {min(limit, len(all_memories))} enhanced memories from global collection")

        except Exception as e:
            logger.error(f"Failed to display all enhanced memories: {e}", exc_info=True)
            print(f"\n❌ Error displaying all memories: {e}\n")

    # =================== BACKWARD COMPATIBILITY ALIASES ===================
    def _detect_emotion(self, text: str) -> Optional[str]:
        """Backward compatibility method for emotion detection."""
        emotion, _ = self._detect_emotion_with_scores(text)
        return emotion

    def _extract_memory_content(self, user_message: str, ai_response: str) -> Tuple[str, str]:
        """Backward compatibility method for memory content extraction."""
        combined_content = f"User: {user_message}\nMai: {ai_response}"
        context = self._classify_context(combined_content)
        
        # Create enhanced summary
        user_emotion, _ = self._detect_emotion_with_scores(user_message)
        ai_emotion, _ = self._detect_emotion_with_scores(ai_response)
        
        summary_parts = []
        if user_emotion:
            summary_parts.append(f"User expressed {user_emotion}")
        if ai_emotion:
            summary_parts.append(f"Mai responded with {ai_emotion}")
        summary_parts.append(f"Context: {context}")
        
        enhanced_summary = " | ".join(summary_parts)
        return combined_content, enhanced_summary


# =================== ENHANCED CONSOLE INTERFACE ===================
if __name__ == "__main__":
    # Ensure a directory for memory persistence exists
    test_persist_dir = "./mai_memory"
    Path(test_persist_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Initialize enhanced memory manager (without LLM for console testing)
        memory_manager = MemoryManagerV4(persist_directory=test_persist_dir)

        # Pre-define test user setup
        RISHI_EMAIL = "restlessrishi@gmail.com"
        FIXED_RISHI_ID = "6653d9f1-b272-434f-8f2b-b0a96c35a1d2"
        
        # Example for a new user
        new_user_email = "newuser@example.com"
        new_user_name = "Alice"
        alice_user_id = memory_manager.get_or_create_user_id(new_user_email, new_user_name)

        while True:
            print("\n--- Enhanced MemoryManagerV4 with Session Episodes Console ---")
            print("1. Store new conversation (user/AI)")
            print("2. Display memories for a specific user (with flash)")
            print("3. Display all memories (across users + flash)")
            print("4. Store episode summary")
            print("5. Get flash memories for user")
            print("6. Link memories together")
            print("7. Clean expired memories")
            print("8. Delete memories by ID(s)")
            print("9. Delete memories by content substring")
            print("10. Delete ALL memories for a specific user")
            print("11. Show enhanced memory statistics")
            print("12. Get user ID for an email (test)")
            print("13. Test emotion detection")
            print("14. Force end session and create episode")
            print("15. Show active session stats")
            print("16. Cleanup expired sessions")
            print("17. Quit")
            
            action = input("Enter your choice (1-17): ").strip()

            if action == '1':
                email_for_storage = input("Enter user's email for this conversation: ").strip()
                current_user_id_for_storage = memory_manager.get_or_create_user_id(
                    email_for_storage, 
                    email_for_storage.split('@')[0].capitalize()
                )
                current_user_name_for_storage = memory_manager.get_user_name(current_user_id_for_storage)

                user_msg = input("Enter user message: ").strip()
                ai_resp = input("Enter Mai's response: ").strip()
                
                # Optional memory type
                mem_type = input("Memory type (core/episodic/volatile/conversational) [conversational]: ").strip()
                mem_type = mem_type if mem_type else "conversational"
                
                # Optional force store
                force = input("Force store even if not memorable? (y/n) [n]: ").strip().lower()
                force_store = force == 'y'

                stored_count = memory_manager.store_conversation(
                    user_message=user_msg,
                    ai_response=ai_resp,
                    user_id=current_user_id_for_storage,
                    user_name=current_user_name_for_storage,
                    memory_type=mem_type,
                    force_store=force_store
                )
                print(f"✅ Stored {stored_count} persistent memories (flash memory and session always updated)")
            
            elif action == '2':
                email_for_display = input("Enter user's email to display memories for: ").strip()
                user_id_for_display = memory_manager.get_or_create_user_id(email_for_display)
                include_ids_input = input("Include memory IDs? (yes/no): ").lower().strip()
                include_ids = include_ids_input == 'yes'
                include_flash_input = input("Include flash memories? (yes/no) [yes]: ").lower().strip()
                include_flash = include_flash_input != 'no'
                num_memories = input("How many memories to display (default 10)? ").strip()
                num_memories = int(num_memories) if num_memories.isdigit() else 10
                memory_manager.display_memories(user_id_for_display, limit=num_memories, 
                                              include_ids=include_ids, include_flash=include_flash)

            elif action == '3':
                include_ids_input = input("Include memory IDs? (yes/no): ").lower().strip()
                include_ids = include_ids_input == 'yes'
                include_flash_input = input("Include flash memories? (yes/no) [yes]: ").lower().strip()
                include_flash = include_flash_input != 'no'
                num_memories_all = input("How many total memories to display (default 20)? ").strip()
                num_memories_all = int(num_memories_all) if num_memories_all.isdigit() else 20
                memory_manager.display_all_memories(limit=num_memories_all, 
                                                  include_ids=include_ids, include_flash=include_flash)

            elif action == '4':
                email_for_episode = input("Enter user's email for episode summary: ").strip()
                user_id_for_episode = memory_manager.get_or_create_user_id(email_for_episode)
                user_name_for_episode = memory_manager.get_user_name(user_id_for_episode)
                
                summary_text = input("Enter episode summary: ").strip()
                episode_context = input("Episode context (default: session_summary): ").strip()
                episode_context = episode_context if episode_context else "session_summary"
                
                success = memory_manager.store_episode_summary(
                    summary_text, user_id_for_episode, episode_context, user_name_for_episode
                )
                if success:
                    print("✅ Episode summary stored successfully")
                else:
                    print("❌ Failed to store episode summary")

            elif action == '5':
                email_for_flash = input("Enter user's email to get flash memories: ").strip()
                user_id_for_flash = memory_manager.get_or_create_user_id(email_for_flash)
                flash_memories = memory_manager.get_flash_memories(user_id_for_flash)
                
                print(f"\n🔥 Flash Memories for {memory_manager.get_user_name(user_id_for_flash) or user_id_for_flash}")
                print("=" * 60)
                for i, flash in enumerate(flash_memories):
                    print(f"[{i+1}] {flash.get('timestamp', 'unknown')}")
                    print(f"    User: {flash.get('user_message', '')}")
                    print(f"    Mai: {flash.get('ai_response', '')}")
                    print(f"    Context: {flash.get('context', 'unknown')}")
                    if flash.get('user_emotion'):
                        print(f"    Emotion: {flash.get('user_emotion')}")
                print("=" * 60)

            elif action == '6':
                current_id = input("Enter current memory ID: ").strip()
                linked_ids_str = input("Enter comma-separated IDs to link to: ").strip()
                linked_ids = [uid.strip() for uid in linked_ids_str.split(',') if uid.strip()]
                
                if current_id and linked_ids:
                    memory_manager.link_memory_to_others(current_id, linked_ids)
                    print(f"✅ Linked memory {current_id} to {len(linked_ids)} other memories")
                else:
                    print("❌ Please provide valid memory IDs")

            elif action == '7':
                confirm = input("Clean expired memories? This will delete volatile memories older than 30 days (yes/no): ").lower().strip()
                if confirm == 'yes':
                    cleaned_count = memory_manager.clean_expired_memories()
                    print(f"✅ Cleaned {cleaned_count} expired memories")
                else:
                    print("Cleanup cancelled")

            elif action == '8':
                user_email_to_delete = input("Enter user's email for memory deletion context: ").strip()
                user_id_to_delete_by_id = memory_manager.get_or_create_user_id(user_email_to_delete)
                ids_to_delete_str = input("Enter comma-separated IDs to delete: ").strip()
                ids_to_delete = [uid.strip() for uid in ids_to_delete_str.split(',') if uid.strip()]
                if ids_to_delete:
                    confirm = input(f"Delete {len(ids_to_delete)} memories? (yes/no): ").lower().strip()
                    if confirm == 'yes':
                        deleted_count = memory_manager.delete_memories(user_id=user_id_to_delete_by_id, ids=ids_to_delete)
                        print(f"🗑️ Deleted {deleted_count} memories")
                    else:
                        print("Deletion cancelled")

            elif action == '9':
                user_email_for_content_delete = input("Enter user's email (leave blank for all users): ").strip()
                user_id_for_content_delete = memory_manager.get_or_create_user_id(user_email_for_content_delete) if user_email_for_content_delete else None
                
                content_substring = input("Enter content substring to match for deletion: ").strip()
                if content_substring:
                    confirm = input(f"Delete memories containing '{content_substring}'? (yes/no): ").lower().strip()
                    if confirm == 'yes':
                        deleted_count = memory_manager.delete_memories(user_id=user_id_for_content_delete, content_contains=content_substring)
                        print(f"🗑️ Deleted {deleted_count} memories")
                    else:
                        print("Deletion cancelled")

            elif action == '10':
                email_to_delete_all = input("Enter user's email to delete ALL their memories: ").strip()
                if email_to_delete_all:
                    user_id_to_delete_all = memory_manager.get_or_create_user_id(email_to_delete_all)
                    user_name_for_confirm = memory_manager.get_user_name(user_id_to_delete_all) or user_id_to_delete_all
                    confirm = input(f"WARNING: Delete ALL memories for '{user_name_for_confirm}'? (yes/no): ").lower().strip()
                    if confirm == 'yes':
                        deleted_count = memory_manager.delete_all_user_memories(user_id_to_delete_all)
                        print(f"🗑️ Deleted {deleted_count} memories")
                    else:
                        print("Deletion cancelled")

            elif action == '11':
                stats = memory_manager.get_memory_stats()
                print(f"\n📊 Enhanced Memory Statistics")
                print("=" * 80)
                print(f"Persistent Memories: {stats.get('total_persistent_memories', 0)}")
                print(f"Flash Memories: {stats.get('total_flash_memories', 0)}")
                print(f"Collection: {stats.get('collection_name', 'Unknown')}")
                
                # Flash memory stats
                flash_stats = stats.get('flash_memory_by_user', {})
                if flash_stats:
                    print(f"\n🔥 Flash Memory by User:")
                    for user, count in sorted(flash_stats.items(), key=lambda x: x[1], reverse=True):
                        print(f"  • {user}: {count} turns")
                
                # Session stats
                session_stats = stats.get('active_sessions', {})
                if session_stats and session_stats.get('total_active_sessions', 0) > 0:
                    print(f"\n🔄 Active Sessions:")
                    print(f"  Total: {session_stats['total_active_sessions']}")
                    for user, session_info in session_stats.get('sessions_by_user', {}).items():
                        print(f"  • {user}: {session_info['interactions']} interactions, {session_info['duration_minutes']}min")
                
                # Enhanced stats display
                for stat_name, stat_data in [
                    ("Users", "memories_by_user"),
                    ("Contexts", "memories_by_context"),
                    ("Types", "memories_by_type"),
                    ("Importance", "memories_by_importance"),
                    ("Emotions", "memories_by_emotion")
                ]:
                    stat_dict = stats.get(stat_data, {})
                    if stat_dict:
                        print(f"\n🏷️ {stat_name}:")
                        for item, count in sorted(stat_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
                            print(f"  • {item}: {count}")
                
                insights = stats.get('insights', [])
                if insights:
                    print(f"\n💡 Insights:")
                    for insight in insights:
                        print(f"  • {insight}")
                print("=" * 80)

            elif action == '12':
                test_email = input("Enter an email to get/create user ID: ").strip()
                if test_email:
                    test_user_name = input("Enter a default name (optional): ").strip()
                    user_id_fetched = memory_manager.get_or_create_user_id(test_email, test_user_name if test_user_name else None)
                    retrieved_name = memory_manager.get_user_name(user_id_fetched)
                    print(f"User ID for '{test_email}': {user_id_fetched}")
                    print(f"Associated Name: {retrieved_name if retrieved_name else 'None'}")

            elif action == '13':
                test_text = input("Enter text to test emotion detection: ").strip()
                if test_text:
                    emotion, emotion_scores = memory_manager._detect_emotion_with_scores(test_text)
                    sentiment = memory_manager._calculate_sentiment_valence(test_text, emotion_scores)
                    importance = memory_manager._determine_importance(test_text, "general", emotion_scores)
                    
                    print(f"\n🧠 Emotion Analysis Results:")
                    print(f"Dominant Emotion: {emotion or 'None detected'}")
                    print(f"Emotion Scores: {emotion_scores}")
                    print(f"Sentiment Valence: {sentiment}")
                    print(f"Importance: {importance}")

            elif action == '14':
                email_for_session_end = input("Enter user's email to force end session: ").strip()
                if email_for_session_end:
                    user_id_for_session = memory_manager.get_or_create_user_id(email_for_session_end)
                    user_name_for_session = memory_manager.get_user_name(user_id_for_session)
                    
                    if user_id_for_session in memory_manager._active_sessions:
                        session_info = memory_manager._active_sessions[user_id_for_session]
                        print(f"Active session found for {user_name_for_session or user_id_for_session}:")
                        print(f"  Session ID: {session_info['session_id']}")
                        print(f"  Interactions: {len(session_info['interactions'])}")
                        print(f"  Started: {session_info['start_time']}")
                        
                        confirm = input("Force end this session and create episode? (yes/no): ").lower().strip()
                        if confirm == 'yes':
                            memory_manager.force_end_session(user_id_for_session)
                            print("✅ Session ended and episode creation initiated")
                        else:
                            print("Session end cancelled")
                    else:
                        print(f"No active session found for {user_name_for_session or user_id_for_session}")

            elif action == '15':
                session_stats = memory_manager.get_active_session_stats()
                print(f"\n🔄 Active Session Statistics")
                print("=" * 60)
                print(f"Total Active Sessions: {session_stats['total_active_sessions']}")
                
                if session_stats['sessions_by_user']:
                    print(f"\nSession Details:")
                    for user, info in session_stats['sessions_by_user'].items():
                        print(f"  👤 {user}:")
                        print(f"    • Interactions: {info['interactions']}")
                        print(f"    • Duration: {info['duration_minutes']} minutes")
                        print(f"    • Last Activity: {info['last_activity']}")
                
                if session_stats['session_durations']:
                    avg_duration = sum(session_stats['session_durations']) / len(session_stats['session_durations'])
                    print(f"\nAverage Session Duration: {avg_duration:.1f} minutes")
                
                print("=" * 60)

            elif action == '16':
                print("Checking for expired sessions...")
                initial_count = len(memory_manager._active_sessions)
                memory_manager.cleanup_expired_sessions()
                # Note: This is synchronous cleanup, actual episode creation happens asynchronously
                remaining_count = len(memory_manager._active_sessions)
                expired_count = initial_count - remaining_count
                
                if expired_count > 0:
                    print(f"✅ Found {expired_count} expired sessions, episode creation initiated")
                else:
                    print("No expired sessions found")

            elif action == '17':
                print("Exiting Enhanced MemoryManagerV4 with Session Episodes console.")
                break

            else:
                print("Invalid action. Please choose from the available options.")

    except Exception as e:
        logger.critical(f"Critical error in enhanced console: {e}", exc_info=True)
        print(f"\n❌ A critical error occurred: {e}")