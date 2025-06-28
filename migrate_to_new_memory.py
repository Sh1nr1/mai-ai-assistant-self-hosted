#!/usr/bin/env python3
"""
Memory Migration Script for Mai v4 Enhanced Memory System

Migrates legacy memories from ChromaDB collection to include new metadata fields
compatible with MemoryManagerV4's enhanced memory system.

Usage:
    python migrate_to_new_memory.py --user_id <uuid> --simulate True
    python migrate_to_new_memory.py --migrate_all True --overwrite False
    python migrate_to_new_memory.py --generate_episodes True --session_gap_minutes 15
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid
from collections import defaultdict

# Import the enhanced memory manager
try:
    from memory_manager import MemoryManager
except ImportError:
    print("❌ Error: Could not import MemoryManagerV4. Ensure memory_manager_v4.py is in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryMigrator:
    """
    Handles migration of legacy memories to enhanced v4 format.
    """
    
    def __init__(self, persist_directory: str = "./mai_memory", collection_name: str = "mai_memories"):
        """
        Initialize the memory migrator.
        
        Args:
            persist_directory: Directory containing ChromaDB data
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.migration_log = []
        self.session_gap_minutes = 15  # Default session gap
        
        try:
            # Initialize enhanced memory manager
            self.memory_manager = MemoryManagerV4(
                collection_name=collection_name,
                persist_directory=str(persist_directory)
            )
            logger.info(f"✅ Connected to memory collection: {collection_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize memory manager: {e}")
            raise

    def load_all_memories(self) -> Dict[str, List[Dict]]:
        """
        Load all memories from the collection, grouped by user.
        
        Returns:
            Dictionary mapping user_id to list of memory dictionaries
        """
        try:
            logger.info("📖 Loading all memories from collection...")
            
            # Get all memories with metadata
            results = self.memory_manager.collection.get(
                include=["documents", "metadatas"]
            )
            
            if not results["documents"]:
                logger.warning("No memories found in collection")
                return {}
            
            # Group memories by user_id
            user_memories = defaultdict(list)
            
            for i, document in enumerate(results["documents"]):
                metadata = results["metadatas"][i]
                memory_id = results["ids"][i]
                user_id = metadata.get("user_id", "unknown")
                
                memory_dict = {
                    "id": memory_id,
                    "content": document,
                    "metadata": metadata,
                    "original_metadata": metadata.copy()  # Keep original for comparison
                }
                
                user_memories[user_id].append(memory_dict)
            
            logger.info(f"📊 Loaded {len(results['documents'])} memories for {len(user_memories)} users")
            return dict(user_memories)
            
        except Exception as e:
            logger.error(f"❌ Failed to load memories: {e}")
            raise

    def detect_sessions(self, memories: List[Dict], gap_minutes: int = 15) -> List[List[Dict]]:
        """
        Detect conversation sessions based on timestamp gaps.
        
        Args:
            memories: List of memory dictionaries for a user
            gap_minutes: Minutes gap to consider a new session
            
        Returns:
            List of sessions, where each session is a list of memories
        """
        if not memories:
            return []
        
        # Sort memories by timestamp
        sorted_memories = sorted(
            memories,
            key=lambda m: m["metadata"].get("timestamp", "")
        )
        
        sessions = []
        current_session = []
        last_timestamp = None
        
        for memory in sorted_memories:
            timestamp_str = memory["metadata"].get("timestamp")
            
            if not timestamp_str:
                # If no timestamp, add to current session
                current_session.append(memory)
                continue
            
            try:
                current_timestamp = datetime.fromisoformat(timestamp_str)
                
                # Check if this starts a new session
                if (last_timestamp and 
                    (current_timestamp - last_timestamp).total_seconds() > (gap_minutes * 60)):
                    
                    # Save current session and start new one
                    if current_session:
                        sessions.append(current_session)
                    current_session = [memory]
                else:
                    current_session.append(memory)
                
                last_timestamp = current_timestamp
                
            except Exception as e:
                logger.warning(f"Could not parse timestamp '{timestamp_str}': {e}")
                current_session.append(memory)
        
        # Add the last session
        if current_session:
            sessions.append(current_session)
        
        logger.debug(f"Detected {len(sessions)} sessions from {len(memories)} memories")
        return sessions

    def classify_legacy_importance(self, content: str, metadata: Dict) -> str:
        """
        Classify importance of legacy memory content.
        
        Args:
            content: Memory content text
            metadata: Existing metadata
            
        Returns:
            Importance level: 'high', 'medium', or 'low'
        """
        content_lower = content.lower()
        
        # Check for high importance indicators
        high_importance_indicators = [
            'love', 'hate', 'amazing', 'terrible', 'breakthrough', 'crisis',
            'emergency', 'urgent', 'critical', 'important', 'significant',
            'milestone', 'achievement', 'failure', 'success', 'problem',
            'help me', 'advice', 'guidance', 'decision', 'choose'
        ]
        
        # Check for emotional intensity
        emotional_indicators = [
            'feel', 'emotion', 'heart', 'soul', 'devastated', 'ecstatic',
            'terrified', 'thrilled', 'heartbroken', 'overjoyed'
        ]
        
        # Check existing emotion metadata
        existing_emotion = metadata.get("emotion") or metadata.get("user_emotion")
        
        # Scoring system
        importance_score = 0
        
        # High importance keywords
        high_matches = sum(1 for indicator in high_importance_indicators 
                          if indicator in content_lower)
        importance_score += high_matches * 2
        
        # Emotional content
        emotion_matches = sum(1 for indicator in emotional_indicators 
                             if indicator in content_lower)
        importance_score += emotion_matches * 1.5
        
        # Existing emotion classification
        if existing_emotion and existing_emotion != 'neutral':
            importance_score += 1
        
        # Length consideration (longer messages often more important)
        if len(content) > 200:
            importance_score += 1
        
        # Questions are often important
        if '?' in content:
            importance_score += 0.5
        
        # Classify based on score
        if importance_score >= 3:
            return 'high'
        elif importance_score >= 1:
            return 'medium'
        else:
            return 'low'

    def determine_legacy_lifespan(self, importance: str, content: str, metadata: Dict) -> str:
        """
        Determine lifespan for legacy memory.
        
        Args:
            importance: Classified importance level
            content: Memory content
            metadata: Existing metadata
            
        Returns:
            Lifespan: 'permanent', 'volatile', or 'ephemeral'
        """
        # Core/important memories should be permanent
        if importance == 'high':
            return 'permanent'
        
        # Check for personal/emotional content
        personal_indicators = ['personal', 'family', 'relationship', 'love', 'trust', 'feel']
        content_lower = content.lower()
        
        has_personal_content = any(indicator in content_lower for indicator in personal_indicators)
        
        if has_personal_content and importance == 'medium':
            return 'permanent'
        
        # Check for technical/business content
        technical_indicators = ['project', 'code', 'system', 'technical', 'business', 'work']
        has_technical_content = any(indicator in content_lower for indicator in technical_indicators)
        
        if has_technical_content:
            return 'volatile' if importance == 'medium' else 'ephemeral'
        
        # Default classification
        return 'volatile' if importance == 'medium' else 'ephemeral'

    def classify_legacy_type(self, content: str, metadata: Dict) -> str:
        """
        Classify memory type for legacy content.
        
        Args:
            content: Memory content
            metadata: Existing metadata
            
        Returns:
            Memory type: 'core', 'episodic', 'conversational', 'volatile'
        """
        content_lower = content.lower()
        speaker = metadata.get("speaker", "")
        
        # Check for core memory indicators
        core_indicators = [
            'personality', 'identity', 'values', 'beliefs', 'core', 'fundamental',
            'who i am', 'what i believe', 'my nature', 'my purpose'
        ]
        
        if any(indicator in content_lower for indicator in core_indicators):
            return 'core'
        
        # Check for system-generated content
        if speaker == 'system' or 'summary' in content_lower or 'episode' in content_lower:
            return 'episodic'
        
        # Check for temporary/volatile content
        volatile_indicators = ['temp', 'temporary', 'quick', 'just checking', 'test']
        if any(indicator in content_lower for indicator in volatile_indicators):
            return 'volatile'
        
        # Default to conversational
        return 'conversational'

    def generate_emotion_scores(self, content: str) -> Tuple[Dict[str, float], float]:
        """
        Generate emotion scores and sentiment valence for legacy content.
        
        Args:
            content: Memory content text
            
        Returns:
            Tuple of (emotion_scores_dict, sentiment_valence)
        """
        try:
            # Use enhanced memory manager's emotion detection
            emotion, emotion_scores = self.memory_manager._detect_emotion_with_scores(content)
            sentiment_valence = self.memory_manager._calculate_sentiment_valence(content, emotion_scores)
            
            return emotion_scores, sentiment_valence
            
        except Exception as e:
            logger.warning(f"Failed to generate emotion scores: {e}")
            
            # Fallback to simple keyword-based analysis
            emotion_scores = {}
            sentiment_valence = 0.0
            
            content_lower = content.lower()
            
            # Simple positive/negative word counting
            positive_words = ['good', 'great', 'love', 'happy', 'amazing', 'wonderful', 'excellent']
            negative_words = ['bad', 'hate', 'sad', 'terrible', 'awful', 'horrible', 'worst']
            
            positive_count = sum(1 for word in positive_words if word in content_lower)
            negative_count = sum(1 for word in negative_words if word in content_lower)
            
            if positive_count > 0 or negative_count > 0:
                total_count = positive_count + negative_count
                sentiment_valence = (positive_count - negative_count) / total_count
                
                if positive_count > negative_count:
                    emotion_scores = {'joy': 0.7}
                elif negative_count > positive_count:
                    emotion_scores = {'sadness': 0.7}
            
            return emotion_scores, sentiment_valence

    def upgrade_memory_metadata(self, memory: Dict, simulate: bool = True) -> Dict:
        """
        Upgrade a single memory's metadata to v4 format.
        
        Args:
            memory: Memory dictionary with content and metadata
            simulate: If True, don't actually update the database
            
        Returns:
            Updated memory dictionary
        """
        content = memory["content"]
        metadata = memory["metadata"].copy()
        memory_id = memory["id"]
        
        # Store original metadata for logging
        original_metadata = memory["original_metadata"]
        
        # Check if already migrated (has new v4 fields)
        if "importance" in metadata and "lifespan" in metadata and "type" in metadata:
            logger.debug(f"Memory {memory_id} already migrated. Skipping.")
            
            # --- Proactively check and fix incompatible data types from previous attempts ---
            fixes_applied = False
            if "emotion_scores" in metadata and isinstance(metadata["emotion_scores"], dict):
                metadata["emotion_scores"] = json.dumps(metadata["emotion_scores"])
                fixes_applied = True

            if "linked_to" in metadata and isinstance(metadata["linked_to"], list):
                metadata["linked_to"] = ",".join(metadata["linked_to"]) if metadata["linked_to"] else ""
                fixes_applied = True
            
            if fixes_applied and not simulate:
                try:
                    self.memory_manager.collection.update(ids=[memory_id], metadatas=[metadata])
                    logger.info(f"🔧 Fixed data compatibility for memory {memory_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to apply compatibility fix for memory {memory_id}: {e}")
            
            memory["metadata"] = metadata
            return memory

        logger.debug(f"🔄 Upgrading memory {memory_id}...")
        
        upgrades = {}
        
        # Classify importance, lifespan, and type
        importance = self.classify_legacy_importance(content, metadata)
        metadata["importance"] = importance
        
        lifespan = self.determine_legacy_lifespan(importance, content, metadata)
        metadata["lifespan"] = lifespan
        
        memory_type = self.classify_legacy_type(content, metadata)
        metadata["type"] = memory_type
        
        # Generate emotion scores and sentiment
        emotion_scores, sentiment_valence = self.generate_emotion_scores(content)
        
        # =================================================================
        # THE FIX: Convert the emotion_scores dictionary to a JSON string
        # =================================================================
        metadata["emotion_scores"] = json.dumps(emotion_scores)
        metadata["sentiment_valence"] = sentiment_valence

        # Initialize other new fields
        metadata["linked_to"] = ""
        metadata["is_validated"] = True
        
        timestamp = metadata.get("timestamp")
        metadata["time_of_day"] = self.memory_manager._determine_time_of_day(timestamp)

        # Log the full upgrade details
        upgrades = {
            "importance": importance,
            "lifespan": lifespan,
            "type": memory_type,
            "emotion_scores": emotion_scores, # Log the dict, not the string
            "sentiment_valence": sentiment_valence,
            "linked_to": "",
            "is_validated": True,
            "time_of_day": metadata["time_of_day"]
        }
        
        upgrade_log = {
            "memory_id": memory_id,
            "content_preview": content[:50] + "..." if len(content) > 50 else content,
            "original_metadata": original_metadata,
            "upgrades_applied": upgrades,
            "new_metadata_saved": metadata, # Log the actual data being saved
            "timestamp": datetime.now().isoformat()
        }
        self.migration_log.append(upgrade_log)
        
        # Update in database if not simulating
        if not simulate:
            try:
                self.memory_manager.collection.update(
                    ids=[memory_id],
                    metadatas=[metadata]
                )
                logger.debug(f"✅ Updated memory {memory_id} in database")
            except Exception as e:
                # Log the error but DO NOT raise it, allowing the migration to continue
                error_msg = f"Failed to update memory {memory_id}: {e}"
                logger.error(f"❌ {error_msg}")
                upgrade_log["error"] = error_msg # Add the error to the log entry
        else:
            logger.debug(f"🔍 [SIMULATE] Would update memory {memory_id}")
        
        # Update the memory object in-place for any subsequent steps in the migration
        memory["metadata"] = metadata
        return memory
        
        logger.debug(f"🔄 Upgrading memory {memory_id}...")
        
        # Add new metadata fields
        upgrades = {}
        
        # Classify importance
        if "importance" not in metadata:
            importance = self.classify_legacy_importance(content, metadata)
            upgrades["importance"] = importance
            metadata["importance"] = importance
        
        # Determine lifespan
        if "lifespan" not in metadata:
            lifespan = self.determine_legacy_lifespan(
                metadata.get("importance", "medium"), content, metadata
            )
            upgrades["lifespan"] = lifespan
            metadata["lifespan"] = lifespan
        
        # Classify type
        if "type" not in metadata:
            memory_type = self.classify_legacy_type(content, metadata)
            upgrades["type"] = memory_type
            metadata["type"] = memory_type
        
        # Generate emotion scores
        if "emotion_scores" not in metadata:
            emotion_scores, sentiment_valence = self.generate_emotion_scores(content)
            upgrades["emotion_scores"] = emotion_scores
            upgrades["sentiment_valence"] = sentiment_valence
            metadata["emotion_scores"] = emotion_scores
            metadata["sentiment_valence"] = sentiment_valence
        
        # Initialize linked_to field
        if "linked_to" not in metadata:
            upgrades["linked_to"] = ""  # Empty string instead of empty list
            metadata["linked_to"] = ""
        
        # Set validation flag
        if "is_validated" not in metadata:
            upgrades["is_validated"] = True
            metadata["is_validated"] = True
        
        # Ensure time_of_day is set
        if "time_of_day" not in metadata:
            timestamp = metadata.get("timestamp")
            time_of_day = self.memory_manager._determine_time_of_day(timestamp)
            upgrades["time_of_day"] = time_of_day
            metadata["time_of_day"] = time_of_day
        
        # Log the upgrade
        upgrade_log = {
            "memory_id": memory_id,
            "content_preview": content[:50] + "..." if len(content) > 50 else content,
            "original_metadata": original_metadata,
            "upgrades_applied": upgrades,
            "new_metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        self.migration_log.append(upgrade_log)
        
        # Update in database if not simulating
        if not simulate:
            try:
                self.memory_manager.collection.update(
                    ids=[memory_id],
                    metadatas=[metadata]
                )
                logger.debug(f"✅ Updated memory {memory_id} in database")
            except Exception as e:
                logger.error(f"❌ Failed to update memory {memory_id}: {e}")
                raise
        else:
            logger.debug(f"🔍 [SIMULATE] Would update memory {memory_id}")
        
        # Update the memory object
        memory["metadata"] = metadata
        return memory

    def generate_session_summary(self, session_memories: List[Dict]) -> str:
        """
        Generate a summary for a session of memories.
        
        Args:
            session_memories: List of memories in the session
            
        Returns:
            Session summary text
        """
        if not session_memories:
            return "Empty session"
        
        # Extract key information
        total_messages = len(session_memories)
        user_messages = [m for m in session_memories if m["metadata"].get("speaker") == "user"]
        ai_messages = [m for m in session_memories if m["metadata"].get("speaker") == "ai"]
        
        # Get session timeframe
        timestamps = [m["metadata"].get("timestamp") for m in session_memories if m["metadata"].get("timestamp")]
        
        session_start = "unknown"
        session_end = "unknown"
        
        if timestamps:
            try:
                sorted_timestamps = sorted([datetime.fromisoformat(ts) for ts in timestamps])
                session_start = sorted_timestamps[0].strftime("%Y-%m-%d %H:%M")
                session_end = sorted_timestamps[-1].strftime("%Y-%m-%d %H:%M")
            except:
                pass
        
        # Detect dominant emotions
        all_emotions = []
        for memory in session_memories:
            emotion = memory["metadata"].get("emotion")
            if emotion:
                all_emotions.append(emotion)
        
        emotion_summary = ""
        if all_emotions:
            emotion_counts = {}
            for emotion in all_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
            emotion_summary = f"Dominant emotion: {dominant_emotion}. "
        
        # Detect main topics (simple keyword extraction)
        all_content = " ".join([m["content"] for m in session_memories])
        content_lower = all_content.lower()
        
        topic_keywords = {
            'work': ['work', 'job', 'career', 'project', 'business'],
            'personal': ['feel', 'think', 'personal', 'life', 'family'],
            'technical': ['code', 'system', 'technical', 'programming', 'software'],
            'creative': ['art', 'design', 'creative', 'music', 'writing'],
            'learning': ['learn', 'study', 'understand', 'knowledge', 'skill']
        }
        
        detected_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_topics.append(topic)
        
        topic_summary = f"Topics: {', '.join(detected_topics)}. " if detected_topics else ""
        
        # Generate summary
        summary = (f"Session from {session_start} to {session_end} with {total_messages} messages "
                  f"({len(user_messages)} user, {len(ai_messages)} AI). "
                  f"{emotion_summary}{topic_summary}"
                  f"Key themes discussed during this conversation period.")
        
        return summary

    def store_session_episode(self, session_memories: List[Dict], user_id: str, simulate: bool = True) -> bool:
        """
        Store a session as an episodic memory.
        
        Args:
            session_memories: List of memories in the session
            user_id: User identifier
            simulate: If True, don't actually store
            
        Returns:
            True if successfully stored
        """
        try:
            # Generate session summary
            summary = self.generate_session_summary(session_memories)
            
            # Get user name
            user_name = self.memory_manager.get_user_name(user_id)
            
            # Determine session context
            if len(session_memories) >= 5:
                episode_context = "extended_conversation"
            elif any('emotion' in m["metadata"] for m in session_memories):
                episode_context = "emotional_exchange"
            else:
                episode_context = "brief_interaction"
            
            if not simulate:
                # Store the episode
                success = self.memory_manager.store_episode_summary(
                    summary_text=summary,
                    user_id=user_id,
                    episode_context=episode_context,
                    user_name=user_name
                )
                
                if success:
                    logger.info(f"✅ Stored episode for user {user_id}: {len(session_memories)} memories")
                    
                    # Log the episode creation
                    episode_log = {
                        "user_id": user_id,
                        "user_name": user_name,
                        "episode_context": episode_context,
                        "session_memory_count": len(session_memories),
                        "summary": summary,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.migration_log.append(episode_log)
                    
                return success
            else:
                logger.info(f"🔍 [SIMULATE] Would store episode for user {user_id}: {len(session_memories)} memories")
                logger.info(f"    Summary: {summary[:100]}...")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to store session episode: {e}")
            return False

    def link_session_memories(self, session_memories: List[Dict], simulate: bool = True):
        """
        Link memories within a session to each other.
        
        Args:
            session_memories: List of memories in the session
            simulate: If True, don't actually update database
        """
        if len(session_memories) < 2:
            return
        
        try:
            # Create bidirectional links between adjacent memories
            for i, memory in enumerate(session_memories):
                memory_id = memory["id"]
                linked_ids = []
                
                # Link to previous memory
                if i > 0:
                    linked_ids.append(session_memories[i-1]["id"])
                
                # Link to next memory
                if i < len(session_memories) - 1:
                    linked_ids.append(session_memories[i+1]["id"])
                
                if linked_ids and not simulate:
                    self.memory_manager.link_memory_to_others(memory_id, linked_ids)
                    logger.debug(f"🔗 Linked memory {memory_id} to {len(linked_ids)} others")
                elif linked_ids:
                    logger.debug(f"🔍 [SIMULATE] Would link memory {memory_id} to {len(linked_ids)} others")
                    
        except Exception as e:
            logger.error(f"❌ Failed to link session memories: {e}")

    def cleanup_incompatible_data(self, simulate: bool = True) -> int:
        """
        Clean up any existing incompatible data types in the database.
        
        Args:
            simulate: If True, don't actually update the database
            
        Returns:
            Number of memories cleaned up
        """
        try:
            logger.info("🧹 Starting cleanup of incompatible data types...")
            
            # Get all memories
            results = self.memory_manager.collection.get(include=["metadatas"])
            
            if not results["metadatas"]:
                logger.info("No memories found for cleanup")
                return 0
            
            cleanup_count = 0
            
            for i, metadata in enumerate(results["metadatas"]):
                memory_id = results["ids"][i]
                fixes_needed = False
                
                # Fix emotion_scores if it's a dict
                if "emotion_scores" in metadata:
                    if isinstance(metadata["emotion_scores"], dict):
                        metadata["emotion_scores"] = json.dumps(metadata["emotion_scores"])
                        fixes_needed = True
                        logger.debug(f"Fixed emotion_scores dict for {memory_id}")
                    elif metadata["emotion_scores"] == {}:  # Handle empty dict objects
                        metadata["emotion_scores"] = "{}"
                        fixes_needed = True
                        logger.debug(f"Fixed empty emotion_scores dict for {memory_id}")
                
                # Fix linked_to if it's a list
                if "linked_to" in metadata and isinstance(metadata["linked_to"], list):
                    metadata["linked_to"] = ",".join(metadata["linked_to"]) if metadata["linked_to"] else ""
                    fixes_needed = True
                    logger.debug(f"Fixed linked_to for {memory_id}")
                
                # Apply fixes if needed
                if fixes_needed:
                    cleanup_count += 1
                    if not simulate:
                        try:
                            self.memory_manager.collection.update(
                                ids=[memory_id],
                                metadatas=[metadata]
                            )
                        except Exception as e:
                            logger.error(f"Failed to cleanup memory {memory_id}: {e}")
                            continue
                    else:
                        logger.debug(f"[SIMULATE] Would cleanup memory {memory_id}")
            
            logger.info(f"✅ Cleaned up {cleanup_count} memories with incompatible data types")
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup incompatible data: {e}")
            return 0

    def migrate_user_memories(
        self, 
        user_id: str, 
        memories: List[Dict], 
        simulate: bool = True,
        generate_episodes: bool = True,
        link_sessions: bool = True
    ) -> Dict:
        """
        Migrate all memories for a specific user.
        
        Args:
            user_id: User identifier
            memories: List of memory dictionaries for the user
            simulate: If True, don't actually update database
            generate_episodes: Whether to generate episodic memories from sessions
            link_sessions: Whether to link memories within sessions
            
        Returns:
            Migration results dictionary
        """
        logger.info(f"🚀 Starting migration for user {user_id} ({len(memories)} memories)")
        
        results = {
            "user_id": user_id,
            "total_memories": len(memories),
            "upgraded_memories": 0,
            "episodes_created": 0,
            "sessions_detected": 0,
            "errors": []
        }
        
        try:
            # Detect sessions
            sessions = self.detect_sessions(memories, self.session_gap_minutes)
            results["sessions_detected"] = len(sessions)
            
            logger.info(f"📊 Detected {len(sessions)} sessions for user {user_id}")
            
            # Process each session
            for i, session_memories in enumerate(sessions):
                logger.debug(f"Processing session {i+1}/{len(sessions)} ({len(session_memories)} memories)")
                
                # Upgrade metadata for each memory in session
                for memory in session_memories:
                    try:
                        self.upgrade_memory_metadata(memory, simulate)
                        results["upgraded_memories"] += 1
                    except Exception as e:
                        error_msg = f"Failed to upgrade memory {memory['id']}: {e}"
                        logger.error(f"❌ {error_msg}")
                        results["errors"].append(error_msg)
                
                # Generate episode if session has enough meaningful content
                if generate_episodes and len(session_memories) >= 3:  # Minimum 3 memories for episode
                    try:
                        episode_created = self.store_session_episode(session_memories, user_id, simulate)
                        if episode_created:
                            results["episodes_created"] += 1
                    except Exception as e:
                        error_msg = f"Failed to create episode for session {i+1}: {e}"
                        logger.error(f"❌ {error_msg}")
                        results["errors"].append(error_msg)
                
                # Link memories within session
                if link_sessions:
                    try:
                        self.link_session_memories(session_memories, simulate)
                    except Exception as e:
                        error_msg = f"Failed to link session {i+1} memories: {e}"
                        logger.warning(f"⚠️ {error_msg}")
                        results["errors"].append(error_msg)
            
            logger.info(f"✅ Completed migration for user {user_id}")
            
        except Exception as e:
            error_msg = f"Critical error during user {user_id} migration: {e}"
            logger.error(f"❌ {error_msg}")
            results["errors"].append(error_msg)
        
        return results

    def migrate_all_users(
        self, 
        simulate: bool = True,
        generate_episodes: bool = True,
        link_sessions: bool = True,
        target_user_id: Optional[str] = None,
        cleanup_first: bool = True
    ) -> Dict:
        """
        Migrate memories for all users or a specific user.
        
        Args:
            simulate: If True, don't actually update database
            generate_episodes: Whether to generate episodic memories
            link_sessions: Whether to link session memories
            target_user_id: If provided, only migrate this user
            cleanup_first: Whether to run data cleanup first
            
        Returns:
            Overall migration results
        """
        logger.info("🚀 Starting comprehensive memory migration...")
        
        # Step 1: Cleanup incompatible data types first
        cleanup_count = 0
        if cleanup_first:
            logger.info("🧹 Step 1: Cleaning up incompatible data types...")
            cleanup_count = self.cleanup_incompatible_data(simulate)
        
        # Step 2: Load all memories
        user_memories = self.load_all_memories()
        
        if not user_memories:
            logger.warning("No memories found to migrate")
            return {"error": "No memories found"}
        
        # Filter to specific user if requested
        if target_user_id:
            if target_user_id in user_memories:
                user_memories = {target_user_id: user_memories[target_user_id]}
                logger.info(f"🎯 Targeting specific user: {target_user_id}")
            else:
                logger.error(f"❌ User {target_user_id} not found in memories")
                return {"error": f"User {target_user_id} not found"}
        
        overall_results = {
            "migration_started": datetime.now().isoformat(),
            "simulate_mode": simulate,
            "cleanup_count": cleanup_count,
            "total_users": len(user_memories),
            "user_results": {},
            "summary": {
                "total_memories_processed": 0,
                "total_memories_upgraded": 0,
                "total_episodes_created": 0,
                "total_sessions_detected": 0,
                "total_errors": 0
            }
        }
        
        # Process each user
        for user_id, memories in user_memories.items():
            try:
                user_results = self.migrate_user_memories(
                    user_id=user_id,
                    memories=memories,
                    simulate=simulate,
                    generate_episodes=generate_episodes,
                    link_sessions=link_sessions
                )
                
                overall_results["user_results"][user_id] = user_results
                
                # Update summary
                overall_results["summary"]["total_memories_processed"] += user_results["total_memories"]
                overall_results["summary"]["total_memories_upgraded"] += user_results["upgraded_memories"]
                overall_results["summary"]["total_episodes_created"] += user_results["episodes_created"]
                overall_results["summary"]["total_sessions_detected"] += user_results["sessions_detected"]
                overall_results["summary"]["total_errors"] += len(user_results["errors"])
                
            except Exception as e:
                logger.error(f"❌ Failed to process user {user_id}: {e}")
                overall_results["user_results"][user_id] = {"error": str(e)}
                overall_results["summary"]["total_errors"] += 1
        
        overall_results["migration_completed"] = datetime.now().isoformat()
        
        # Log summary
        summary = overall_results["summary"]
        logger.info("📊 Migration Summary:")
        if cleanup_count > 0:
            logger.info(f"   Data cleanup: {cleanup_count} memories fixed")
        logger.info(f"   Users processed: {overall_results['total_users']}")
        logger.info(f"   Memories processed: {summary['total_memories_processed']}")
        logger.info(f"   Memories upgraded: {summary['total_memories_upgraded']}")
        logger.info(f"   Episodes created: {summary['total_episodes_created']}")
        logger.info(f"   Sessions detected: {summary['total_sessions_detected']}")
        logger.info(f"   Errors encountered: {summary['total_errors']}")
        
        return overall_results

    def save_migration_log(self, results: Dict, output_file: str = "migration_log.json"):
        """
        Save migration log and results to file.
        
        Args:
            results: Migration results dictionary
            output_file: Output file path
        """
        try:
            log_data = {
                "migration_results": results,
                "detailed_log": self.migration_log,
                "generated_at": datetime.now().isoformat()
            }
            
            output_path = Path(output_file)
            with open(output_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            logger.info(f"📄 Migration log saved to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save migration log: {e}")


def main():
    """Main CLI interface for the migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate Mai's legacy memories to enhanced v4 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simulate migration for specific user
  python migrate_to_new_memory.py --user_id "12345" --simulate

  # Actually migrate all users with episode generation
  python migrate_to_new_memory.py --migrate_all --generate_episodes

  # Migrate specific user without simulation
  python migrate_to_new_memory.py --user_id "12345" --no-simulate --overwrite

  # Custom session gap and output file
  python migrate_to_new_memory.py --migrate_all --session_gap_minutes 20 --output migration_report.json
        """
    )
    
    # Target options
    parser.add_argument(
        '--user_id',
        type=str,
        help='Migrate memories for specific user ID only'
    )
    
    parser.add_argument(
        '--migrate_all',
        action='store_true',
        help='Migrate memories for all users'
    )
    
    # Migration behavior
    parser.add_argument(
        '--simulate',
        action='store_true',
        default=True,
        help='Simulate migration without making changes (default: True)'
    )
    
    parser.add_argument(
        '--no-simulate',
        dest='simulate',
        action='store_false',
        help='Actually perform migration (disable simulation)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Allow overwriting existing memories (use with caution)'
    )
    
    # Feature options
    parser.add_argument(
        '--generate_episodes',
        action='store_true',
        default=True,
        help='Generate episodic memories from conversation sessions (default: True)'
    )
    
    parser.add_argument(
        '--no-episodes',
        dest='generate_episodes',
        action='store_false',
        help='Skip episodic memory generation'
    )
    
    parser.add_argument(
        '--link_sessions',
        action='store_true',
        default=True,
        help='Link memories within conversation sessions (default: True)'
    )
    
    parser.add_argument(
        '--no-linking',
        dest='link_sessions',
        action='store_false',
        help='Skip memory linking'
    )
    
    parser.add_argument(
        '--cleanup_only',
        action='store_true',
        help='Only run data cleanup without full migration'
    )
    
    parser.add_argument(
        '--no-cleanup',
        dest='cleanup_first',
        action='store_false',
        default=True,
        help='Skip initial data cleanup step'
    )
    
    # Configuration options
    parser.add_argument(
        '--session_gap_minutes',
        type=int,
        default=15,
        help='Minutes gap to consider a new conversation session (default: 15)'
    )
    
    parser.add_argument(
        '--persist_directory',
        type=str,
        default='./mai_memory',
        help='Directory containing ChromaDB data (default: ./mai_memory)'
    )
    
    parser.add_argument(
        '--collection_name',
        type=str,
        default='mai_memories',
        help='ChromaDB collection name (default: mai_memories)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='migration_log.json',
        help='Output file for migration log (default: migration_log.json)'
    )
    
    # Logging options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress most output except errors'
    )

    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Validation
    if args.cleanup_only:
        # Special mode: only run cleanup
        pass
    elif not args.user_id and not args.migrate_all:
        parser.error("Must specify either --user_id, --migrate_all, or --cleanup_only")
    elif args.user_id and args.migrate_all:
        parser.error("Cannot specify both --user_id and --migrate_all")
    
    # Safety check for non-simulation mode
    if not args.simulate and not args.overwrite:
        print("⚠️  WARNING: You are about to modify the database without simulation mode.")
        print("   This will permanently alter your memory collection.")
        print("   Use --overwrite flag to confirm you want to proceed.")
        confirm = input("   Continue anyway? (yes/no): ").lower().strip()
        if confirm != 'yes':
            print("❌ Migration cancelled for safety.")
            return
    
    try:
        # Initialize migrator
        print(f"🔧 Initializing Memory Migrator...")
        print(f"   Collection: {args.collection_name}")
        print(f"   Directory: {args.persist_directory}")
        print(f"   Simulate Mode: {args.simulate}")
        print()
        
        migrator = MemoryMigrator(
            persist_directory=args.persist_directory,
            collection_name=args.collection_name
        )
        
        # Set session gap
        migrator.session_gap_minutes = args.session_gap_minutes
        
        # Run cleanup only mode
        if args.cleanup_only:
            print("🧹 Running data cleanup only...")
            cleanup_count = migrator.cleanup_incompatible_data(simulate=args.simulate)
            
            print(f"\n🧹 CLEANUP RESULTS:")
            print(f"   Memories cleaned: {cleanup_count}")
            
            if args.simulate:
                print("🔍 SIMULATION - No changes were made")
            else:
                print("✅ CLEANUP COMPLETE - Database updated")
            
            return 0
        
        # Run migration
        if args.migrate_all:
            print("🚀 Starting migration for ALL users...")
            results = migrator.migrate_all_users(
                simulate=args.simulate,
                generate_episodes=args.generate_episodes,
                link_sessions=args.link_sessions,
                cleanup_first=args.cleanup_first
            )
        else:
            print(f"🎯 Starting migration for user: {args.user_id}")
            results = migrator.migrate_all_users(
                simulate=args.simulate,
                generate_episodes=args.generate_episodes,
                link_sessions=args.link_sessions,
                target_user_id=args.user_id,
                cleanup_first=args.cleanup_first
            )
        
        # Display results
        print("\n" + "="*60)
        print("📊 MIGRATION RESULTS")
        print("="*60)
        
        if "error" in results:
            print(f"❌ Migration failed: {results['error']}")
            return
        
        summary = results.get("summary", {})
        cleanup_count = results.get("cleanup_count", 0)
        
        if cleanup_count > 0:
            print(f"🧹 Data cleanup: {cleanup_count} memories fixed")
        print(f"👥 Users processed: {results.get('total_users', 0)}")
        print(f"💾 Memories processed: {summary.get('total_memories_processed', 0)}")
        print(f"⬆️  Memories upgraded: {summary.get('total_memories_upgraded', 0)}")
        print(f"📚 Episodes created: {summary.get('total_episodes_created', 0)}")
        print(f"🔗 Sessions detected: {summary.get('total_sessions_detected', 0)}")
        print(f"❌ Errors: {summary.get('total_errors', 0)}")
        
        # Show mode indicators
        mode_indicators = []
        if args.simulate:
            mode_indicators.append("🔍 SIMULATION")
        else:
            mode_indicators.append("✅ ACTUAL")
        
        if args.generate_episodes:
            mode_indicators.append("📚 EPISODES")
        
        if args.link_sessions:
            mode_indicators.append("🔗 LINKING")
        
        print(f"\n🎛️  Mode: {' | '.join(mode_indicators)}")
        
        # Save migration log
        print(f"\n📄 Saving migration log to {args.output}...")
        migrator.save_migration_log(results, args.output)
        
        # Final status
        if args.simulate:
            print("\n🔍 SIMULATION COMPLETE - No changes were made to the database")
            print("   Review the migration log and run with --no-simulate to apply changes")
        else:
            print("\n✅ MIGRATION COMPLETE - Database has been updated")
            print("   Check the migration log for detailed results")
        
        # Show any errors
        if summary.get('total_errors', 0) > 0:
            print(f"\n⚠️  {summary['total_errors']} errors occurred during migration")
            print("   Check the migration log for details")
        
    except KeyboardInterrupt:
        print("\n❌ Migration interrupted by user")
    except Exception as e:
        logger.error(f"❌ Migration failed with error: {e}")
        print(f"\n❌ Migration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)