"""
Memory Manager for Mai - Emotionally Intelligent AI Assistant
Handles long-term memory storage and retrieval using ChromaDB
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages Mai's long-term memory using ChromaDB for vector storage"""
    
    def __init__(self, persist_directory: str = "./mai_memory", collection_name: str = "mai_memories"):
        """
        Initialize the memory manager
        
        Args:
            persist_directory: Directory to store ChromaDB data
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embedding model (using a lightweight model for local use)
        try:
            logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Mai's long-term conversation memories"}
            )
            
            logger.info(f"ChromaDB initialized. Collection '{collection_name}' ready.")
            logger.info(f"Current memory count: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using SentenceTransformer
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def _create_memory_id(self) -> str:
        """Generate unique ID for memory entry"""
        return str(uuid.uuid4())

    def _extract_memory_content(self, user_message: str, ai_response: str) -> List[Dict]:
        """
        Extract meaningful memory content from conversation
        
        Args:
            user_message: User's message
            ai_response: Mai's response
            
        Returns:
            List of memory entries to store
        """
        memories = []
        timestamp = datetime.now().isoformat()
        
        # Store user message if it contains personal information or important context
        if self._is_memorable_content(user_message):
            memories.append({
                "content": user_message,
                "speaker": "user",
                "timestamp": timestamp,
                "context": "user_message"
            })
        
        # Store AI response if it contains relationship-building content
        if self._is_memorable_response(ai_response):
            memories.append({
                "content": ai_response,
                "speaker": "mai",
                "timestamp": timestamp,
                "context": "mai_response"
            })
        
        # Store conversation pair for context
        conversation_summary = f"User: {user_message[:200]}... Mai: {ai_response[:200]}..."
        memories.append({
            "content": conversation_summary,
            "speaker": "conversation",
            "timestamp": timestamp,
            "context": "conversation_pair"
        })
        
        return memories

    def _is_memorable_content(self, text: str) -> bool:
        """
        Determine if user content should be stored in long-term memory
        
        Args:
            text: Text to analyze
            
        Returns:
            True if content should be remembered
        """
        memorable_indicators = [
            # Personal information
            "my name is", "i'm", "i am", "i work", "i live", "my job",
            "my family", "my wife", "my husband", "my kids", "my children",
            "my parents", "my friend", "my pet", "my dog", "my cat",
            
            # Preferences and interests
            "i like", "i love", "i enjoy", "i prefer", "my favorite",
            "i hate", "i dislike", "i can't stand",
            
            # Important events
            "birthday", "anniversary", "wedding", "graduation", "promotion",
            "vacation", "trip", "moving", "bought", "sold", "started", "finished",
            
            # Emotional states and problems
            "i'm worried", "i'm excited", "i'm sad", "i'm happy", "i'm stressed",
            "problem", "issue", "challenge", "difficult", "struggling",
            
            # Goals and plans
            "planning", "hoping", "want to", "going to", "will", "goal", "dream"
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in memorable_indicators)

    def _is_memorable_response(self, text: str) -> bool:
        """
        Determine if Mai's response should be stored in long-term memory
        
        Args:
            text: Mai's response text
            
        Returns:
            True if response should be remembered
        """
        # Store responses that show relationship building or important advice
        memorable_response_indicators = [
            "remember", "recall", "mentioned", "told me", "shared",
            "advice", "suggest", "recommend", "important", "celebrate",
            "congratulations", "sorry to hear", "understand", "support"
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in memorable_response_indicators)

    def store_conversation(self, user_message: str, ai_response: str, user_id: str = "default_user") -> int:
        """
        Store a conversation in long-term memory
        
        Args:
            user_message: User's message
            ai_response: Mai's response
            user_id: Identifier for the user
            
        Returns:
            Number of memories stored
        """
        try:
            memories = self._extract_memory_content(user_message, ai_response)
            stored_count = 0
            
            for memory in memories:
                # Add user ID to metadata
                memory["user_id"] = user_id
                
                # Generate embedding
                embedding = self._generate_embedding(memory["content"])
                
                # Create unique ID
                memory_id = self._create_memory_id()
                
                # Store in ChromaDB
                self.collection.add(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[memory["content"]],
                    metadatas=[memory]
                )
                
                stored_count += 1
                logger.debug(f"Stored memory: {memory['context']} - {memory['content'][:50]}...")
            
            logger.info(f"Stored {stored_count} memories from conversation")
            return stored_count
            
        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")
            return 0

    def retrieve_memories(self, query: str, user_id: str = "default_user", 
                         max_memories: int = 5, relevance_threshold: float = 0.7) -> List[str]:
        """
        Retrieve relevant memories based on query
        
        Args:
            query: Search query (usually the user's current message)
            user_id: Identifier for the user
            max_memories: Maximum number of memories to return
            relevance_threshold: Minimum similarity score to include
            
        Returns:
            List of relevant memory strings
        """
        try:
            if self.collection.count() == 0:
                logger.info("No memories stored yet")
                return []
            
            # Generate embedding for query
            query_embedding = self._generate_embedding(query)
            
            # Search for similar memories
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_memories * 2,  # Get more results to filter by user_id
                where={"user_id": user_id}  # Filter by user ID
            )
            
            if not results["documents"] or not results["documents"][0]:
                logger.info("No relevant memories found")
                return []
            
            # Process and filter results
            relevant_memories = []
            documents = results["documents"][0]
            distances = results["distances"][0] if results["distances"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            
            for i, (doc, distance, metadata) in enumerate(zip(documents, distances, metadatas)):
                # Convert distance to similarity (lower distance = higher similarity)
                similarity = 1 - distance
                
                if similarity >= relevance_threshold:
                    # Format memory with context
                    memory_context = metadata.get("context", "unknown")
                    timestamp = metadata.get("timestamp", "unknown time")
                    formatted_memory = f"[{memory_context}] {doc}"
                    
                    relevant_memories.append(formatted_memory)
                    logger.debug(f"Retrieved memory (similarity: {similarity:.3f}): {doc[:50]}...")
                
                if len(relevant_memories) >= max_memories:
                    break
            
            logger.info(f"Retrieved {len(relevant_memories)} relevant memories")
            return relevant_memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    def get_recent_memories(self, user_id: str = "default_user", limit: int = 10) -> List[Dict]:
        """
        Get recent memories for a user
        
        Args:
            user_id: Identifier for the user
            limit: Maximum number of memories to return
            
        Returns:
            List of recent memory dictionaries
        """
        try:
            # Get all memories for user
            results = self.collection.get(
                where={"user_id": user_id},
                include=["documents", "metadatas"]
            )
            
            if not results["documents"]:
                return []
            
            # Combine documents and metadata
            memories = []
            for doc, metadata in zip(results["documents"], results["metadatas"]):
                memory = {
                    "content": doc,
                    "timestamp": metadata.get("timestamp"),
                    "context": metadata.get("context"),
                    "speaker": metadata.get("speaker")
                }
                memories.append(memory)
            
            # Sort by timestamp (most recent first)
            memories.sort(key=lambda x: x["timestamp"] or "", reverse=True)
            
            return memories[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get recent memories: {e}")
            return []

    def clear_user_memories(self, user_id: str) -> int:
        """
        Clear all memories for a specific user
        
        Args:
            user_id: Identifier for the user
            
        Returns:
            Number of memories deleted
        """
        try:
            # Get all memory IDs for user
            results = self.collection.get(
                where={"user_id": user_id},
                include=["documents"]
            )
            
            if not results["ids"]:
                return 0
            
            # Delete memories
            self.collection.delete(
                where={"user_id": user_id}
            )
            
            deleted_count = len(results["ids"])
            logger.info(f"Deleted {deleted_count} memories for user {user_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return 0

    def get_memory_stats(self) -> Dict:
        """
        Get statistics about stored memories
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            total_memories = self.collection.count()
            
            # Get all metadata to analyze
            if total_memories > 0:
                results = self.collection.get(include=["metadatas"])
                metadatas = results["metadatas"]
                
                # Count by context type
                context_counts = {}
                user_counts = {}
                
                for metadata in metadatas:
                    context = metadata.get("context", "unknown")
                    user_id = metadata.get("user_id", "unknown")
                    
                    context_counts[context] = context_counts.get(context, 0) + 1
                    user_counts[user_id] = user_counts.get(user_id, 0) + 1
                
                return {
                    "total_memories": total_memories,
                    "memories_by_context": context_counts,
                    "memories_by_user": user_counts,
                    "collection_name": self.collection_name
                }
            else:
                return {
                    "total_memories": 0,
                    "memories_by_context": {},
                    "memories_by_user": {},
                    "collection_name": self.collection_name
                }
                
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}


# Example usage and testing
if __name__ == "__main__":
    try:
        # Initialize memory manager
        memory_manager = MemoryManager()
        
        # Test storing a conversation
        user_msg = "Hi Mai, my name is Alex and I'm a software developer from San Francisco."
        ai_response = "Hello Alex! It's wonderful to meet you. I'd love to learn more about your work as a software developer in San Francisco."
        
        stored_count = memory_manager.store_conversation(user_msg, ai_response, "alex_user")
        print(f"✅ Stored {stored_count} memories")
        
        # Test retrieving memories
        query = "What do you know about my job?"
        memories = memory_manager.retrieve_memories(query, "alex_user")
        print(f"✅ Retrieved {len(memories)} relevant memories:")
        for memory in memories:
            print(f"  - {memory}")
        
        # Get memory stats
        stats = memory_manager.get_memory_stats()
        print(f"✅ Memory stats: {stats}")
        
    except Exception as e:
        print(f"❌ Error testing memory manager: {e}")
        print("Make sure to install required packages: pip install sentence-transformers chromadb")
