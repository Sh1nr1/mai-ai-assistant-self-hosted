"""
Mai AI Assistant - Sentiment Analysis Module
===========================================

Lightweight emotion classification module for real-time sentiment analysis.
Integrates with Mai's async FastAPI architecture and voice interface.

Author: Elite Senior AI Engineer
Compatible with: Mai v1.0+ (FastAPI + async)
"""

import asyncio
import logging
from functools import lru_cache
from typing import Tuple, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import numpy as np
from pathlib import Path
import json

# Configure logging
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Async sentiment analysis engine using GoEmotions DistilBERT model.
    
    Features:
    - Real-time emotion classification (6 primary emotions + neutral)
    - Async-compatible inference
    - Model caching and fallback handling
    - Confidence scoring with thresholds
    - Easy upgradability for multi-label classification
    """
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.is_initialized = False
        
        # Emotion mapping (expandable for future multi-label support)
        self.emotion_map = {
            'LABEL_0': 'sadness',
            'LABEL_1': 'joy', 
            'LABEL_2': 'love',
            'LABEL_3': 'anger',
            'LABEL_4': 'fear',
            'LABEL_5': 'surprise',
            'sadness': 'sadness',
            'joy': 'joy',
            'love': 'joy',  # Map love to joy for primary emotions
            'anger': 'anger',
            'fear': 'fear', 
            'surprise': 'surprise'
        }
        
        # Primary emotions for Mai's core functionality
        self.primary_emotions = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
        
        # Confidence thresholds
        self.confidence_threshold = 0.3
        self.high_confidence_threshold = 0.7

        # Store max sequence length for truncation (common for DistilRoBERTa is 512)
        # We will retrieve this from the tokenizer once loaded
        self.max_model_input_length = None 

    async def initialize(self) -> bool:
        """
        Initialize the sentiment analysis model asynchronously.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if self.is_initialized:
            return True
            
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            
            # Load model and tokenizer in thread pool to avoid blocking
            model_data = await asyncio.to_thread(self._load_model)
            
            if model_data:
                self.model, self.tokenizer, self.classifier = model_data
                # Get the max_model_input_length from the tokenizer
                self.max_model_input_length = self.tokenizer.model_max_length
                logger.info(f"Model max input length: {self.max_model_input_length}")

                self.is_initialized = True
                logger.info("Sentiment analysis model loaded successfully")
                return True
            else:
                logger.error("Failed to load sentiment model")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {str(e)}")
            return False

    def _load_model(self) -> Optional[Tuple[Any, Any, Any]]:
        """
        Load the sentiment analysis model (runs in thread pool).
        
        Returns:
            Tuple of (model, tokenizer, classifier) or None if failed
        """
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline for easier inference
            # --- IMPORTANT: Add truncation=True here ---
            classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True,
                truncation=True # <--- THIS IS THE KEY ADDITION
            )
            
            return model, tokenizer, classifier
            
        except Exception as e:
            logger.error(f"Failed to load model components: {str(e)}")
            return None

    async def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment and emotion from input text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Tuple[str, float]: (primary_emotion, confidence_score)
            
        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> await analyzer.initialize()
            >>> emotion, confidence = await analyzer.analyze_sentiment("I'm so happy today!")
            >>> print(f"Emotion: {emotion}, Confidence: {confidence:.2f}")
            # Output: Emotion: joy, Confidence: 0.89
        """
        # Input validation
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text for sentiment analysis")
            return "neutral", 0.0
            
        text = text.strip()
        if len(text) == 0:
            return "neutral", 0.0
            
        # Initialize if needed
        if not self.is_initialized:
            success = await self.initialize()
            if not success:
                return await self._fallback_analysis(text)
        
        try:
            # Run inference in thread pool to maintain async compatibility
            # The truncation will happen automatically within _classify_emotion via the pipeline
            result = await asyncio.to_thread(self._classify_emotion, text)
            return result
            
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {str(e)}")
            return await self._fallback_analysis(text)

    def _classify_emotion(self, text: str) -> Tuple[str, float]:
        """
        Classify emotion using the loaded model (runs in thread pool).
        The pipeline configured with `truncation=True` handles text longer than max_model_input_length.
        
        Args:
            text (str): Text to classify
            
        Returns:
            Tuple[str, float]: (emotion, confidence)
        """
        try:
            # Get predictions from model
            # The pipeline will automatically truncate the input 'text' if it exceeds
            # the model's maximum sequence length (e.g., 512 tokens).
            predictions = self.classifier(text) 
            
            # Process results - predictions is a list of dicts with labels and scores
            if isinstance(predictions, list) and len(predictions) > 0:
                # The pipeline with return_all_scores=True gives a list containing a list of dicts: [[{}, {}, ...]]
                # or directly a list of dicts if single input: [{}, {}, ...]
                # Handle both cases by getting the first element if it's a list.
                prediction_scores = predictions[0] if isinstance(predictions[0], list) else predictions
                
                # Find highest confidence prediction
                best_pred = max(prediction_scores, key=lambda x: x['score'])
                raw_emotion = best_pred['label'].lower()
                confidence = float(best_pred['score'])
                
                # Map to primary emotion
                primary_emotion = self._map_to_primary_emotion(raw_emotion)
                
                # Apply confidence threshold
                if confidence < self.confidence_threshold:
                    primary_emotion = "neutral"
                    confidence = max(confidence, 0.1)  # Minimum confidence for neutral
                
                logger.debug(f"Classified '{text[:50]}...' as {primary_emotion} (confidence: {confidence:.3f})")
                return primary_emotion, confidence
            
            else:
                logger.warning("Unexpected prediction format from model")
                return "neutral", 0.1
                
        except Exception as e:
            logger.error(f"Error in emotion classification: {str(e)}")
            # Consider adding the full traceback here for better debugging
            # logger.exception(f"Error in emotion classification: {e}")
            return "neutral", 0.1

    def _map_to_primary_emotion(self, raw_emotion: str) -> str:
        """
        Map model output to primary emotion categories.
        
        Args:
            raw_emotion (str): Raw emotion from model
            
        Returns:
            str: Primary emotion category
        """
        # Direct mapping first
        if raw_emotion in self.emotion_map:
            mapped = self.emotion_map[raw_emotion]
            if mapped in self.primary_emotions:
                return mapped
        
        # Fuzzy matching for robustness
        emotion_keywords = {
            'joy': ['joy', 'happy', 'excited', 'love', 'optimism'],
            'sadness': ['sadness', 'sad', 'grief', 'disappointment'],
            'anger': ['anger', 'angry', 'rage', 'annoyance', 'irritation'],
            'fear': ['fear', 'afraid', 'anxiety', 'nervousness', 'worry'],
            'surprise': ['surprise', 'surprised', 'amazement'],
        }
        
        raw_lower = raw_emotion.lower()
        for primary, keywords in emotion_keywords.items():
            if any(keyword in raw_lower for keyword in keywords):
                return primary
        
        return "neutral"

    async def _fallback_analysis(self, text: str) -> Tuple[str, float]:
        """
        Fallback sentiment analysis using simple keyword matching.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Tuple[str, float]: (emotion, confidence)
        """
        logger.info("Using fallback sentiment analysis")
        
        # Simple keyword-based classification
        text_lower = text.lower()
        
        # Emotion keyword dictionaries
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'great', 'awesome', 'love', 'wonderful', 'amazing', 'fantastic'],
            'sadness': ['sad', 'crying', 'depressed', 'upset', 'disappointed', 'hurt', 'heartbroken'],
            'anger': ['angry', 'mad', 'furious', 'hate', 'annoyed', 'frustrated', 'irritated'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'frightened', 'terrified'],
            'surprise': ['surprised', 'shocked', 'amazed', 'wow', 'incredible', 'unbelievable']
        }
        
        # Score each emotion
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                emotion_scores[emotion] = score / len(keywords)  # Normalize
        
        if emotion_scores:
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = min(emotion_scores[best_emotion] * 0.6, 0.8)  # Cap fallback confidence
            return best_emotion, confidence
        
        return "neutral", 0.2

    async def get_emotion_vector(self, text: str) -> Dict[str, float]:
        """
        Get emotion scores for all primary emotions (future upgrade feature).
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Emotion scores for all categories
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # This is a placeholder for future multi-label functionality
            primary_emotion, confidence = await self.analyze_sentiment(text)
            
            # Create emotion vector with primary emotion having highest score
            emotion_vector = {emotion: 0.1 for emotion in self.primary_emotions}
            emotion_vector[primary_emotion] = confidence
            
            return emotion_vector
            
        except Exception as e:
            logger.error(f"Error getting emotion vector: {str(e)}")
            return {emotion: 0.1 for emotion in self.primary_emotions}

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'model_name': self.model_name,
            'is_initialized': self.is_initialized,
            'primary_emotions': self.primary_emotions,
            'confidence_threshold': self.confidence_threshold,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'max_input_length': self.max_model_input_length
        }


# Global analyzer instance (singleton pattern)
_analyzer_instance: Optional[SentimentAnalyzer] = None

@lru_cache(maxsize=1)
def get_analyzer() -> SentimentAnalyzer:
    """
    Get singleton sentiment analyzer instance.
    
    Returns:
        SentimentAnalyzer: Cached analyzer instance
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SentimentAnalyzer()
    return _analyzer_instance

async def analyze_sentiment(text: str) -> Tuple[str, float]:
    """
    Main async function for sentiment analysis (Mai integration point).
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Tuple[str, float]: (primary_emotion, confidence_score)
        
    Example Usage:
        # In voice_interface.py or app.py:
        from sentiment_analysis import analyze_sentiment
        
        async def process_voice_input(audio_data):
            transcribed_text = await transcribe_audio(audio_data)
            emotion, confidence = await analyze_sentiment(transcribed_text)
            
            # Use emotion data for response generation or memory storage
            print(f"User emotion: {emotion} (confidence: {confidence:.2f})")
            
            # Continue with LLM and memory processing...
            response = await llm_handler.generate_response(transcribed_text, emotion_context=emotion)
            await memory_manager.store_conversation(transcribed_text, response, emotion=emotion)
    """
    analyzer = get_analyzer()
    return await analyzer.analyze_sentiment(text)

async def initialize_sentiment_analysis() -> bool:
    """
    Initialize the sentiment analysis system (call from app.py startup).
    
    Returns:
        bool: True if successful, False otherwise
    """
    analyzer = get_analyzer()
    return await analyzer.initialize()

# For testing and development
if __name__ == "__main__":
    async def test_sentiment_analysis():
        """Test the sentiment analysis functionality."""
        print("Testing Mai Sentiment Analysis Module")
        print("=" * 40)
        
        # Configure logging for local testing
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Initialize analyzer
        success = await initialize_sentiment_analysis()
        if not success:
            print("Failed to initialize sentiment analyzer")
            return
        
        # Test cases
        test_texts = [
            "I'm so happy and excited about this project!",
            "I'm feeling really sad and down today.",
            "This makes me so angry and frustrated!",
            "I'm scared and worried about the future.",
            "Wow, that's really surprising and amazing!",
            "Hello, how are you doing today?",
            "",
            "The weather is nice.",
            # Add a long text to test truncation
            "This is a very long piece of text that definitely exceeds the typical 512 token limit for models like DistilRoBERTa. We are writing this to test if the truncation mechanism in the Hugging Face pipeline works correctly. If it does, we should not see the 'Token indices sequence length is longer' error anymore. The model should simply process the first 512 tokens or so and provide a prediction based on that. This is crucial for maintaining stability and preventing crashes when users input very verbose messages. Hopefully, the pipeline handles this gracefully, allowing the rest of our application to function without interruption. We need to ensure that the core functionality remains robust, even under extreme input conditions. This long sentence continues to emphasize the importance of text handling. It just keeps going and going, testing the limits of what the model can handle without explicitly truncating the string beforehand. The `truncation=True` parameter in the pipeline is designed precisely for scenarios like this, where you want the model to process the input up to its maximum capacity and disregard anything beyond that. This approach simplifies the pre-processing step significantly for the developer. It's a convenient feature that prevents common errors related to sequence length. So, if everything is set up correctly, this sentence should be smoothly handled, even if only a portion of it is actually used for the prediction. The goal is resilience."
        ]
        
        for text in test_texts:
            emotion, confidence = await analyze_sentiment(text)
            print(f"\nText: '{text[:100]}...'") # Print a snippet for long texts
            print(f"Emotion: {emotion}, Confidence: {confidence:.3f}")
            print("-" * 30)
    
    # Run test
    import sys # Need to import sys for logging config in __main__
    asyncio.run(test_sentiment_analysis())