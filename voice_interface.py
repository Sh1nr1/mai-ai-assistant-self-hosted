"""
Voice Interface for Mai - Emotionally Intelligent AI Assistant
Handles voice input/output, orchestrating transcription, LLM interaction, and TTS.
Now includes self-sentiment analysis for expressive responses.
"""

import os
import time
import asyncio
import logging
import uuid
from typing import Any, Optional, Dict, List, Callable
from pathlib import Path

# --- Local Application Imports ---
from llm_handler import LLMHandler
# Import the sentiment analyzer for both user input AND Mai's output
from sentiment_analysis import analyze_sentiment, initialize_sentiment_analysis
# The new, modular way to handle Text-to-Speech
from tts_manager import TTSManager, EdgeTTSManager 

# --- Audio Processing Imports ---
try:
    import pyaudio
    import numpy as np
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    logging.warning("PyAudio not available. Voice input from a live microphone will be disabled.")

# --- Speech Recognition Imports ---
try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    HAS_SPEECH_RECOGNITION = False
    logging.warning("SpeechRecognition not available. Using a placeholder for voice input.")

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    logging.warning("Whisper not available. Falling back to Google Speech Recognition or disabling file transcription.")

# --- Translation Imports ---
try:
    from googletrans import Translator
    HAS_GOOGLETRANS = True
except ImportError:
    HAS_GOOGLETRANS = False
    logging.warning("googletrans not available. Hindi translation will be disabled.")

# --- Audio Playback Imports ---
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    try:
        import playsound
        HAS_PLAYSOUND = True
    except ImportError:
        HAS_PLAYSOUND = False
        logging.warning("No audio playback library available (pygame/playsound). Voice output will not be played.")


# --- Global Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class VoiceInterface:
    """Orchestrates the voice conversation flow for Mai."""
    
    def __init__(self, llm_handler: LLMHandler, tts_manager: TTSManager, 
                 enable_hindi_translation: bool = True):
        """
        Initialize the voice interface.

        Args:
            llm_handler: An instance of LLMHandler for generating text responses.
            tts_manager: A configured instance of a class that implements the TTSManager interface
                         (e.g., EdgeTTSManager). This is dependency injection.
            enable_hindi_translation: Whether to enable Hindi to English translation for input.
        """
        self.llm_handler = llm_handler
        self.tts_manager = tts_manager
        self.enable_hindi_translation = enable_hindi_translation
        
        self.audio_output_dir = Path("audio_output")
        self.audio_output_dir.mkdir(exist_ok=True)
        
        self.translator = None
        if self.enable_hindi_translation:
            if HAS_GOOGLETRANS:
                self.translator = Translator()
                logger.info("Hindi to English translation enabled.")
            else:
                logger.warning("Hindi translation requested but 'googletrans' is not available. Install with: pip install googletrans==4.0.0-rc1")
        
        # Audio settings for microphone
        self.sample_rate = 16000
        self.chunk_size = 1024
        
        # Speech recognition setup
        self.recognizer = None
        self.microphone = None
        if HAS_SPEECH_RECOGNITION and HAS_PYAUDIO:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone(sample_rate=self.sample_rate)
            self._calibrate_microphone()
        
        # Whisper model (loaded on first use to save memory)
        self.whisper_model = None
        
        # Pygame mixer initialization for playback
        self.is_pygame_ready = False  # Instance variable to track pygame's initialized state.
        if HAS_PYGAME:
            try:
                pygame.mixer.init()
                self.is_pygame_ready = True
                logger.info("Pygame mixer initialized successfully.")
            except Exception as e:
                # is_pygame_ready remains False
                logger.error(f"CRITICAL: Failed to initialize Pygame mixer: {e}")
        
        # State management
        self.is_listening = False
        self.is_speaking = False
        
        logger.info(f"VoiceInterface initialized with TTS Manager: {type(tts_manager).__name__}")
    
    def _calibrate_microphone(self):
        """Calibrates the microphone for ambient noise to improve recognition."""
        if not self.recognizer or not self.microphone:
            return
        try:
            with self.microphone as source:
                logger.info("Calibrating microphone for ambient noise... Please be quiet.")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                logger.info("Microphone calibration complete.")
        except Exception as e:
            logger.error(f"Could not calibrate microphone: {e}")
    
    def _load_whisper_model(self):
        """Loads the Whisper ASR model if it hasn't been loaded yet."""
        if self.whisper_model is None:
            if not HAS_WHISPER:
                raise ImportError("Whisper library is not installed. Cannot load model.")
            logger.info("Loading Whisper model (this may take a moment)...")
            # Using 'base' model is a good balance. 'tiny' is faster, 'medium' is more accurate.
            self.whisper_model = whisper.load_model("base") 
            logger.info("Whisper model loaded successfully.")

    async def _translate_hindi_to_english(self, text: str) -> Dict[str, Any]:
        """Translates Hindi text to English if translation is enabled."""
        if not self.translator or not self.enable_hindi_translation:
            return {'success': False, 'error': 'Translation not available or disabled'}
        
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.translator.translate, text, 'en', 'hi')
            translated_text = result.text.strip()
            if translated_text:
                logger.info(f"Translation successful: '{text}' -> '{translated_text}'")
                return {'success': True, 'translated_text': translated_text, 'original_text': text}
            else:
                logger.warning(f"Translation returned an empty result for: '{text}'")
                return {'success': False, 'error': 'Translation returned empty result'}
        except Exception as e:
            error_msg = f"Translation failed for '{text}': {e}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}

    async def transcribe_audio_file(self, file_path: Path) -> Dict[str, Any]:
        """Transcribes an audio file using Whisper, with optional Hindi translation."""
        if not HAS_WHISPER:
            return {'success': False, 'error': "Whisper is not available."}
        if not file_path.exists():
            return {'success': False, 'error': f"Audio file not found: {file_path}"}

        try:
            self._load_whisper_model()
            logger.info(f"Transcribing audio file: {file_path}")
            result = self.whisper_model.transcribe(str(file_path))
            text = result["text"].strip()
            lang = result.get("language", "en")
            
            final_text, was_translated, original_text = text, False, text
            if lang == 'hi' and self.enable_hindi_translation:
                translation_result = await self._translate_hindi_to_english(text)
                if translation_result['success']:
                    final_text = translation_result['translated_text']
                    was_translated = True
            
            return {'success': True, 'transcribed_text': final_text, 'original_text': original_text, 'language': lang, 'was_translated': was_translated}
        except Exception as e:
            logger.error(f"Failed to transcribe audio file {file_path}: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
            
    async def listen_for_speech(self, timeout: Optional[float] = 10.0) -> Optional[str]:
        """Listens for speech from the microphone and returns the transcribed text."""
        if not self.recognizer:
            return self._placeholder_speech_input()
        
        try:
            self.is_listening = True
            logger.info("Listening for speech...")
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=15)
            self.is_listening = False
            logger.info("Processing captured audio...")
            
            if HAS_WHISPER:
                return await self._transcribe_with_whisper_from_sr_audio(audio)
            else:
                return await self._transcribe_with_google(audio)
        except sr.WaitTimeoutError:
            logger.info("No speech detected within the timeout period.")
        except Exception as e:
            logger.error(f"An error occurred during live speech recognition: {e}")
        finally:
            self.is_listening = False
        return None
    
    async def _transcribe_with_whisper_from_sr_audio(self, audio: sr.AudioData) -> Optional[str]:
        """Helper to transcribe SpeechRecognition audio data with Whisper."""
        try:
            self._load_whisper_model()
            wav_data = audio.get_wav_data()
            audio_np = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            result = self.whisper_model.transcribe(audio_np, fp16=False) # fp16=False for CPU
            text = result['text'].strip()
            
            if text and result.get('language') == 'hi' and self.enable_hindi_translation:
                translation = await self._translate_hindi_to_english(text)
                return translation.get('translated_text', text)
            return text if text else None
        except Exception as e:
            logger.error(f"Whisper live transcription failed: {e}")
            return None

    async def _transcribe_with_google(self, audio: sr.AudioData) -> Optional[str]:
        """Fallback helper to transcribe with Google's web speech API."""
        try:
            text = self.recognizer.recognize_google(audio)
            return text.strip() if text else None
        except sr.UnknownValueError:
            logger.info("Google Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition service error: {e}")
        return None
    
    def _placeholder_speech_input(self) -> str:
        """Provides a text input prompt when audio hardware is unavailable."""
        print("\n🎤 Voice input not available. Please type your message:")
        return input("You: ").strip()
    
    async def generate_speech(self, text: str, filename: str, emotion: Optional[str] = None) -> str:
        """
        Generates speech by delegating to the TTS manager, now with emotion.
        """
        if not self.tts_manager:
            logger.warning("No TTS manager is configured. Cannot generate speech.")
            return ""

        output_path = self.audio_output_dir / filename
        # Pass the emotion to the TTS manager
        return await self.tts_manager.generate_speech(text, output_path, emotion=emotion)
    
    def play_audio(self, audio_path: str) -> bool:
        """Plays an audio file using pygame or playsound."""
        if not os.path.exists(audio_path):
            logger.error(f"Cannot play audio. File not found: {audio_path}")
            return False
        
        playback_library = "None"
        if HAS_PYGAME: playback_library = "Pygame"
        elif HAS_PLAYSOUND: playback_library = "Playsound"

        logger.info(f"Attempting to play '{audio_path}' using {playback_library}.")
        
        try:
            self.is_speaking = True
            if HAS_PYGAME:
                pygame.mixer.music.load(audio_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            elif HAS_PLAYSOUND:
                import playsound
                playsound.playsound(audio_path)
            else:
                logger.warning("No audio playback library available.")
                return False
            
            logger.info("Audio playback completed.")
            return True
        except Exception as e:
            logger.error(f"Failed to play audio file '{audio_path}': {e}", exc_info=True)
            return False
        finally:
            self.is_speaking = False

    async def process_voice_interaction(self, **kwargs) -> Dict:
        """
        Processes a full voice interaction cycle: Listen -> Think -> Analyze -> Respond.
        """
        user_input = kwargs.get("user_input")
        chat_history = kwargs.get("chat_history", [])
        
        try:
            # === Step 1: Get User Input ===
            if user_input is None:
                user_input = await self.listen_for_speech()
                if not user_input:
                    return {"success": False, "error": "No speech detected", "mai_response": ""}

            logger.info(f"User input received: '{user_input}'")
            
            # --- Optional: Analyze user's emotion for context ---
            user_emotion, user_emotion_confidence = await analyze_sentiment(user_input)
            logger.info(f"Analyzed user emotion: {user_emotion} (Confidence: {user_emotion_confidence:.2f})")
            
            # === Step 2: Generate LLM Response ===
            response_data = await self.llm_handler.generate_response(
                user_message=user_input,
                chat_history=chat_history,
                # Pass user's emotion to the LLM for better contextual responses
                user_emotion=user_emotion,
                emotion_confidence=user_emotion_confidence
            )
            
            if not response_data.get("success"):
                error_msg = response_data.get('error', 'Unknown LLM error')
                logger.error(f"LLM response generation failed: {error_msg}")
                return {"success": False, "error": error_msg, "mai_response": "I'm sorry, I had trouble thinking of a response."}

            mai_response = response_data["response"]
            logger.info(f"Mai's response: '{mai_response}'")
            
            # --- FIX APPLIED HERE ---
            # Analyze Mai's own response to determine the emotion for her voice.
            mai_emotion, mai_emotion_confidence = await analyze_sentiment(mai_response)
            logger.info(f"Analyzed Mai's response emotion for TTS: {mai_emotion} (Confidence: {mai_emotion_confidence:.2f})")

            audio_filename = f"mai_response_{uuid.uuid4()}.mp3"

            # Pass the detected emotion to the speech generation method.
            # Your current code is likely missing the 'emotion=mai_emotion' part of this line.
            audio_full_path = await self.generate_speech(mai_response, audio_filename, emotion=mai_emotion)
            
            return {
                "success": True,
                "user_input": user_input,
                "user_emotion": user_emotion,
                "mai_response": mai_response,
                "mai_emotion": mai_emotion,
                "audio_path": audio_full_path,
                "usage": response_data.get("usage", {}),
            }
        except Exception as e:
            logger.error(f"Critical error in process_voice_interaction: {e}", exc_info=True)
            return {"success": False, "error": str(e), "mai_response": "I'm sorry, a critical error occurred."}
    
    async def voice_chat_loop(self):
        """Starts a continuous voice chat session, for local console testing."""
        logger.info("Starting interactive voice chat loop (for testing)...")
        greeting = "Hi there! I'm Mai. How can I help you today?"
        print(f"\nMai: {greeting}")
        
        # Give the greeting a friendly tone
        audio_path = await self.generate_speech(greeting, "greeting.mp3", emotion="friendly")
        if audio_path: self.play_audio(audio_path)
        
        turn_count = 0
        while True:
            turn_count += 1
            print(f"\n--- Turn {turn_count} ---")
            
            result = await self.process_voice_interaction()
            
            if result.get("success"):
                print(f"You ({result.get('user_emotion', 'N/A')}): {result['user_input']}")
                print(f"Mai ({result.get('mai_emotion', 'N/A')}): {result['mai_response']}")
                if result.get("audio_path"):
                    self.play_audio(result["audio_path"])
                
                if "quit" in (result.get("user_input") or "").lower():
                    break
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                if result.get("mai_response"):
                    print(f"Mai: {result['mai_response']}")
            
        farewell = "Goodbye! It was nice talking to you."
        print(f"Mai: {farewell}")
        farewell_audio = await self.generate_speech(farewell, "farewell.mp3", emotion="friendly")
        if farewell_audio: self.play_audio(farewell_audio)
        logger.info("Voice chat loop ended.")

    # --- Configuration Methods ---
    def set_voice(self, voice_name: str):
        """Changes Mai's voice by calling the configured TTS manager."""
        self.tts_manager.set_voice(voice_name)

    def get_available_voices(self) -> List[str]:
        """Gets the list of available voices from the configured TTS manager."""
        return self.tts_manager.get_available_voices()

    def get_audio_output_dir(self) -> Path:
        """Returns the path to the directory where audio files are saved."""
        return self.audio_output_dir


# --- Example Usage and Testing ---
async def main():
    """Main function to demonstrate and test the refactored VoiceInterface."""
    logger.info("--- Running Voice Interface Test with Self-Sentiment Analysis ---")
    
    try:
        # Step 0: Initialize sentiment analysis model on startup
        await initialize_sentiment_analysis()

        # Step 1: Initialize the components Mai depends on.
        llm_handler = LLMHandler()
        
        # Step 2: Choose and initialize the desired TTS Manager.
        tts_provider = EdgeTTSManager(voice_name="en-US-AriaNeural")
        
        # Step 3: Inject the dependencies into the VoiceInterface.
        voice_interface = VoiceInterface(
            llm_handler=llm_handler,
            tts_manager=tts_provider,
            enable_hindi_translation=False # Disabled for this test run
        )
        
        # --- Run Tests ---
        print("\n=== 1. Testing Expressive Text-to-Speech Generation ===")
        test_text = "This is wonderful news! I'm so happy."
        print(f"Text: {test_text}")
        mai_emotion, _ = await analyze_sentiment(test_text)
        print(f"Detected emotion for TTS: {mai_emotion}")
        audio_path = await voice_interface.generate_speech(test_text, "test_expressive_tts.mp3", emotion=mai_emotion)
        if audio_path:
            print(f"✅ Expressive TTS generation successful: {audio_path}")
            voice_interface.play_audio(audio_path)
        else:
            print("❌ Expressive TTS generation failed.")
            
        print("\n=== 2. Starting Interactive Chat Loop (Optional) ===")
        if HAS_PYAUDIO and HAS_SPEECH_RECOGNITION:
             print("Say 'quit' to end the conversation.")
             await voice_interface.voice_chat_loop()
        else:
            print("Skipping interactive loop because microphone support is not available.")

    except ImportError as e:
        logger.error(f"An essential library is missing: {e}")
        print(f"Please install the required libraries to run the test: {e}")
    except Exception as e:
        logger.error(f"The test failed with a critical error: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting.")
