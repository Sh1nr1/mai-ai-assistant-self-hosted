"""
Voice Interface for Mai - Emotionally Intelligent AI Assistant
Handles voice input/output for natural conversation with Mai
"""

import os
import io
import wave
import time
import asyncio
import threading
import logging
from typing import Any
from typing import Optional, Dict, List, Tuple, Callable
from pathlib import Path
import uuid # Import uuid for generating unique filenames

# Audio processing imports
try:
    import pyaudio
    import numpy as np
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    logging.warning("PyAudio not available. Voice input from live microphone will be disabled.")

# Speech recognition imports
try:
    import speech_recognition as sr
    HAS_SPEECH_RECOGNITION = True
except ImportError:
    HAS_SPEECH_RECOGNITION = False
    logging.warning("SpeechRecognition not available. Using placeholder for voice input.")

try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    logging.warning("Whisper not available. Falling back to Google Speech Recognition for live mic or disabling file transcription.")

# Text-to-speech imports
try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False
    logging.warning("Edge TTS not available. Voice output will be disabled.")

# Audio playback
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    try:
        import playsound
        HAS_PLAYSOUND = True
        HAS_PYGAME = False
    except ImportError:
        HAS_PLAYSOUND = False
        HAS_PYGAME = False
        logging.warning("No audio playback library available (pygame/playsound).")

from llm_handler import LLMHandler
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import httpx # edge_tts uses httpx internally for some things, or aiohttp


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class VoiceInterface:
    """Handles voice input and output for Mai's conversational AI"""
    
    def __init__(self, llm_handler: LLMHandler, voice_name: str = "en-IN-NeerjaNeural"):
        """
        Initialize voice interface.
        
        Args:
            llm_handler: Instance of LLMHandler for generating responses
            voice_name: Edge TTS voice to use for Mai's responses
        """
        self.llm_handler = llm_handler
        self.voice_name = voice_name
        self.audio_output_dir = Path("audio_output") # This path is used by app.py
        self.audio_output_dir.mkdir(exist_ok=True)
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16 if HAS_PYAUDIO else None
        self.channels = 1
        
        # Voice activity detection settings
        self.silence_threshold = 500  # Adjust based on environment
        self.silence_duration = 2.0   # Seconds of silence before stopping
        
        # Initialize speech recognition for live mic input
        if HAS_SPEECH_RECOGNITION:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self._calibrate_microphone()
        
        # Initialize Whisper model (load on first use)
        self.whisper_model = None
        global HAS_PYGAME
        # Initialize pygame for audio playback
        if HAS_PYGAME:
            try:
                pygame.mixer.init()
                logger.info("Pygame mixer initialized successfully in VoiceInterface.")
            except Exception as e:
                logger.error(f"CRITICAL: Failed to initialize Pygame mixer in VoiceInterface: {e}")
                HAS_PYGAME = False # Ensure we don't try to use it if initialization fails
        
        # State management
        self.is_listening = False
        self.is_speaking = False
        self.chat_history = []
        self.memory_context = []
        
        logger.info(f"VoiceInterface initialized with voice: {self.voice_name}")
    
    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise"""
        if not HAS_SPEECH_RECOGNITION:
            return
            
        try:
            with self.microphone as source:
                logger.info("Calibrating microphone for ambient noise... Please be quiet.")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info("Microphone calibration complete.")
        except Exception as e:
            logger.error(f"Failed to calibrate microphone: {e}")
    
    def _load_whisper_model(self):
        """Loads the Whisper model if not already loaded."""
        if self.whisper_model is None:
            if not HAS_WHISPER:
                raise ImportError("Whisper library is not installed. Cannot load model.")
            logger.info("Loading Whisper model (this may take a moment)...")
            # Using 'base' model for general purpose. Consider 'base.en' for English-only.
            self.whisper_model = whisper.load_model("base") 
            logger.info("Whisper model loaded.")

    async def transcribe_audio_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Transcribes an audio file from a given path using Whisper model.
        This method is suitable for audio blobs received from frontend.

        Args:
            file_path: Path to the audio file (e.g., .webm, .wav, .mp3).

        Returns:
            Dict[str, Any]: A dictionary containing:
                'success' (bool): True if transcription was successful, False otherwise.
                'transcribed_text' (str): The transcribed text if successful, else an empty string.
                'error' (str): Error message if transcription failed.
        """
        if not HAS_WHISPER:
            error_msg = "Whisper is not available. Cannot transcribe audio file."
            logger.error(error_msg)
            return {'success': False, 'transcribed_text': '', 'language': 'en', 'error': error_msg}
        
        if not file_path.exists():
            error_msg = f"Audio file not found: {file_path}"
            logger.error(error_msg)
            return {'success': False, 'transcribed_text': '', 'language': 'en', 'error': error_msg}

        try:
            self._load_whisper_model() # Ensure model is loaded

            logger.info(f"Attempting to transcribe audio file: {file_path}")
            result = self.whisper_model.transcribe(str(file_path))
            text = result["text"].strip()
            raw_detected_language = result.get("language")
            
            # Define the allowed languages
            allowed_languages = ['en', 'hi']
            
            # Determine the effective language based on allowed languages
            effective_language = 'en' # Default to English
            if raw_detected_language in allowed_languages:
                effective_language = raw_detected_language
            else:
                logger.warning(f"Whisper detected language '{raw_detected_language}', which is not 'en' or 'hi'. Defaulting to 'en'.")
            
            if text:
                logger.info(f"Whisper file transcription: '{text}' (Effective Language: {effective_language})")
                return {
                    'success': True, 
                    'transcribed_text': text, 
                    'language': effective_language, # Return the filtered/effective language
                    'error': None
                }
            else:
                logger.info("Whisper transcribed audio file, but no text was extracted. Returning effective language.")
                return {
                    'success': False, 
                    'transcribed_text': '', 
                    'language': effective_language, # Still return effective language even if no text
                    'error': 'No text extracted from audio'
                }
            
        except Exception as e:
            error_msg = f"Failed to transcribe audio file {file_path} with Whisper: {e}"
            logger.error(error_msg, exc_info=True) # Log full traceback for debugging
            # Default to 'en' in case of an error during transcription
            return {'success': False, 'transcribed_text': '', 'language': 'en', 'error': error_msg}
            
    async def listen_for_speech(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Listen for speech input from the microphone and convert to text.
        
        Args:
            timeout: Maximum time to wait for speech (None for no timeout)
            
        Returns:
            Transcribed text or None if no speech detected/error occurred
        """
        if not HAS_SPEECH_RECOGNITION:
            return self._placeholder_speech_input()
        
        try:
            self.is_listening = True
            logger.info("Listening for speech from microphone...")
            
            with self.microphone as source:
                # Listen for audio with timeout
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=10  # Max 10 seconds per phrase
                )
            
            self.is_listening = False
            logger.info("Audio captured from microphone, processing...")
            
            # Try Whisper first if available
            if HAS_WHISPER:
                return await self._transcribe_with_whisper_from_sr_audio(audio)
            else:
                return await self._transcribe_with_google(audio)
                
        except sr.WaitTimeoutError:
            logger.info("No speech detected within timeout period.")
            self.is_listening = False
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition request failed: {e}")
            self.is_listening = False
            return None
        except Exception as e:
            logger.error(f"Error during live speech recognition: {e}")
            self.is_listening = False
            return None
    
    async def _transcribe_with_whisper_from_sr_audio(self, audio) -> Optional[str]:
        """Transcribe audio (from speech_recognition.AudioData) using Whisper model"""
        try:
            self._load_whisper_model() # Ensure model is loaded
            
            # Convert audio to numpy array (Whisper expects float32)
            audio_data = np.frombuffer(audio.get_wav_data(), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Transcribe
            result = self.whisper_model.transcribe(audio_data)
            text = result["text"].strip()
            
            if text:
                logger.info(f"Whisper live transcription: '{text}'")
                return text
            return None
            
        except Exception as e:
            logger.error(f"Whisper live transcription failed: {e}")
            # Fallback to Google if Whisper fails for live mic input
            return await self._transcribe_with_google(audio)
    
    async def _transcribe_with_google(self, audio) -> Optional[str]:
        """Transcribe audio using Google Speech Recognition (for live mic)"""
        try:
            text = self.recognizer.recognize_google(audio)
            if text:
                logger.info(f"Google transcription: '{text}'")
                return text
            return None
        except sr.UnknownValueError:
            logger.info("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition error: {e}")
            return None
    
    def _placeholder_speech_input(self) -> str:
        """Placeholder for speech input when libraries aren't available"""
        print("\n🎤 Voice input not available. Please type your message:")
        return input("You: ").strip()
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5),
           retry=retry_if_exception_type(httpx.ConnectError) | retry_if_exception_type(httpx.ReadError) | \
                 retry_if_exception_type(asyncio.TimeoutError) | retry_if_exception_type(ConnectionResetError))
    async def _save_speech_with_retry(self, communicate: edge_tts.Communicate, output_path: Path):
        """Internal helper to save speech with retry logic."""
        await communicate.save(str(output_path))

    async def generate_speech(self, text: str, filename: str = "response.mp3") -> str:
        if not HAS_EDGE_TTS:
            logger.warning("Edge TTS not available. Cannot generate speech.")
            return ""

        try:
            output_path = self.audio_output_dir / filename
            communicate = edge_tts.Communicate(text, self.voice_name)

            # Use the retry helper
            await self._save_speech_with_retry(communicate, output_path)

            logger.info(f"Speech generated: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to generate speech after retries: {e}")
            return ""
    
    def play_audio(self, audio_path: str) -> bool:
        logger.info(f"DEBUG: Attempting to play audio file from path: {audio_path}")
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return False
        
        try:
            self.is_speaking = True
            
            if HAS_PYGAME: # This HAS_PYGAME refers to the global variable
                pygame.mixer.music.load(audio_path)
                logger.info(f"DEBUG: Loaded {audio_path} into Pygame mixer.")
                pygame.mixer.music.play()
                logger.info("DEBUG: Pygame play() called. Checking busy status...")
                
                # Wait for playback to complete
                # Add a counter to confirm the loop is entered
                busy_checks = 0
                while pygame.mixer.music.get_busy():
                    busy_checks += 1
                    time.sleep(0.1)
                    if busy_checks % 10 == 0: # Log every second if sleep is 0.1s
                        logger.info(f"DEBUG: Pygame is busy... (checked {busy_checks} times)")
                
                if busy_checks == 0:
                    logger.warning("DEBUG: Pygame was not busy after calling play(). This might indicate an issue with playback not starting.")
                else:
                    logger.info(f"DEBUG: Pygame finished playing after {busy_checks} checks.")

            elif HAS_PLAYSOUND:
                playsound.playsound(audio_path)
                logger.info("Playsound playback completed.")
            else:
                logger.warning("No audio playback library available (pygame/playsound).")
                return False
            
            self.is_speaking = False
            logger.info("Audio playback completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to play audio: {e}", exc_info=True) # Use exc_info=True for full traceback
            self.is_speaking = False
            return False
    
    async def process_voice_interaction(self, user_input: Optional[str] = None) -> Dict:
        """
        Process a complete voice interaction cycle (transcribe, LLM, TTS).
        
        Args:
            user_input: Pre-provided user input (e.g., from an audio file transcription
                        or direct text input from frontend)
            
        Returns:
            Dictionary with interaction results
        """
        try:
            # Step 1: Get user input (from pre-provided or live mic)
            if user_input is None:
                # This branch is for local microphone input
                user_text = await self.listen_for_speech(timeout=10)
                if not user_text:
                    return {
                        "success": False,
                        "error": "No speech detected",
                        "user_input": "",
                        "mai_response": "",
                        "audio_path": ""
                    }
            else:
                # This branch is for input provided by the calling Flask app (e.g., from blob)
                user_text = user_input
            
            logger.info(f"User input received: '{user_text}'")
            
            # Step 2: Generate Mai's response
            # AWAIT the coroutine here
            response_data = await self.llm_handler.generate_response(
                user_message=user_text,
                chat_history=self.chat_history,
                memory_context=self.memory_context
            )
            
            if not response_data["success"]:
                logger.error(f"LLM response generation failed: {response_data.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "error": response_data["error"],
                    "user_input": user_text,
                    "mai_response": response_data["response"],
                    "audio_path": ""
                }
            
            mai_response = response_data["response"]
            logger.info(f"Mai's response: '{mai_response}'")
            
            # Step 3: Generate speech for Mai's response
            # Use UUID for a truly unique filename to avoid conflicts, especially in web env
            audio_filename = f"mai_response_{uuid.uuid4()}.mp3"
            audio_full_path = await self.generate_speech(mai_response, audio_filename)
            
            # Step 4: Update chat history
            self.chat_history.append({"role": "user", "content": user_text})
            self.chat_history.append({"role": "assistant", "content": mai_response})
            
            # Keep only last 20 messages to manage memory
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]
            
            return {
                "success": True,
                "user_input": user_text,
                "mai_response": mai_response,
                "audio_path": audio_full_path, # Return full path here, app.py will extract filename
                "usage": response_data.get("usage", {}),
                "model": response_data.get("model", "")
            }
            
        except Exception as e:
            logger.error(f"Error in process_voice_interaction: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "user_input": user_input or "",
                "mai_response": "I'm sorry, I encountered an error processing your request.",
                "audio_path": ""
            }
    
    async def voice_chat_loop(self, max_turns: Optional[int] = None, 
                              callback: Optional[Callable] = None) -> None:
        """
        Main voice chat loop for continuous conversation (for local console usage).
        
        Args:
            max_turns: Maximum number of conversation turns (None for unlimited)
            callback: Optional callback function called after each interaction
        """
        logger.info("Starting voice chat loop...")
        print("\n🎙️ Mai is ready to chat! Say something or type 'quit' to exit.")
        
        # Initial greeting
        greeting = "Hi there! I'm Mai, your emotionally intelligent AI assistant. How are you feeling today?"
        print(f"\nMai: {greeting}")
        
        # Generate and play greeting
        greeting_audio = await self.generate_speech(greeting, "greeting.mp3")
        if greeting_audio:
            self.play_audio(greeting_audio)
        
        turn_count = 0
        
        try:
            while True:
                # Check max turns limit
                if max_turns and turn_count >= max_turns:
                    break
                
                print(f"\n--- Turn {turn_count + 1} ---")
                print("🎤 Listening... (speak now or type 'quit')")
                
                # Process voice interaction (will use listen_for_speech directly)
                result = await self.process_voice_interaction(user_input=None)
                
                if result["success"]:
                    print(f"\nYou: {result['user_input']}")
                    print(f"Mai: {result['mai_response']}")
                    
                    # Play Mai's response
                    if result["audio_path"]:
                        print("🔊 Playing Mai's response...")
                        self.play_audio(result["audio_path"])
                    
                    # Call callback if provided
                    if callback:
                        callback(result)
                        
                else:
                    print(f"\nError: {result['error']}")
                    if result["mai_response"]:
                        print(f"Mai: {result['mai_response']}")
                
                # Check for quit condition
                user_input_lower = result.get("user_input", "").lower()
                if any(quit_word in user_input_lower for quit_word in ["quit", "exit", "goodbye", "bye"]):
                    farewell = "It was wonderful talking with you! Take care, and feel free to chat with me anytime."
                    print(f"\nMai: {farewell}")
                    
                    farewell_audio = await self.generate_speech(farewell, "farewell.mp3")
                    if farewell_audio:
                        self.play_audio(farewell_audio)
                    break
                
                turn_count += 1
                
        except KeyboardInterrupt:
            print("\n\nVoice chat interrupted by user.")
        except Exception as e:
            logger.error(f"Error in voice chat loop: {e}")
        
        logger.info("Voice chat loop ended.")
    
    def set_memory_context(self, memories: List[str]) -> None:
        """Set memory context for the conversation"""
        self.memory_context = memories
    
    def get_chat_history(self) -> List[Dict]:
        """Get current chat history"""
        return self.chat_history.copy()
    
    def clear_chat_history(self) -> None:
        """Clear chat history"""
        self.chat_history.clear()
        logger.info("Chat history cleared")
    
    def set_voice(self, voice_name: str) -> None:
        """Change Mai's voice"""
        self.voice_name = voice_name
        logger.info(f"Voice changed to: {voice_name}")
    
    def get_available_voices(self) -> List[str]:
        """Get list of recommended voices for Mai"""
        return [
            "en-IN-NeerjaNeural",      # Soft, expressive Indian English
            "en-US-JennyMultilingualNeural", # Natural, friendly US English
            "en-US-AriaNeural",        # Conversational US English
            "en-GB-SoniaNeural",       # Professional UK English
            "en-AU-NatashaNeural",     # Warm Australian English
            "en-CA-ClaraNeural",       # Friendly Canadian English
        ]

    def get_audio_output_dir(self) -> Path:
        """Returns the Path object for the audio output directory."""
        return self.audio_output_dir

# --- Example usage and testing ---
async def main():
    """Test the voice interface"""
    logger.info("Testing Voice Interface...")
    
    # Initialize LLM handler
    try:
        # NOTE: Make sure LLMHandler is correctly set up for your environment.
        # This might require API keys or local model paths.
        llm_handler = LLMHandler() 
        logger.info("LLM Handler initialized")
    except Exception as e:
        logger.error(f"Failed to initialize LLM Handler: {e}")
        print("Please ensure your LLMHandler is correctly configured (e.g., API keys, model paths).")
        return
    
    # Initialize voice interface
    voice_interface = VoiceInterface(llm_handler)
    
    # Test single interaction with provided text (simulating frontend audio blob)
    print("\n=== Testing Single Voice Interaction (Simulating Frontend Input) ===")
    
    # You would typically create a dummy audio file here for actual testing
    # For now, we'll simulate transcription by passing text directly
    test_text_input = "Hello Mai, can you tell me about the weather today?"
    print(f"Simulating frontend sending: '{test_text_input}'")

    # Call process_voice_interaction with the pre-provided text
    result = await voice_interface.process_voice_interaction(user_input=test_text_input)
    print(f"Result: {result}")
    
    # If the TTS generated an audio file, you can try to play it here (for local testing)
    if result.get("success") and result.get("audio_path"):
        print(f"Attempting to play generated audio: {result['audio_path']}")
        voice_interface.play_audio(result['audio_path'])

    # Test voice chat loop (uncomment to run for live mic interaction)
    # print("\n=== Starting Voice Chat Loop (for live microphone input) ===")
    # await voice_interface.voice_chat_loop(max_turns=2)


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())
