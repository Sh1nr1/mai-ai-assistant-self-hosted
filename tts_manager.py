# tts_manager.py
# Contains the logic for Text-to-Speech (TTS) services, with corrected Edge TTS implementation.

import asyncio
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict
import html

# Third-party imports
try:
    import edge_tts
    import httpx
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

logger = logging.getLogger(__name__)

class TTSManager(ABC):
    """Abstract Base Class for Text-to-Speech managers."""
    @abstractmethod
    async def generate_speech(self, text: str, output_path: Path, emotion: Optional[str] = None) -> str:
        """
        Generates speech from text, optionally with emotion.
        """
        pass

    @abstractmethod
    def get_available_voices(self) -> List[str]:
        """Returns a list of voice names available for this TTS service."""
        pass
    
    @abstractmethod
    def get_supported_emotions(self) -> List[str]:
        """Returns a list of emotions that the TTS engine can express."""
        pass

    @abstractmethod
    def set_voice(self, voice_name: str) -> None:
        """Sets the active voice to be used for speech synthesis."""
        pass

class EdgeTTSManager(TTSManager):
    """A concrete implementation of TTSManager using Microsoft Edge's service with expressive styles."""

    def __init__(self, voice_name: str = "en-US-AnaNeural"):
        if not HAS_EDGE_TTS:
            raise ImportError("The 'edge-tts' library is not installed. Please run: pip install edge-tts")
        self.voice_name = voice_name
        
        # Note: Custom SSML and expressive styles are no longer supported
        # due to Microsoft's restrictions on Edge TTS service
        # However, we can still adjust rate, volume, and pitch for different emotions
        
        self.emotion_adjustments: Dict[str, Dict[str, str]] = {
            'LABEL_0': {'rate': '-15%', 'volume': '-10%', 'pitch': '-10Hz'},  # sadness
            'LABEL_1': {'rate': '+10%', 'volume': '+3%',  'pitch': '+3Hz'},   # joy
            'LABEL_2': {'rate': '+5%',  'volume': '+5%',  'pitch': '+2Hz'},   # love
            'LABEL_3': {'rate': '+20%', 'volume': '+10%', 'pitch': '+7Hz'},   # anger
            'LABEL_4': {'rate': '-25%', 'volume': '-5%',  'pitch': '-5Hz'},   # fear (slowed down, quiet, less pitch)
            'LABEL_5': {'rate': '+25%', 'volume': '+12%', 'pitch': '+8Hz'},   # surprise

            'sadness': {'rate': '-15%', 'volume': '-10%', 'pitch': '-10Hz'},
            'joy':     {'rate': '+10%', 'volume': '+3%',  'pitch': '+3Hz'},
            'love':    {'rate': '+5%',  'volume': '+5%',  'pitch': '+2Hz'},
            'anger':   {'rate': '+20%', 'volume': '+10%', 'pitch': '+7Hz'},
            'fear':    {'rate': '-25%', 'volume': '-5%',  'pitch': '-5Hz'},
            'surprise':{'rate': '+25%', 'volume': '+12%', 'pitch': '+8Hz'}
        }
        
        logger.info(f"EdgeTTSManager initialized with voice: {self.voice_name}")
        logger.info("Emotional adjustments available via rate/volume/pitch modifications")
        logger.info(f"Supported emotions: {list(set(self.emotion_adjustments.keys()))}")

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        retry=(
            retry_if_exception_type(httpx.ConnectError) |
            retry_if_exception_type(httpx.ReadError) |
            retry_if_exception_type(asyncio.TimeoutError) |
            retry_if_exception_type(ConnectionResetError)
        ),
        before_sleep=lambda retry_state: logger.warning(f"TTS generation failed, retrying in {retry_state.next_action.sleep}s...")
    )
    async def _save_speech_with_retry(self, communicate: 'edge_tts.Communicate', output_path: Path):
        """Internal helper to save speech with robust retry logic."""
        await communicate.save(str(output_path))

    async def generate_speech(self, text: str, output_path: Path, emotion: Optional[str] = None) -> str:
        """
        Generates speech using Edge TTS with rate, volume, and pitch adjustments for emotions.
        Uses Microsoft-allowed prosody adjustments instead of custom SSML.
        """
        try:
            # Get emotion-based adjustments
            adjustments = {}
            if emotion:
                safe_emotion = emotion.lower().strip()
                if safe_emotion in self.emotion_adjustments:
                    adjustments = self.emotion_adjustments[safe_emotion]
                    logger.info(f"Applying emotion '{emotion}' with adjustments: {adjustments}")
                else:
                    logger.warning(f"Emotion '{emotion}' not recognized. Using default voice settings.")
            
            # Create Edge TTS communicate object with adjustments
            if adjustments:
                communicate = edge_tts.Communicate(
                    text, 
                    self.voice_name,
                    rate=adjustments.get('rate', '+0%'),
                    volume=adjustments.get('volume', '+0%'),
                    pitch=adjustments.get('pitch', '+0Hz')
                )
            else:
                # Use standard Edge TTS without adjustments
                communicate = edge_tts.Communicate(text, self.voice_name)

            # Generate and save the audio
            await self._save_speech_with_retry(communicate, output_path)
            logger.info(f"Speech generated successfully with EdgeTTS: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to generate speech with EdgeTTS after all retries: {e}", exc_info=True)
            return ""

    def set_voice(self, voice_name: str) -> None:
        """Sets the active Edge TTS voice."""
        self.voice_name = voice_name
        logger.info(f"EdgeTTS voice changed to: {voice_name}")
    
    def get_supported_emotions(self) -> List[str]:
        """Returns list of emotions that can be expressed through prosody adjustments."""
        return list(set(self.emotion_adjustments.keys()))

    def get_available_voices(self) -> List[str]:
        """Returns a list of recommended Edge TTS voices."""
        return [
            "en-US-AriaNeural", "en-US-AnaNeural", "en-US-JennyNeural",
            "en-US-GuyNeural", "en-US-ChristopherNeural", "en-US-ElizabethNeural",
            "en-GB-SoniaNeural", "en-GB-MaisieNeural", "ja-JP-NanamiNeural",
            "en-IN-NeerjaNeural", "en-AU-NatashaNeural", "en-CA-ClaraNeural",
        ]

class OpenVoiceV2Manager(TTSManager):
    """(Future Placeholder) A concrete implementation for OpenVoice V2."""
    def __init__(self, model_path: str = "path/to/openvoice_v2"):
        logger.info(f"OpenVoiceV2Manager initialized (placeholder).")
        self.voice_name = "default" 

    async def generate_speech(self, text: str, output_path: Path, emotion: Optional[str] = None) -> str:
        logger.error("OpenVoiceV2Manager `generate_speech` is not implemented yet.")
        raise NotImplementedError(f"OpenVoiceV2 generation is not yet implemented.")

    def set_voice(self, voice_name: str) -> None:
        logger.error("OpenVoiceV2Manager `set_voice` is not implemented yet.")
        self.voice_name = voice_name
    
    def get_supported_emotions(self) -> List[str]:
        logger.warning("OpenVoiceV2Manager `get_supported_emotions` is not implemented yet.")
        return []

    def get_available_voices(self) -> List[str]:
        logger.error("OpenVoiceV2Manager `get_available_voices` is not implemented yet.")
        return ["default_openvoice_v2"]