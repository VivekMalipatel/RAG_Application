import whisper
import os
import logging

class AudioProcessor:
    """Transcribes audio files using OpenAI's Whisper model."""

    def __init__(self, model_size="base"):
        """
        Initializes Whisper ASR model.

        Args:
            model_size (str, optional): Size of the Whisper model to load. Defaults to "base".
        """
        try:
            os.environ["PATH"] += os.pathsep + "/usr/bin"
            self.model = whisper.load_model(model_size)
            logging.info(f"Loaded Whisper model: {model_size}")
        except Exception as e:
            logging.error(f"Failed to load Whisper model: {e}")
            self.model = None

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes speech from an audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            str: Transcribed text, or an empty string if an error occurs.
        """
        if not self.model:
            logging.error("Whisper model not loaded. Cannot transcribe.")
            return ""

        try:
            result = self.model.transcribe(audio_path)
            logging.info(f"Successfully transcribed audio: {audio_path}")
            return result.get("text", "").strip()
        except FileNotFoundError:
            logging.error(f"Audio file not found: {audio_path}")
        except Exception as e:
            logging.error(f"Error during transcription for '{audio_path}': {e}")

        return ""