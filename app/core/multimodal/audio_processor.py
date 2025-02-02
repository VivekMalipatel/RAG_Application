import whisper
import os

class AudioProcessor:
    """Transcribes audio files using Whisper."""

    def __init__(self, model_size="base"):
        """Ensure FFmpeg is correctly installed and available."""
        os.environ["PATH"] += os.pathsep + "/usr/bin"
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str):
        """Transcribes speech from an audio file."""
        try:
            result = self.model.transcribe(audio_path)
            return result["text"]
        except Exception as e:
            print(f"🔹 Audio Transcription Error: {e}")
            return ""