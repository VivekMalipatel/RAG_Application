import asyncio
import whisper

class AudioProcessor:
    """Converts speech to text asynchronously using OpenAI Whisper."""

    def __init__(self):
        self.model = whisper.load_model("base")

    async def transcribe_audio(self, audio_path: str):
        """Asynchronously transcribes audio files to text."""
        return await asyncio.to_thread(self._transcribe_audio_sync, audio_path)

    def _transcribe_audio_sync(self, audio_path: str):
        """Sync function wrapped for async execution."""
        result = self.model.transcribe(audio_path)
        return result["text"]