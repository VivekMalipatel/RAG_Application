import asyncio
import whisper

class AudioChunker:
    """Splits transcribed speech into sentences."""

    def __init__(self, min_length=3, max_length=10):
        self.min_length = min_length
        self.max_length = max_length

    async def chunk_audio_transcription(self, transcript: str):
        """Asynchronously splits a long transcript into smaller chunks."""
        return await asyncio.to_thread(self._split_sentences, transcript)

    def _split_sentences(self, transcript: str):
        """Splits transcript into short, context-aware segments."""
        sentences = transcript.split(". ")
        chunks = []

        temp_chunk = []
        for sentence in sentences:
            temp_chunk.append(sentence)
            if len(temp_chunk) >= self.min_length:
                chunks.append(" ".join(temp_chunk))
                temp_chunk = []

        if temp_chunk:
            chunks.append(" ".join(temp_chunk))

        return chunks