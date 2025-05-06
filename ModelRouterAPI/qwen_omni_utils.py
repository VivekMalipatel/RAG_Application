import base64
import io
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image

try:
    import decord
    import numpy as np
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

logger = logging.getLogger(__name__)

DEFAULT_QWEN_SPEAKER = "Chelsie"
QWEN_AUDIO_SYSTEM_PROMPT = (
    "You are Qwen, an AI assistant developed by the Qwen Team at Alibaba. "
    "You are capable of multimodal understanding across text, images, and audio. "
    "You can also generate both text and natural-sounding speech responses. "
    "Always be helpful, harmless, and honest in your responses."
)

def process_mm_info(
    messages: List[Dict[str, Any]], 
    use_audio_in_video: bool = True
) -> Tuple[List[Optional[np.ndarray]], List[Image.Image], List[np.ndarray]]:
    """
    Process multimodal information from conversation messages to extract audio, images, and videos.
    
    Args:
        messages: List of conversation messages with multimodal content
        use_audio_in_video: Whether to extract audio from video files
        
    Returns:
        Tuple of (audio_list, image_list, video_list) - Lists of processed media
    """
    audios = []
    images = []
    videos = []
    
    for message in messages:
        if message.get("role") not in ["user", "assistant"]:
            continue
            
        content = message.get("content", [])
        if isinstance(content, str):
            continue
        
        # Handle list of content items (multimodal)
        if isinstance(content, list):
            for item in content:
                item_type = item.get("type")
                
                if item_type == "image":
                    # Handle image data
                    image_data = item.get("image", "")
                    if image_data:
                        try:
                            if isinstance(image_data, str):
                                # Check if it's a URL or base64
                                if image_data.startswith(("http://", "https://")):
                                    # URL handling would go here
                                    logger.warning("URL image handling not implemented")
                                else:
                                    # Assume base64
                                    if image_data.startswith("data:image"):
                                        # Strip the prefix if it exists
                                        image_data = image_data.split(",", 1)[1]
                                    img_data = base64.b64decode(image_data)
                                    img = Image.open(io.BytesIO(img_data))
                                    img = img.convert("RGB")
                                    images.append(img)
                            elif hasattr(image_data, "read"):  # File-like object
                                img = Image.open(image_data)
                                img = img.convert("RGB")
                                images.append(img)
                        except Exception as e:
                            logger.error(f"Error processing image: {e}")
                
                elif item_type == "video":
                    # Handle video data - requires decord
                    if not HAS_DECORD:
                        logger.warning("Decord not installed - video processing unavailable")
                        continue
                        
                    video_data = item.get("video", "")
                    if video_data:
                        try:
                            if isinstance(video_data, str):
                                if video_data.startswith(("http://", "https://")):
                                    # URL handling would go here
                                    logger.warning("URL video handling not implemented")
                                else:
                                    # Assume base64
                                    if video_data.startswith("data:video"):
                                        video_data = video_data.split(",", 1)[1]
                                    video_bytes = base64.b64decode(video_data)
                                    video_arr = decord.VideoReader(io.BytesIO(video_bytes))
                                    videos.append(video_arr)
                        except Exception as e:
                            logger.error(f"Error processing video: {e}")
                
                elif item_type == "audio":
                    # Handle audio data
                    audio_data = item.get("audio", "")
                    if audio_data:
                        try:
                            if isinstance(audio_data, str):
                                if audio_data.startswith(("http://", "https://")):
                                    # URL handling would go here
                                    logger.warning("URL audio handling not implemented")
                                else:
                                    # Assume base64
                                    if audio_data.startswith("data:audio"):
                                        audio_data = audio_data.split(",", 1)[1]
                                    audio_bytes = base64.b64decode(audio_data)
                                    # Audio processing would go here - typically using librosa
                                    # For now, placeholder
                                    audios.append(None)
                        except Exception as e:
                            logger.error(f"Error processing audio: {e}")
    
    return audios, images, videos