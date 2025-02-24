import logging
import asyncio
from typing import List, Dict, Any
import torch
from imagebind import data
from imagebind  import imagebind_model
from imagebind  import ModalityType
from imagebind  import load_and_transform_audio_data, load_and_transform_vision_data, load_and_transform_text_data


class ImageBindClient:
    """
    ImageBind client for multimodal embedding generation
    Supports: text, image, audio, video, depth, thermal, and IMU
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load ImageBind model with proper device placement"""
        try:
            self.model = imagebind_model.imagebind_huge(pretrained=True)
            self.model.eval()
            self.model.to(self.device)
            self.logger.info("ImageBind model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load ImageBind model: {str(e)}")
            raise

    async def get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Generate text embeddings using ImageBind
        Args:
            text: Input text to embed
        Returns:
            torch.Tensor: Normalized text embedding
        """
        try:
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text_data(
                    [text], self.device
                )
            }
            
            return await self._get_embeddings(inputs, ModalityType.TEXT)
        
        except Exception as e:
            self.logger.error(f"Text embedding failed: {str(e)}")
            return torch.tensor([])

    async def _get_embeddings(self, inputs: Dict, modality) -> torch.Tensor:
        """Core embedding generation method"""
        try:
            with torch.no_grad(), torch.cuda.amp.autocast():
                embeddings = self.model(inputs)
                normalized = torch.nn.functional.normalize(
                    embeddings[modality], dim=-1
                )
            return normalized.cpu()
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            return torch.tensor([])

    # Add other modality methods following the same pattern
    async def get_image_embedding(self, image_path: str):
        """Generate image embeddings"""
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data(
                [image_path], self.device
            )
        }
        return await self._get_embeddings(inputs, ModalityType.VISION)

    async def get_audio_embedding(self, audio_path: str):
        """Generate audio embeddings"""
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(
                [audio_path], self.device
            )
        }
        return await self._get_embeddings(inputs, ModalityType.AUDIO)
