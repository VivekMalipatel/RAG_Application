from enum import Enum

class ModelType(Enum):
    TEXT_GENERATION = "text_generation"
    TEXT_EMBEDDING = "text_embedding"
    IMAGE_EMBEDDING = "image_embedding"
    RERANKER = "reranker"
