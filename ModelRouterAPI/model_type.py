from enum import Enum

class ModelType(Enum):
    TEXT_GENERATION = "text_generation"
    TEXT_EMBEDDING = "text_embedding"
    RERANKER = "reranker"
