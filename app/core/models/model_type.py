
from enum import Enum

class ModelType(Enum):
    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    TEXT_EMBEDDING = "text_embedding"
    IMAGE_EMBEDDING = "image_embedding"
    RERANKER = "reranker"
    NER = "ner"
