from transformers import AutoTokenizer, AutoModel
import torch

class HuggingFaceEmbeddingGenerator:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.torch = torch

    def generate_embedding(self, text: str) -> list[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        sentence_embeddings = outputs.last_hidden_state[:, 0]
        normalized = self.torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return normalized.squeeze().tolist()