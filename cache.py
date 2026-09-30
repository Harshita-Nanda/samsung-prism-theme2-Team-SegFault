import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticCache:
    def __init__(self, threshold=0.50):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold
        self.vectors = []   # har sawaal ke 384 numbers
        self.answers = []   # vectors[i] ka jawab answers[i] me

    def get(self, query):
        if not self.vectors:
            return None, 0.0
        q = self.model.encode(query, normalize_embeddings=True)
        scores = np.array(self.vectors) @ q
        best = int(scores.argmax())
        if scores[best] >= self.threshold:
            return self.answers[best], float(scores[best])
        return None, float(scores[best])

    def put(self, query, answer):
        v = self.model.encode(query, normalize_embeddings=True)
        self.vectors.append(v)
        self.answers.append(answer)