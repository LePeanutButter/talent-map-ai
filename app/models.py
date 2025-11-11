from django.db import models

# Create your models here.
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch


class JobMatchingModel:
    """
    Modelo de compatibilidad laboral basado en embeddings BERT.
    Calcula el porcentaje de coincidencia entre una oferta de trabajo y un conjunto de CVs.
    """

    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _embed_text(self, text: str) -> np.ndarray:
        """Obtiene el embedding promedio de un texto."""
        if not text.strip():
            return np.zeros(768)

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        return embeddings[0]

    def match(self, job_description: str, resumes: list[str]) -> list[dict]:
        """
        Calcula el porcentaje de compatibilidad entre una oferta laboral y varios CVs.
        Retorna una lista con los resultados.
        """
        job_embedding = self._embed_text(job_description)
        results = []

        for idx, resume_text in enumerate(resumes):
            resume_embedding = self._embed_text(resume_text)
            similarity = cosine_similarity(
                job_embedding.reshape(1, -1),
                resume_embedding.reshape(1, -1)
            )[0][0]

            results.append({
                "resume_index": idx,
                "match_score": round(float(similarity * 100), 2)
            })

        return sorted(results, key=lambda x: x["match_score"], reverse=True)
