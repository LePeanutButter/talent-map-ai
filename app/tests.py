from django.test import TestCase

# Create your tests here.
import json
from django.test import TestCase
from app.services.job_matching_service import JobMatchingService
from pathlib import Path

class JobMatchingDatasetTest(TestCase):
    def setUp(self):
        self.service = JobMatchingService()

        # Ruta al dataset real
        dataset_path = Path("data/master_resumes.jsonl")

        # Leemos solo los primeros 50 CVs para no sobrecargar la prueba
        self.cv_samples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 4817:  # cambia a más si quieres probar más CVs
                    break
                try:
                    self.cv_samples.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Error al leer línea {i}")

        # Oferta laboral de prueba
        self.job_offer = """
        We are looking for a Python Developer experienced in Django, REST APIs, 
        and machine learning with TensorFlow. Knowledge of OpenCV and cloud deployment is a plus.
        """

    def test_real_dataset(self):
        """
        Prueba del modelo de matching usando el dataset real master_resumes.jsonl
        """
        results = self.service.compute_match(self.job_offer, self.cv_samples)

        # Validaciones básicas
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        # Mostramos los 10 mejores resultados
        print("\nTop 10 resultados del Job Matching:")
        for r in results[:10]:
            print(f"CV {r['resume_index']} → {r['match_score']}% match")

        # Aseguramos que los puntajes sean válidos
        for r in results:
            self.assertGreaterEqual(r["match_score"], 0)
            self.assertLessEqual(r["match_score"], 100)
