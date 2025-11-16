from app.models import JobMatchingModel
from app.services.preprocessing_service import CVPreprocessingService
from app.mapper.privacy_aware_normalizer import PrivacyAwareNormalizer


class JobMatchingService:
    """
    Servicio que calcula la compatibilidad entre una oferta laboral y una lista de CVs.
    """

    def __init__(self):
        self.model = JobMatchingModel()
        self.normalizer = PrivacyAwareNormalizer()
        self.preprocessor = CVPreprocessingService()

    def compute_match(self, job_description: str, cv_json_list: list[dict]) -> list[dict]:
        """
        Retorna una lista con los CVs más compatibles con la oferta laboral.
        """
        # Normalizar y anonimizar texto de la oferta
        job_clean = self.normalizer.anonymize_cv_for_bert(job_description)

        # Convertir y normalizar cada CV
        resumes_text = []
        for cv_json in cv_json_list:
            text = self.preprocessor.cv_to_text(cv_json)
            text = self.normalizer.anonymize_cv_for_bert(text)
            resumes_text.append(text)

        # Calcular similitud
        return self.model.match(job_clean, resumes_text)
