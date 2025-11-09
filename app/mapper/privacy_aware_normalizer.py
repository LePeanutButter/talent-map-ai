import re
from app.mapper.privacy_aware_anonymizer import PrivacyAwareAnonymizer
from app.mapper.text_normalizer import TextNormalizer


class PrivacyAwareNormalizer:
    """
    Combined normalizer + anonymizer for privacy-preserving job matching.

    This is the RECOMMENDED approach for CV/Resume processing with BERT.
    """

    def __init__(self):
        """Initialize with anonymizer and normalizer."""
        self.anonymizer = PrivacyAwareAnonymizer()
        self.normalizer = TextNormalizer(
            remove_extra_whitespace=True,
            normalize_line_breaks=True,
            remove_special_chars=False,
            lowercase=False,
            remove_urls=False,
            remove_emails=False,
            max_consecutive_newlines=2
        )

    def process(self, text: str) -> dict:
        """
        Process text through anonymization and normalization pipeline.

        Args:
            text: Raw CV/resume text

        Returns:
            Dict with:
                - anonymized_text: Final processed text
                - original_length: Length before processing
                - final_length: Length after processing
                - pii_removed: Approximate count of PII items removed
        """
        if not text:
            return {
                "anonymized_text": "",
                "original_length": 0,
                "final_length": 0,
                "pii_removed": 0
            }

        original_length = len(text)
        anonymized = self.anonymizer.anonymize(text)
        normalized = self.normalizer.normalize(anonymized)
        pii_count = (
                text.count('@') +
                len(re.findall(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', text)) +
                text.count('http') +
                text.count('linkedin')
        )

        return {
            "anonymized_text": normalized,
            "original_length": original_length,
            "final_length": len(normalized),
            "pii_removed": pii_count
        }

    @staticmethod
    def anonymize_cv_for_bert(text: str) -> str:
        """
        Quick function to anonymize and normalize CV text for BERT.

        Usage:
            clean_text = anonymize_cv_for_bert(raw_cv_text)

        Args:
            text: Raw CV/resume text

        Returns:
            Anonymized and normalized text
        """
        processor = PrivacyAwareNormalizer()
        result = processor.process(text)
        return result["anonymized_text"]
