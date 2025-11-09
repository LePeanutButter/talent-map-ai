from app.mapper.text_normalizer import TextNormalizer

class NormalizerPresets:
    """Common normalizer configurations for different use cases."""

    @staticmethod
    def default() -> TextNormalizer:
        """Default configuration - basic cleanup."""
        return TextNormalizer(
            remove_extra_whitespace=True,
            normalize_line_breaks=True,
            remove_special_chars=False,
            lowercase=False,
            remove_urls=False,
            remove_emails=False,
            max_consecutive_newlines=2
        )

    @staticmethod
    def aggressive() -> TextNormalizer:
        """Aggressive cleanup - remove everything non-essential."""
        return TextNormalizer(
            remove_extra_whitespace=True,
            normalize_line_breaks=True,
            remove_special_chars=True,
            lowercase=True,
            remove_urls=True,
            remove_emails=True,
            max_consecutive_newlines=1
        )

    @staticmethod
    def minimal() -> TextNormalizer:
        """Minimal cleanup - preserve as much as possible."""
        return TextNormalizer(
            remove_extra_whitespace=True,
            normalize_line_breaks=True,
            remove_special_chars=False,
            lowercase=False,
            remove_urls=False,
            remove_emails=False,
            max_consecutive_newlines=3
        )

    @staticmethod
    def search_optimized() -> TextNormalizer:
        """Optimized for search indexing."""
        return TextNormalizer(
            remove_extra_whitespace=True,
            normalize_line_breaks=True,
            remove_special_chars=False,
            lowercase=True,
            remove_urls=True,
            remove_emails=False,
            max_consecutive_newlines=1
        )

    @staticmethod
    def for_job_matching() -> TextNormalizer:
        """
        Optimal configuration for job matching with BERT.

        IMPORTANT: This does NOT include PII removal.
        Use for_job_matching_with_privacy() for anonymization.
        """
        return TextNormalizer(
            remove_extra_whitespace=True,
            normalize_line_breaks=True,
            remove_special_chars=False,
            lowercase=False,
            remove_urls=True,
            remove_emails=False,
            max_consecutive_newlines=2
        )

    @staticmethod
    def for_skills_extraction() -> TextNormalizer:
        """
        Optimized for extracting technical skills and keywords.

        More aggressive cleaning since we focus on specific terms.
        """
        return TextNormalizer(
            remove_extra_whitespace=True,
            normalize_line_breaks=True,
            remove_special_chars=False,
            lowercase=False,
            remove_urls=True,
            remove_emails=True,
            max_consecutive_newlines=1
        )

    @staticmethod
    def aggressive_for_embeddings() -> TextNormalizer:
        """
        More aggressive cleaning for pure semantic embeddings.

        Use this if you want to focus on semantic content only,
        removing all noise. Good for initial experimentation.
        """
        return TextNormalizer(
            remove_extra_whitespace=True,
            normalize_line_breaks=True,
            remove_special_chars=False,
            lowercase=True,
            remove_urls=True,
            remove_emails=True,
            max_consecutive_newlines=1
        )
