import re

class TextNormalizer:
    """
    Service responsible for normalizing and cleaning extracted text.

    This normalizer applies various text cleaning and formatting operations
    to prepare extracted text for storage, indexing, or further processing.
    """

    def __init__(
            self,
            remove_extra_whitespace: bool = True,
            normalize_line_breaks: bool = True,
            remove_special_chars: bool = False,
            lowercase: bool = False,
            remove_urls: bool = False,
            remove_emails: bool = False,
            max_consecutive_newlines: int = 2
    ):
        """
        Initialize the text normalizer with configuration options.

        Args:
            remove_extra_whitespace: Remove multiple spaces, tabs
            normalize_line_breaks: Standardize line breaks to \n
            remove_special_chars: Remove non-alphanumeric characters (except basic punctuation)
            lowercase: Convert all text to lowercase
            remove_urls: Remove HTTP/HTTPS URLs
            remove_emails: Remove email addresses
            max_consecutive_newlines: Maximum number of consecutive newlines allowed
        """
        self.remove_extra_whitespace = remove_extra_whitespace
        self.normalize_line_breaks = normalize_line_breaks
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.max_consecutive_newlines = max_consecutive_newlines

    def normalize(self, text: str) -> str:
        """
        Apply all configured normalization operations to the text.

        Args:
            text: Raw text to normalize

        Returns:
            Normalized text string
        """
        if not text or not isinstance(text, str):
            return ""

        normalized = text

        if self.normalize_line_breaks:
            normalized = self._normalize_line_breaks(normalized)

        if self.remove_urls:
            normalized = self._remove_urls(normalized)

        if self.remove_emails:
            normalized = self._remove_emails(normalized)

        if self.remove_special_chars:
            normalized = self._remove_special_characters(normalized)

        if self.remove_extra_whitespace:
            normalized = self._remove_extra_whitespace(normalized)

        if self.lowercase:
            normalized = normalized.lower()

        normalized = self._limit_consecutive_newlines(
            normalized, max_count=self.max_consecutive_newlines
        )

        return normalized.strip()

    @staticmethod
    def _normalize_line_breaks(text: str) -> str:
        """Convert all line break variants to standard \n."""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _remove_extra_whitespace(text: str) -> str:
        """Remove extra spaces and tabs while preserving single spaces."""
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r' *\n *', '\n', text)
        return text

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Remove HTTP/HTTPS URLs from text."""
        pattern = r"\s*https?://[^\s]+\s*"
        return re.sub(pattern, " ", text).strip()

    @staticmethod
    def _remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        pattern = r"\s*\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b\s*"
        return re.sub(pattern, " ", text).strip()

    @staticmethod
    def _remove_special_characters(text: str) -> str:
        """
        Remove special characters, keeping letters, numbers, and basic punctuation.

        Preserves: letters, numbers, spaces, periods, commas, newlines, hyphens, apostrophes
        """
        pattern = r"[^a-zA-Z0-9 .,'\-\n]"
        return re.sub(pattern, "", text)

    @staticmethod
    def _limit_consecutive_newlines(text: str, max_count: int = 2) -> str:
        """Limit consecutive newlines to a maximum count."""
        if max_count < 1:
            max_count = 1
        pattern = r"\n{" + str(max_count + 1) + r",}"
        return re.sub(pattern, "\n" * max_count, text)

    @staticmethod
    def remove_extraction_errors(text: str) -> str:
        """
        Remove common error messages from extraction results.

        This is useful for cleaning up error messages that might have been
        included in the extracted text by the extraction methods.
        """
        error_patterns = [
            r"^Error:.*?$",
            r"^Warning:.*?$",
            r"^\[Error.*?\]$",
            r"^\[Warning.*?\]$"
        ]

        cleaned_lines = []
        for line in text.split("\n"):
            if any(re.search(pat, line.strip(), flags=re.IGNORECASE) for pat in error_patterns):
                continue
            cleaned_lines.append(line.strip())

        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()