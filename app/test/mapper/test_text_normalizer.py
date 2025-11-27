import unittest
from app.mapper import TextNormalizer

class TestTextNormalizer(unittest.TestCase):
    """
    Unit test suite for the TextNormalizer class.

    This suite verifies that individual and combined text normalization operations
    behave correctly. Operations tested include:
        - Normalizing line breaks
        - Removing extra whitespace
        - Removing URLs and emails
        - Removing special characters
        - Converting to lowercase
        - Limiting consecutive newlines
        - Removing extraction errors
    """

    def test_normalize_line_breaks(self):
        """
        Test normalization of line breaks.

        Ensures that different types of line breaks (\r, \r\n) are converted to '\n'.
        """
        norm = TextNormalizer(normalize_line_breaks=True)
        text = "Line1\rLine2\r\nLine3"
        expected = "Line1\nLine2\nLine3"
        self.assertEqual(norm.normalize(text), expected)

    def test_remove_extra_whitespace(self):
        """
        Test removal of extra whitespace.

        Consecutive spaces within lines and leading/trailing spaces are reduced to a single space.
        """
        norm = TextNormalizer(remove_extra_whitespace=True)
        text = "This   is    a   test.\n   New line."
        expected = "This is a test.\nNew line."
        self.assertEqual(norm.normalize(text), expected)

    def test_remove_urls(self):
        """
        Test removal of URLs from text.

        Ensures that web links starting with http/https are stripped.
        """
        norm = TextNormalizer(remove_urls=True)
        text = "Visit https://example.com for info"
        expected = "Visit for info"
        self.assertEqual(norm.normalize(text), expected.strip())

    def test_remove_emails(self):
        """
        Test removal of email addresses.

        Ensures any text matching email patterns is removed.
        """
        norm = TextNormalizer(remove_emails=True)
        text = "Contact me at test@example.com please."
        expected = "Contact me at please."
        self.assertEqual(norm.normalize(text), expected.strip())

    def test_remove_special_characters(self):
        """
        Test removal of special characters.

        Removes punctuation and symbols, leaving only letters, numbers, and spaces.
        """
        norm = TextNormalizer(remove_special_chars=True)
        text = "Hello @World! #2025 :)"
        expected = "Hello World 2025 "
        self.assertEqual(norm.normalize(text), expected.strip())

    def test_lowercase(self):
        """
        Test conversion of text to lowercase.
        """
        norm = TextNormalizer(lowercase=True)
        text = "Hello WORLD"
        self.assertEqual(norm.normalize(text), "hello world")

    def test_limit_consecutive_newlines(self):
        """
        Test limiting the number of consecutive newlines.

        Ensures that sequences of newlines longer than max_consecutive_newlines
        are reduced to the allowed maximum.
        """
        norm = TextNormalizer(max_consecutive_newlines=2)
        text = "Line1\n\n\n\nLine2"
        expected = "Line1\n\nLine2"
        self.assertEqual(norm.normalize(text), expected)

    def test_combined_operations(self):
        """
        Test multiple normalization operations together.

        This includes removing URLs, emails, special characters, converting to lowercase,
        and limiting consecutive newlines.
        """
        norm = TextNormalizer(
            remove_urls=True,
            remove_emails=True,
            remove_special_chars=True,
            lowercase=True,
            max_consecutive_newlines=1
        )
        text = "Hello!!! Visit https://site.com\n\nEmail: test@a.com"
        expected = "hello visit email"
        self.assertEqual(norm.normalize(text), expected)

    def test_remove_extraction_errors(self):
        """
        Test removal of extraction errors from text.

        Ensures lines marked as errors or warnings are removed.
        """
        text =  """
                Error: something failed
                This is text.
                [Warning: bad state]
                More text.
                """
        expected = "This is text.\nMore text."
        self.assertEqual(TextNormalizer.remove_extraction_errors(text), expected)

if __name__ == "__main__":
    unittest.main()
