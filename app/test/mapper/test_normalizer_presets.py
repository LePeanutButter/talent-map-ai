import unittest
from app.mapper import NormalizerPresets

class TestNormalizerPresets(unittest.TestCase):
    """
    Unit test suite for the NormalizerPresets class.

    This suite verifies that predefined normalization presets apply the correct
    combination of normalization operations consistently.
    """

    def test_default_preset(self):
        """
        Test normalization using the default preset.

        The default preset performs basic normalization: line breaks normalization
        and reduction of extra whitespace.
        """
        norm = NormalizerPresets.default()
        text = "Hello   World\r\n\r\nTest"
        expected = "Hello World\n\nTest"
        self.assertEqual(norm.normalize(text), expected)

    def test_aggressive_preset(self):
        """
        Test normalization using the aggressive preset.

        Aggressive preset removes URLs, emails, special characters, lowercases text,
        and limits consecutive newlines.
        """
        norm = NormalizerPresets.aggressive()
        text = "Hello!!! Visit https://x.com.\nEMAIL: test@site.com"
        expected = "hello visit email"
        self.assertEqual(norm.normalize(text), expected)

    def test_minimal_preset(self):
        """
        Test normalization using the minimal preset.

        Minimal preset preserves most formatting but normalizes whitespace.
        """
        norm = NormalizerPresets.minimal()
        text = "A   B\r\n\r\n\r\nC"
        expected = "A B\n\n\nC"
        self.assertEqual(norm.normalize(text), expected)

    def test_search_optimized_preset(self):
        """
        Test normalization using the search-optimized preset.

        Search-optimized preset prepares text for search: lowercases text, removes URLs,
        and normalizes line breaks.
        """
        norm = NormalizerPresets.search_optimized()
        text = "Hello World\n\nVisit http://test.com"
        expected = "hello world\nvisit"
        self.assertEqual(norm.normalize(text), expected)

    def test_for_job_matching_preset(self):
        """
        Test normalization using the job-matching preset.

        Removes URLs while preserving line breaks for structured job descriptions.
        """
        norm = NormalizerPresets.for_job_matching()
        text = "Job data\n\nSee https://example.com"
        expected = "Job data\n\nSee"
        self.assertEqual(norm.normalize(text), expected)

    def test_for_skills_extraction_preset(self):
        """
        Test normalization using the skills-extraction preset.

        Removes emails while keeping line breaks and structured text for skill parsing.
        """
        norm = NormalizerPresets.for_skills_extraction()
        text = "Skills: python, js\nContact test@mail.com"
        expected = "Skills: python, js\nContact"
        self.assertEqual(norm.normalize(text), expected)

    def test_aggressive_for_embeddings_preset(self):
        """
        Test normalization using the aggressive-for-embeddings preset.

        Keeps punctuation but removes URLs and lowercases text for embedding generation.
        """
        norm = NormalizerPresets.aggressive_for_embeddings()
        text = "Hello WORLD!!! Visit https://x.com\nEmail x@y.com"
        expected = "hello world!!! visit email"
        self.assertEqual(norm.normalize(text), expected)


if __name__ == '__main__':
    unittest.main()
