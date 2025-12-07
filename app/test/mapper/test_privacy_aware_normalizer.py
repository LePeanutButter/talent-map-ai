import unittest
from app.mapper.privacy_aware_normalizer import PrivacyAwareNormalizer

class TestPrivacyAwareNormalizer(unittest.TestCase):
    """
    Unit test suite for PrivacyAwareNormalizer.

    This suite verifies that the privacy-preserving text normalization and
    anonymization pipeline works correctly for:
        - Email, phone, URL, and social media anonymization
        - Name and PII removal
        - Education anonymization
        - Combined normalization (whitespace, line breaks)
        - Proper reporting of lengths and PII counts
    """

    def setUp(self):
        """Initialize a PrivacyAwareNormalizer instance for all tests."""
        self.processor = PrivacyAwareNormalizer()

    def test_empty_input(self):
        """
        Test processing of empty string.

        Ensures that empty input returns empty anonymized text
        and zero lengths/PII count.
        """
        result = self.processor.process("")
        self.assertEqual(result["anonymized_text"], "")
        self.assertEqual(result["original_length"], 0)
        self.assertEqual(result["final_length"], 0)
        self.assertEqual(result["pii_removed"], 0)

    def test_email_anonymization(self):
        """
        Test that email addresses are replaced with [EMAIL] placeholder.
        """
        text = "Contact me at john.doe@example.com"
        result = self.processor.process(text)
        self.assertIn("[EMAIL]", result["anonymized_text"])
        self.assertNotIn("john.doe@example.com", result["anonymized_text"])

    def test_phone_number_anonymization(self):
        """
        Test that phone numbers in various formats are replaced with [PHONE].
        """
        text = "Call me at +1 (555) 123-4567 or 555-765-4321"
        result = self.processor.process(text)
        self.assertIn("[PHONE]", result["anonymized_text"])
        self.assertNotIn("555-123-4567", result["anonymized_text"])
        self.assertNotIn("555-765-4321", result["anonymized_text"])

    def test_url_anonymization(self):
        """
        Test that URLs are replaced with [URL].
        """
        text = "My website: https://example.com/profile"
        result = self.processor.process(text)
        self.assertIn("[URL]", result["anonymized_text"])
        self.assertNotIn("https://example.com/profile", result["anonymized_text"])

    def test_social_media_anonymization(self):
        """
        Test that social media handles and profiles are replaced with [SOCIAL].
        """
        text = "Follow me on @username and github.com/johndoe"
        result = self.processor.process(text)
        self.assertIn("[SOCIAL]", result["anonymized_text"])
        self.assertNotIn("@username", result["anonymized_text"])
        self.assertNotIn("github.com/johndoe", result["anonymized_text"])

    def test_name_anonymization(self):
        """
        Test that personal names are replaced with [NAME].
        """
        text = "Juan Pérez\nSoftware Engineer"
        result = self.processor.process(text)
        self.assertIn("[NAME]", result["anonymized_text"])
        self.assertNotIn("Juan Pérez", result["anonymized_text"])

    def test_education_anonymization(self):
        """
        Test that university names are replaced with generic placeholders
        but degree level is preserved.
        """
        text = "Bachelor of Science, University of Example"
        result = self.processor.process(text)
        self.assertIn("Bachelor", result["anonymized_text"])
        self.assertNotIn("University of Example", result["anonymized_text"])

    def test_combined_pii_removal(self):
        """
        Test anonymization of a text containing multiple PII types.

        Verifies that all emails, phones, URLs, social handles, and names are removed.
        """
        text = """
        Name: María López
        Email: maria.lopez@example.com
        Phone: 555-123-4567
        LinkedIn: linkedin.com/in/mlopez
        Bachelor in Engineering, Universidad de Ejemplo
        """
        result = self.processor.process(text)
        anonymized = result["anonymized_text"]
        self.assertIn("[NAME]", anonymized)
        self.assertIn("[EMAIL]", anonymized)
        self.assertIn("[PHONE]", anonymized)
        self.assertIn("[SOCIAL]", anonymized)
        self.assertNotIn("María López", anonymized)
        self.assertNotIn("maria.lopez@example.com", anonymized)
        self.assertNotIn("555-123-4567", anonymized)
        self.assertNotIn("linkedin.com/in/mlopez", anonymized)
        self.assertNotIn("Universidad de Ejemplo", anonymized)
        self.assertIn("Bachelor", anonymized)

    def test_anonymize_cv_for_bert_quick_method(self):
        """
        Test the static convenience method anonymize_cv_for_bert.

        Verifies that it returns the same result as process()['anonymized_text'].
        """
        text = "Email: test@example.com\nPhone: 123-456-7890"
        quick_result = PrivacyAwareNormalizer.anonymize_cv_for_bert(text)
        full_result = self.processor.process(text)["anonymized_text"]
        self.assertEqual(quick_result, full_result)


if __name__ == "__main__":
    unittest.main()
