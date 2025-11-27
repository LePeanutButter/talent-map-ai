import unittest
from app.mapper.privacy_aware_anonymizer import PrivacyAwareAnonymizer


class TestPrivacyAwareAnonymizer(unittest.TestCase):
    """
    Unit test suite for the PrivacyAwareAnonymizer class.

    This suite verifies that the PrivacyAwareAnonymizer class correctly anonymizes
    and removes sensitive information from text. The operations tested include:
        - Anonymizing emails
        - Removing phone numbers
        - Removing URLs
        - Anonymizing social media handles
        - Removing personal identification numbers
        - Removing addresses
        - Removing age-related information
        - Removing gender references
        - Removing marital status
        - Anonymizing education details
        - Removing personal names
        - Cleaning up formatting (e.g., redundant newlines)
        - Full anonymization pipeline

    Each operation checks that sensitive information is properly replaced or removed,
    and that non-sensitive content remains unchanged.
    """

    def setUp(self):
        """
        Set up the PrivacyAwareAnonymizer instance for testing.
        """
        self.anonymizer = PrivacyAwareAnonymizer()

    def test_empty_input(self):
        """
        Test anonymization with empty input.

        Ensures that an empty string returns an empty string without modification.
        """
        self.assertEqual(self.anonymizer.anonymize(""), "")

    def test_email_removal(self):
        """
        Test removal of email addresses from the text.

        Ensures that email addresses are replaced with the placeholder '[EMAIL]'.
        """
        text = "Contact: john.doe@example.com"
        result = self.anonymizer._remove_emails(text)
        self.assertIn("[EMAIL]", result)
        self.assertNotIn("john.doe@example.com", result)

    def test_phone_removal(self):
        """
        Test removal of phone numbers from the text.

        Ensures that phone numbers are replaced with the placeholder '[PHONE]'.
        """
        text = "Call me at +1 (555) 123-4567"
        result = self.anonymizer._remove_phone_numbers(text)
        self.assertIn("[PHONE]", result)
        self.assertNotIn("555-123-4567", result)

    def test_url_removal(self):
        """
        Test removal of URLs from the text.

        Ensures that URLs are replaced with the placeholder '[URL]'.
        """
        text = "My site: https://example.com"
        result = self.anonymizer._remove_urls(text)
        self.assertIn("[URL]", result)
        self.assertNotIn("https://example.com", result)

    def test_social_media_removal(self):
        """
        Test removal of social media handles and links.

        Ensures that social media handles and URLs are replaced with '[SOCIAL]'.
        """
        text = "Follow me @username or github.com/johndoe"
        result = self.anonymizer._remove_social_media(text)
        self.assertIn("[SOCIAL]", result)
        self.assertNotIn("@username", result)
        self.assertNotIn("github.com/johndoe", result)

    def test_id_removal(self):
        """
        Test removal of personal identification numbers (ID).

        Ensures that ID numbers are replaced with the placeholder '[ID]'.
        """
        text = "ID: 123456789"
        result = self.anonymizer._remove_id_numbers(text)
        self.assertIn("[ID]", result)
        self.assertNotIn("123456789", result)

    def test_address_removal(self):
        """
        Test removal of addresses from the text.

        Ensures that street addresses and zip codes are replaced with '[ADDRESS]'.
        """
        text = "123 Main Street, 90210"
        result = self.anonymizer._remove_addresses(text)
        self.assertIn("[ADDRESS]", result)
        self.assertNotIn("123 Main Street", result)
        self.assertNotIn("90210", result)

    def test_age_removal(self):
        """
        Test removal of age-related references from the text.

        Ensures that age-related information is replaced with '[AGE]'.
        """
        text = "I am 25 years old. Age: 30. Born in 1990."
        result = self.anonymizer._remove_age_references(text)
        self.assertIn("[AGE]", result)
        self.assertNotIn("25 years old", result)
        self.assertNotIn("1990", result)

    def test_gender_removal(self):
        """
        Test removal of gender references from the text.

        Ensures that gender-related information is removed from the text.
        """
        text = "Gender: male, Sexo: femenino"
        result = self.anonymizer._remove_gender_references(text)
        self.assertNotIn("male", result)
        self.assertNotIn("femenino", result)

    def test_marital_status_removal(self):
        """
        Test removal of marital status references from the text.

        Ensures that marital status information is removed from the text.
        """
        text = "Married, Estado Civil: single"
        result = self.anonymizer._remove_marital_status(text)
        self.assertNotIn("Married", result)
        self.assertNotIn("single", result)

    def test_education_anonymization(self):
        """
        Test anonymization of education details.

        Ensures that university names are replaced with placeholders.
        If no degree keyword is found, a generic placeholder '[EDUCATION]' is used.
        """
        text = "Bachelor of Science, University of Example"
        result = self.anonymizer._anonymize_education(text)
        self.assertIn("Bachelor", result)
        self.assertNotIn("University of Example", result)

        # No degree keyword -> should use placeholder
        text2 = "Instituto de Ejemplo"
        result2 = self.anonymizer._anonymize_education(text2)
        self.assertIn("[EDUCATION]", result2)

    def test_name_removal(self):
        """
        Test removal of personal names from the text.

        Ensures that names are replaced with '[NAME]', while keeping other non-name information intact.
        """
        text = "Juan Pérez\nSoftware Engineer"
        result = self.anonymizer._remove_personal_names(text)
        self.assertIn("[NAME]", result)
        self.assertNotIn("Juan Pérez", result)

        text2 = "Bachelor of Science\nJohn Doe"
        result2 = self.anonymizer._remove_personal_names(text2)
        self.assertIn("Bachelor of Science", result2)
        self.assertIn("[NAME]", result2)
        self.assertNotIn("John Doe", result2)

    def test_cleanup_formatting(self):
        """
        Test cleanup of unnecessary formatting from the text.

        Ensures that redundant newlines and spaces are removed, while placeholders remain intact.
        """
        text = "Line1   \n\n\n[EMAIL]  \n\nLine2"
        result = self.anonymizer._cleanup_formatting(text)
        self.assertNotIn("\n\n\n", result)
        self.assertIn("[EMAIL]", result)
        self.assertIn("Line1", result)
        self.assertIn("Line2", result)

    def test_full_anonymize_pipeline(self):
        """
        Test the full anonymization pipeline.

        Ensures that a combination of personal data (name, email, phone, social media, etc.)
        is fully anonymized and replaced with appropriate placeholders.
        """
        text = """
        Name: María López
        Email: maria.lopez@example.com
        Phone: +1 (555) 123-4567
        LinkedIn: linkedin.com/in/mlopez
        Bachelor in Engineering, Universidad de Ejemplo
        Age: 30
        Gender: female
        Married
        """
        result = self.anonymizer.anonymize(text)
        self.assertIn("[NAME]", result)
        self.assertIn("[EMAIL]", result)
        self.assertIn("[PHONE]", result)
        self.assertIn("[SOCIAL]", result)
        self.assertIn("Bachelor", result)
        self.assertNotIn("María López", result)
        self.assertNotIn("maria.lopez@example.com", result)
        self.assertNotIn("+1 (555) 123-4567", result)
        self.assertNotIn("linkedin.com/in/mlopez", result)
        self.assertNotIn("Universidad de Ejemplo", result)
        self.assertNotIn("30", result)
        self.assertNotIn("female", result)
        self.assertNotIn("Married", result)


if __name__ == "__main__":
    unittest.main()
