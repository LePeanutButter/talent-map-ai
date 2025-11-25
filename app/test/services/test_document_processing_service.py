import unittest
from unittest.mock import patch
from django.core.files.uploadedfile import SimpleUploadedFile
from app.mapper import TextExtractor, PrivacyAwareNormalizer
from app.services.document_processing_service import DocumentProcessingService


class TestDocumentProcessingService(unittest.TestCase):
    """
    Unit test suite for DocumentProcessingService.
    This suite ensures the proper functioning of the document processing pipeline,
    including file validation, text extraction, normalization, and metadata handling.
    """

    def setUp(self):
        """Initialize DocumentProcessingService for all tests."""
        self.service = DocumentProcessingService()
        self.mock_file_content = b"Some file content"
        self.mock_file_name = "sample.pdf"
        self.mock_uploaded_file = SimpleUploadedFile(self.mock_file_name, self.mock_file_content)
        self.metadata = {"doc_id": "test_doc_1", "user_id": "user_1"}

    def test_process_uploaded_file_success_pdf(self):
        """
        Test successful processing of a PDF file.
        Verifies that text extraction, normalization, and metadata preparation work correctly.
        """
        with patch.object(TextExtractor, 'extract_text',
                          return_value={"extracted_text": "Mock PDF Content"}) as mock_extractor:
            with patch.object(PrivacyAwareNormalizer, 'anonymize_cv_for_bert',
                              return_value="Normalized PDF Content") as mock_normalizer:
                result = self.service.process_uploaded_file(self.mock_uploaded_file)

        mock_extractor.assert_called_once()
        mock_normalizer.assert_called_once()

        self.assertTrue(result["success"])
        self.assertIn("extracted_text", result["data"])
        self.assertEqual(result["data"]["extracted_text"], "Normalized PDF Content")
        self.assertEqual(result["status_code"], 200)

    def test_process_uploaded_file_empty_file(self):
        """
        Test processing when no file is uploaded.
        Verifies that the service returns a correct error message for missing files.
        """
        empty_file = None
        result = self.service.process_uploaded_file(empty_file)

        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "No file received")
        self.assertEqual(result["status_code"], 400)

    def test_process_uploaded_file_large_file(self):
        """
        Test processing of a file that exceeds the maximum size limit.
        Verifies that the service returns an error for large files.
        """
        large_file = SimpleUploadedFile("large_file.pdf", b"a" * (self.service.max_file_size + 1))
        result = self.service.process_uploaded_file(large_file)

        self.assertFalse(result["success"])
        self.assertIn("File too large", result["error"])
        self.assertEqual(result["status_code"], 400)

    def test_process_uploaded_file_unsupported_extension(self):
        """
        Test processing of a file with an unsupported extension.
        Verifies that the service returns an error for unsupported file types.
        """
        unsupported_file = SimpleUploadedFile("file.zip", b"Some content")
        result = self.service.process_uploaded_file(unsupported_file)

        self.assertFalse(result["success"])
        self.assertIn("Unsupported file type", result["error"])
        self.assertEqual(result["status_code"], 400)

    def test_process_uploaded_file_text_extraction_failure(self):
        """
        Test processing of a file where text extraction fails.
        Verifies that the service handles extraction failures gracefully.
        """
        with patch.object(TextExtractor, 'extract_text', return_value={"error": "Extraction failed"}):
            result = self.service.process_uploaded_file(self.mock_uploaded_file)

        self.assertFalse(result["success"])
        self.assertIn("Extraction failed", result["error"])
        self.assertEqual(result["status_code"], 500)

    def test_process_uploaded_file_normalization_failure(self):
        """
        Test processing of a file where normalization fails.
        Verifies that the service handles normalization failures gracefully.
        """
        with patch.object(TextExtractor, 'extract_text', return_value={"extracted_text": "Mock Content"}):
            with patch.object(PrivacyAwareNormalizer, 'anonymize_cv_for_bert',
                              side_effect=Exception("Normalization error")):
                result = self.service.process_uploaded_file(self.mock_uploaded_file)

        self.assertFalse(result["success"])
        self.assertIn("Normalization error", result["error"])
        self.assertEqual(result["status_code"], 500)

    def test_process_uploaded_file_no_extension(self):
        """
        Test processing of a file with no extension.
        Verifies that the service returns an error when the file has no extension.
        """
        no_extension_file = SimpleUploadedFile("file_without_extension", b"Some content")
        result = self.service.process_uploaded_file(no_extension_file)

        self.assertFalse(result["success"])
        self.assertIn("File has no extension", result["error"])
        self.assertEqual(result["status_code"], 400)

    def test_prepare_metadata_with_doc_id(self):
        """
        Test metadata preparation with a provided document ID.
        Verifies that the service correctly generates metadata with the supplied doc_id.
        """
        metadata = self.service._prepare_metadata(
            user_id="user_1", doc_id="doc_123", file_name="test.pdf", file_size=1024
        )
        self.assertEqual(metadata["doc_id"], "doc_123")
        self.assertEqual(metadata["user_id"], "user_1")

    def test_prepare_metadata_without_doc_id(self):
        """
        Test metadata preparation when no document ID is provided.
        Verifies that the service generates a unique document ID.
        """
        metadata = self.service._prepare_metadata(
            user_id="user_1", doc_id=None, file_name="test.pdf", file_size=1024
        )
        self.assertTrue(metadata["doc_id"].startswith("doc_"))
        self.assertEqual(metadata["user_id"], "user_1")

    def test_get_file_extension_valid(self):
        """
        Test file extension extraction.
        Verifies that the correct extension is extracted from the filename.
        """
        extension = self.service._get_file_extension("document.pdf")
        self.assertEqual(extension, ".pdf")

    def test_get_file_extension_invalid(self):
        """
        Test file extension extraction for a filename with no extension.
        """
        extension = self.service._get_file_extension("file_without_extension")
        self.assertEqual(extension, "")


if __name__ == "__main__":
    unittest.main()
