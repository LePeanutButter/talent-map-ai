import unittest
from unittest.mock import MagicMock, patch
from app.mapper import TextExtractor
from PIL import UnidentifiedImageError

MOCK_PDF_TEXT = "Mock PDF Content"
MOCK_DOCX_TEXT = "Mock DOCX Content"
MOCK_IMAGE_TEXT = "Mock Image Content"
MOCK_JSON_TEXT = '{"text": "Mock JSON Text"}'
MOCK_PLAINTEXT = "Mock Plain Text"
MOCK_CONTENT = b'some dummy bytes'

class TestTextExtractor(unittest.TestCase):
    """
    Unit test suite for the TextExtractor class.

    This suite validates the behavior of the TextExtractor class, which is responsible
    for extracting text from various file formats such as PDF, DOCX, images, JSON, etc.

    The tests cover:
        - Extraction from DOCX files
        - Extraction from PDFs
        - Error handling (e.g., image OCR failure, unsupported formats)
        - File extraction via storage client
        - Extraction from specific file formats like JSON and images
    """

    def setUp(self):
        """
        Initializes a TextExtractor instance and metadata for use in the tests.
        """
        self.extractor = TextExtractor()
        self.metadata = {"doc_id": "test_doc_1", "user": "user_1"}

    @patch('app.mapper.text_extractor.TextExtractor._extract_from_docx', return_value=MOCK_DOCX_TEXT)
    def test_extract_text_from_docx_with_content_and_explicit_ext(self, mock_extractor):
        """
        Test extraction from a DOCX file using provided content and explicit extension.

        This test verifies that the method correctly uses the content and the provided
        ".docx" extension to extract text from a DOCX file.
        """
        extractor = TextExtractor()
        result = extractor.extract_text(
            file_path="unknown_file",
            metadata=self.metadata,
            file_content=MOCK_CONTENT,
            extension=".docx"
        )

        mock_extractor.assert_called_once_with(MOCK_CONTENT)
        self.assertEqual(result["extracted_text"], MOCK_DOCX_TEXT)
        self.assertEqual(result["file_type"], ".docx")

    @patch('app.mapper.text_extractor.TextExtractor._extract_from_pdf', return_value=MOCK_PDF_TEXT)
    def test_extract_text_from_pdf_with_content(self, mock_extractor):
        """
        Test extraction from a PDF file using provided content.

        This test ensures that the method can extract text from a PDF file and that
        the appropriate file type is returned.
        """
        extractor = TextExtractor()
        result = extractor.extract_text(
            file_path="sample.pdf",
            metadata=self.metadata,
            file_content=MOCK_CONTENT
        )

        mock_extractor.assert_called_once_with(MOCK_CONTENT)
        self.assertEqual(result["extracted_text"], MOCK_PDF_TEXT)
        self.assertEqual(result["file_type"], ".pdf")
        self.assertEqual(result["doc_id"], "test_doc_1")
        self.assertIsNone(result.get("error"))

    @patch('app.mapper.text_extractor.Image.open', side_effect=UnidentifiedImageError("Unsupported image object"))
    @patch('app.mapper.text_extractor.pytesseract.image_to_string', side_effect=Exception("OCR failure"))
    def test_extract_text_catches_internal_exception(self, mock_ocr, mock_image_open):
        """
        Test handling of exceptions from external libraries (e.g., Image.open, OCR failure).

        This test checks if the method correctly handles exceptions thrown by
        external libraries like Image.open or pytesseract.
        """
        image_content = b'mock image bytes'

        result = self.extractor.extract_text(
            file_path="test.jpg",
            metadata=self.metadata,
            file_content=image_content
        )
        self.assertIn("Extraction failed: Unsupported image object", result.get("error", ""))
        self.assertNotIn("extracted_text", result)

        mock_image_open.side_effect = None
        mock_ocr.side_effect = Exception("OCR failure")
        result = self.extractor.extract_text(
            file_path="test.jpg",
            metadata=self.metadata,
            file_content=image_content
        )
        self.assertIn("Extraction failed: OCR failure", result.get("error", ""))

    def test_extract_text_unsupported_format(self):
        """
        Test handling of unsupported file extensions.

        This test ensures that the method handles cases where an unsupported file
        extension (e.g., `.zip`) is provided.
        """
        result = self.extractor.extract_text(
            file_path="sample.zip",
            metadata=self.metadata,
            file_content=b'zip content'
        )
        self.assertIn("Unsupported or unknown file format: .zip", result.get("error", ""))
        self.assertEqual(result["file_type"], ".zip")

    def test_extract_text_no_content_and_no_storage(self):
        """
        Test handling of missing content and no available storage client.

        This test ensures that the method returns an error when no content is
        provided and no storage client is available.
        """
        result = self.extractor.extract_text("sample.pdf", self.metadata, file_content=None)
        self.assertIn("No content provided and no storage client available", result.get("error", ""))

    def test_extract_text_from_storage(self):
        """
        Test extraction using a mock storage client.

        This test verifies that the method correctly extracts text from a file stored
        on an external storage client (mocked in this case).
        """
        mock_storage = MagicMock()
        mock_storage.load_file_content.return_value = MOCK_PLAINTEXT.encode('utf-8')
        mock_storage.get_file_extension.return_value = ".txt"
        storage_extractor = TextExtractor(storage_client=mock_storage)

        result = storage_extractor.extract_text("remote/path/file.txt", self.metadata)

        mock_storage.load_file_content.assert_called_once_with("remote/path/file.txt")
        mock_storage.get_file_extension.assert_called_once_with("remote/path/file.txt")
        self.assertEqual(result["extracted_text"], MOCK_PLAINTEXT)
        self.assertIsNone(result.get("error"))

    def test_extract_text_file_not_found_from_storage(self):
        """
        Test handling of FileNotFoundError from the storage client.

        This test ensures that the method handles cases where the file is not
        found in the storage client and returns an appropriate error message.
        """
        mock_storage = MagicMock()
        mock_storage.load_file_content.side_effect = FileNotFoundError("test_file.txt")
        storage_extractor = TextExtractor(storage_client=mock_storage)

        result = storage_extractor.extract_text("remote/path/file.txt", self.metadata)

        self.assertIn("File not found: test_file.txt", result.get("error", ""))

    def test_get_extension_from_path(self):
        """
        Test extraction of file extensions from various paths.

        This test ensures that the method correctly extracts file extensions
        from different file paths, including cases with uppercase extensions
        and paths with multiple dots.
        """
        self.assertEqual(TextExtractor._get_extension_from_path("sample.pdf"), ".pdf")
        self.assertEqual(TextExtractor._get_extension_from_path("FILE.DOCX"), ".docx")
        self.assertEqual(TextExtractor._get_extension_from_path("/path/to/archive.v1.json"), ".json")
        self.assertEqual(TextExtractor._get_extension_from_path("no_ext_file"), "")
        self.assertEqual(TextExtractor._get_extension_from_path("filename."), ".")

    @patch('app.mapper.text_extractor.PdfReader', autospec=True)
    def test_extract_from_pdf(self, mock_pdf_reader):
        """
        Test extraction from a PDF document.

        This test ensures that the method correctly extracts text from a multi-page
        PDF using the PdfReader library.
        """
        mock_page_1 = MagicMock()
        mock_page_1.extract_text.return_value = "Page 1 text"
        mock_page_2 = MagicMock()
        mock_page_2.extract_text.return_value = "Page 2 text"

        mock_reader = mock_pdf_reader.return_value
        mock_reader.pages = [mock_page_1, mock_page_2]

        content = b'mock pdf bytes'
        extracted = TextExtractor._extract_from_pdf(content)

        self.assertEqual(extracted, "Page 1 text\n\nPage 2 text")
        mock_pdf_reader.assert_called_once()

    @patch('app.mapper.text_extractor.docx.Document')
    def test_extract_from_docx(self, mock_document):
        """
        Test extraction from a DOCX document.

        This test verifies that text is correctly extracted from a DOCX file,
        including text from paragraphs and tables.
        """
        mock_doc = mock_document.return_value
        mock_doc.paragraphs = [
            MagicMock(text="Paragraph 1"),
            MagicMock(text="  "),
            MagicMock(text="Paragraph 2")
        ]
        mock_cell_1 = MagicMock(text="Cell A")
        mock_cell_2 = MagicMock(text="Cell B")
        mock_row = MagicMock(cells=[mock_cell_1, mock_cell_2])
        mock_table = MagicMock(rows=[mock_row])
        mock_doc.tables = [mock_table]

        content = b'mock docx bytes'
        extracted = TextExtractor._extract_from_docx(content)

        expected_text = (
            "Paragraph 1\nParagraph 2\n\n"
            "--- Tables ---\n"
            "Cell A | Cell B"
        )
        self.assertEqual(extracted, expected_text)
        mock_document.assert_called_once()

    def test_extract_from_json_with_text_field(self):
        """
        Test extraction from JSON content with a text field.

        This test verifies that the method correctly extracts text from a JSON
        string where the text is stored in a specific field. It checks that
        the extracted text matches the expected value.
        """
        content = MOCK_JSON_TEXT.encode('utf-8')
        extracted = TextExtractor._extract_from_json(content)
        self.assertEqual(extracted, "Mock JSON Text")

    def test_extract_from_json_invalid(self):
        """
        Test extraction from invalid JSON content.

        This test ensures that the method handles incomplete or malformed JSON
        and returns the original content when extraction fails.
        """
        invalid_content = b'{"key": "incomplete'
        extracted = TextExtractor._extract_from_json(invalid_content)
        self.assertEqual(extracted, invalid_content.decode('utf-8'))

    @patch('app.mapper.text_extractor.pytesseract.image_to_string', return_value=MOCK_IMAGE_TEXT)
    @patch('app.mapper.text_extractor.Image.open')
    def test_extract_from_image(self, mock_image_open, mock_tesseract):
        """
        Test extraction from an image file.

        This test verifies that the method correctly extracts text from an image
        using an image processing library (e.g., pytesseract). It checks that
        the image is processed and the correct text is returned.
        """
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image

        content = b'mock image bytes'
        extracted = TextExtractor._extract_from_image(content)

        mock_image_open.assert_called_once()
        mock_tesseract.assert_called_once_with(mock_image, lang='eng')
        self.assertEqual(extracted, MOCK_IMAGE_TEXT)

    def test_extract_from_image_no_text(self):
        """
        Test extraction from an image with no text.

        This test ensures that the method handles cases where no text can be
        extracted from the image and returns a warning message.
        """
        with patch('app.mapper.text_extractor.pytesseract.image_to_string', return_value=" \n \n "):
            with patch('app.mapper.text_extractor.Image.open'):
                content = b'mock image bytes'
                extracted = TextExtractor._extract_from_image(content)
                self.assertIn("Warning: No text could be extracted from image", extracted)


if __name__ == "__main__":
    unittest.main()