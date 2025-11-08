from typing import Dict, Any, Optional
from django.core.files.uploadedfile import UploadedFile
from app.mapper.text_extractor import TextExtractor
from datetime import timezone

class DocumentProcessingService:
    """
    Service responsible for orchestrating document processing workflows.

    This service handles the complete pipeline from file upload to text extraction,
    coordinating between different components like text extractors, validators, etc.
    """

    def __init__(self):
        """Initialize the service with required extractors and validators."""
        self.text_extractor = TextExtractor()
        self.supported_extensions = {
            '.pdf', '.docx', '.json', '.txt', '.text',
            '.jpg', '.jpeg', '.png'
        }
        self.max_file_size = 10 * 1024 * 1024

    def process_uploaded_file(
            self,
            uploaded_file: UploadedFile,
            user_id: Optional[str] = "guest",
            doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process an uploaded file through the complete extraction pipeline.

        Workflow:
            1. Validates file presence and basic properties
            2. Reads file content from memory
            3. Validates file size and extension
            4. Prepares metadata
            5. Extracts text using TextExtractor
            6. Returns structured result

        Args:
            uploaded_file: Django UploadedFile object from request.FILES
            user_id: User identifier (defaults to "guest")
            doc_id: Document identifier (auto-generated if not provided)

        Returns:
            Dict containing:
                - success: bool indicating if processing succeeded
                - data: extracted text and metadata (if successful)
                - error: error message (if failed)
                - status_code: suggested HTTP status code
        """
        if not uploaded_file:
            return self._error_response(
                "No file received",
                status_code=400
            )

        try:
            file_content = uploaded_file.read()
            file_name = uploaded_file.name
        except Exception as e:
            return self._error_response(
                f"Error reading file: {str(e)}",
                status_code=400
            )

        validation_result = self._validate_file(
            file_name=file_name,
            file_content=file_content
        )
        if not validation_result["valid"]:
            return self._error_response(
                validation_result["error"],
                status_code=400
            )

        extension = self._get_file_extension(file_name)

        metadata = self._prepare_metadata(
            user_id=user_id,
            doc_id=doc_id,
            file_name=file_name,
            file_size=len(file_content)
        )

        extraction_result = self.text_extractor.extract_text(
            file_path=file_name,
            metadata=metadata,
            file_content=file_content,
            extension=extension
        )

        if "error" in extraction_result:
            return self._error_response(
                extraction_result["error"],
                status_code=500,
                details=extraction_result
            )

        return self._success_response(extraction_result)

    def _validate_file(
            self,
            file_name: str,
            file_content: bytes
    ) -> Dict[str, Any]:
        """
        Validate file properties like size and extension.

        Args:
            file_name: Name of the uploaded file
            file_content: Binary content of the file

        Returns:
            Dict with 'valid' boolean and optional 'error' message
        """
        file_size = len(file_content)
        if file_size > self.max_file_size:
            return {
                "valid": False,
                "error": f"File too large. Maximum size: {self.max_file_size / (1024 * 1024):.1f} MB"
            }

        if file_size == 0:
            return {
                "valid": False,
                "error": "File is empty"
            }

        # Check extension
        extension = self._get_file_extension(file_name)
        if not extension:
            return {
                "valid": False,
                "error": "File has no extension"
            }

        if extension not in self.supported_extensions:
            return {
                "valid": False,
                "error": f"Unsupported file type: {extension}. Supported: {', '.join(sorted(self.supported_extensions))}"
            }

        return {"valid": True}

    @staticmethod
    def _get_file_extension(file_name: str) -> str:
        """Extract and normalize file extension."""
        if '.' in file_name:
            return '.' + file_name.split('.')[-1].lower()
        return ''

    @staticmethod
    def _prepare_metadata(
            user_id: str,
            doc_id: Optional[str],
            file_name: str,
            file_size: int
    ) -> Dict[str, Any]:
        """
        Prepare metadata dictionary for document processing.

        Args:
            user_id: User identifier
            doc_id: Document identifier (generated if None)
            file_name: Original filename
            file_size: Size in bytes

        Returns:
            Metadata dictionary
        """
        import uuid
        from datetime import datetime

        return {
            "doc_id": doc_id or f"doc_{uuid.uuid4().hex[:12]}",
            "user_id": user_id,
            "original_filename": file_name,
            "file_size": file_size,
            "processed_at": datetime.now(timezone.utc).isoformat()
        }

    @staticmethod
    def _success_response(data: Dict[str, Any]) -> Dict[str, Any]:
        """Format a successful response."""
        return {
            "success": True,
            "data": data,
            "status_code": 200
        }

    @staticmethod
    def _error_response(
            error_message: str,
            status_code: int = 400,
            details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format an error response."""
        response: Dict[str, Any] = {
            "success": False,
            "error": error_message,
            "status_code": status_code
        }
        if details:
            response["details"] = details
        return response