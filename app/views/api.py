from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from ..services.document_processing_service import DocumentProcessingService

@csrf_exempt
def extract_view(request):
    """
    Endpoint to extract text from a file sent via POST.

    This view processes the uploaded file directly from memory without saving it to disk.

    Workflow:
        1. Checks that the request method is POST.
        2. Reads the uploaded file content and determines its extension.
        3. Prepares metadata including a demo document ID, user ID, original filename, and file size.
        4. Uses TextExtractor to extract text from the file content.
        5. Returns a JSON response with the extracted text or an error message.

    Args:
        request (HttpRequest): The incoming HTTP request object.

    Returns:
        JsonResponse: A JSON object containing either the extracted text or an error message.
            The HTTP status code is determined as follows:
            - 200: Extraction succeeded
            - 400: Bad request (missing file or reading error)
            - 405: Method not allowed (non-POST request)
            - 500: Internal extraction error
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST method only allowed"}, status=405)

    uploaded_file = request.FILES.get("file")

    service = DocumentProcessingService()

    result = service.process_uploaded_file(
        uploaded_file=uploaded_file,
        user_id="guest",
        doc_id=None
    )

    status_code = result.pop("status_code", 200)

    return JsonResponse(result, status=status_code)